from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import asyncio
import os
import gc
from dotenv import load_dotenv
from typing import Optional, List, Dict

# ----------------------------------------------------------------------
# Load environment variables (.env optional)
# ----------------------------------------------------------------------
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN", None)

app = FastAPI()

# ----------------------------------------------------------------------
# Available models (targeting a 3090)
# ----------------------------------------------------------------------
AVAILABLE_MODELS = {
    # ------------------------------------------------------------------
    # NIVEL 1: VELOCIDAD MÁXIMA (FP16 Nativo)
    # ------------------------------------------------------------------
    "llama31-8b": {
        "id": "meta-llama/Llama-3.1-8B-Instruct",
        "needs_token": True,
        "quant": None,   # ~16GB VRAM.
    },
    "qwen25-7b": {
        "id": "Qwen/Qwen2.5-7B-Instruct",
        "needs_token": False,
        "quant": None,   # ~15GB VRAM.
    },
    "deepseek-r1-qwen-7b": {
        "id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "needs_token": False,
        "quant": None,   # ~15GB VRAM.
    },

    # ------------------------------------------------------------------
    # NIVEL 2: PUNTO DULCE "SWEET SPOT" (Int8 - 8 bit)
    # ------------------------------------------------------------------
    "mistral-nemo-12b": {
        "id": "mistralai/Mistral-Nemo-Instruct-2407",
        "needs_token": True,
        "quant": 8,      # ~13GB VRAM. Mucho mejor que 4-bit, más seguro que FP16.
    },
    "gemma2-9b": {
        "id": "google/gemma-2-9b-it",
        "needs_token": True,
        "quant": None,   # Nota: Gemma 9B cabe en FP16 (~19GB), 
    },

    # ------------------------------------------------------------------
    # NIVEL 3: INTELIGENCIA SUPERIOR (4-bit en GPU)
    # ------------------------------------------------------------------
    "qwen25-32b": {
        "id": "Qwen/Qwen2.5-32B-Instruct",
        "needs_token": False,
        "quant": 4,      # ~18GB VRAM.
    },

    # ------------------------------------------------------------------
    # NIVEL 4: CLASE "TITAN" (Offloading a CPU RAM)
    # ------------------------------------------------------------------
    "llama31-70b": {
        "id": "meta-llama/Llama-3.1-70B-Instruct",
        "needs_token": True,
        "quant": 4,      # ~40GB Total.
    },
    "qwen25-72b": {
        "id": "Qwen/Qwen2.5-72B-Instruct",
        "needs_token": False,
        "quant": 4,      # ~42GB Total.
    }
}
# Cache of loaded models
loaded_models = {}          
active_model_name = None
active_model = None         
tokenizer = None            
# Global lock to serialize model operations and inference
model_lock = asyncio.Lock()

# ----------------------------------------------------------------------
# Request/response schemas
# ----------------------------------------------------------------------
class ModelSelection(BaseModel):
    name: str

class ChatMessage(BaseModel):
    role: str
    content: str

class PredictionRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int = 1024              # maximum response tokens
    temperature: float = 0.0           # 0 => deterministic, >0 => sampling
    top_p: Optional[float] = None      # nucleus sampling (optional)
    top_k: Optional[int] = None        # top-k sampling (optional)
    seed: Optional[int] = None         # optional seed for reproducible sampling


# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------
@app.post("/select-model")
async def select_model(selection: ModelSelection):
    """
    Download (if needed) and set a model as active.
    """
    global active_model_name, active_model, tokenizer, loaded_models

    async with model_lock:
        name = selection.name

        if name not in AVAILABLE_MODELS:
            raise HTTPException(status_code=404, detail=f"Model '{name}' is not available.")

        if active_model_name is not None and active_model_name != name:
                print(f"[INFO] Auto-unloading '{active_model_name}' to free VRAM...")
                
                # Eliminamos referencias del diccionario
                if active_model_name in loaded_models:
                    del loaded_models[active_model_name]
                
                # Eliminamos referencias globales
                del active_model
                del tokenizer
                
                # Forzamos limpieza profunda
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                # Reiniciamos variables de estado
                active_model = None
                tokenizer = None
                loaded_models = {}
                active_model_name = None

        model_cfg = AVAILABLE_MODELS[name]
        model_id = model_cfg["id"]
        needs_token = model_cfg["needs_token"]
        quant = model_cfg["quant"]

        if needs_token and HF_TOKEN is None:
            raise HTTPException(
                status_code=401,
                detail=f"Model '{name}' requires HF_TOKEN set in the environment."
            )

        auth_token = HF_TOKEN if needs_token else None

        # Download and cache if needed
        if name not in loaded_models:
            print(f"[INFO] Loading model '{model_id}' (alias: {name}) ...")

            # Load Tokenizer
            local_tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                token=auth_token,
                trust_remote_code=True
            )

            # Load Model (falta gestionar offload a RAM + quantización avanzada)
            if quant == 4:
                print(f"[INFO] Loading in 4-bit quantization...")
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                local_model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",
                    quantization_config=quant_config,
                    token=auth_token,
                    trust_remote_code=True
                )
            elif quant == 8:
                print(f"[INFO] Loading in 8-bit quantization...")
                quant_config = BitsAndBytesConfig(
                    load_in_8bit=True
                )
                local_model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",
                    quantization_config=quant_config,
                    token=auth_token,
                    trust_remote_code=True
                )
            else:
                print(f"[INFO] Loading in native FP16...")
                local_model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",
                    dtype=torch.float16,
                    token=auth_token,
                    trust_remote_code=True
                )

            # Ensure pad_token_id exists
            if local_model.config.pad_token_id is None:
                local_model.config.pad_token_id = local_tokenizer.eos_token_id

            local_model.eval()
            loaded_models[name] = {
                "model": local_model,
                "tokenizer": local_tokenizer,
            }
            gc.collect()

        # Set active model
        active_model_name = name
        active_model = loaded_models[name]["model"]
        tokenizer = loaded_models[name]["tokenizer"]

        return {"message": f"Model '{name}' selected and ready."}


@app.get("/model-status")
def model_status():
    if active_model_name is None:
        raise HTTPException(status_code=404, detail="No model is loaded.")
    return {"active_model": active_model_name}


@app.post("/unload-model")
async def unload_model():
    global active_model_name, active_model, tokenizer, loaded_models

    async with model_lock:
        if active_model_name is None:
            raise HTTPException(status_code=400, detail="No active model to unload.")

        print(f"[INFO] Unloading model '{active_model_name}' from memory...")

        try:
            del loaded_models[active_model_name]
        except KeyError:
            pass

        active_model_name = None
        active_model = None
        tokenizer = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {"message": "Model unloaded successfully."}


@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    Generate a response using the active model.
    STRICT MODE: Uses apply_chat_template with return_tensors="pt".
    """
    global active_model_name, active_model, tokenizer

    async with model_lock:
        if active_model_name is None or active_model is None or tokenizer is None:
            raise HTTPException(status_code=400, detail="No model is loaded.")

        # Per-request seeding for sampling scenarios
        if request.seed is not None:
            torch.manual_seed(request.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(request.seed)
        # Prepare conversation
        conversation = [m.dict() for m in request.messages]

        # Get device
        first_param = next(active_model.parameters(), None)
        device = first_param.device

        # Apply Chat Template -> DIRECT TO TENSORS
        try:
            input_ids = tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(device)
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Error applying chat template: {str(e)}"
            )
        attention_mask = torch.ones_like(input_ids)
        # Calculate input length for slicing later
        input_len = input_ids.shape[1]

        # Generation config
        gen_kwargs = {
            "max_new_tokens": request.max_tokens,
            "pad_token_id": active_model.config.pad_token_id,
            "do_sample": request.temperature > 0.0,
        }

        if request.temperature > 0.0:
            gen_kwargs["temperature"] = request.temperature
            if request.top_p: gen_kwargs["top_p"] = request.top_p
            if request.top_k: gen_kwargs["top_k"] = request.top_k

        # Generate
        with torch.inference_mode():
            outputs = active_model.generate(
                input_ids,
                attention_mask=attention_mask,
                **gen_kwargs
            )

        # Slice off the prompt (Input)
        generated_tokens = outputs[0][input_len:]

        # Decode only the response
        response_text = tokenizer.decode(generated_tokens, skip_special_tokens=False).strip() #falta gestionar special tokens

        return {"response": response_text}
