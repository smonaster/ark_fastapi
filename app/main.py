from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os
import gc
from dotenv import load_dotenv

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
    # Strong generalist, long context, requires token
    "llama31-8b": {
        "id": "meta-llama/Llama-3.1-8B-Instruct",
        "needs_token": True,
        "quant": None,   # FP16 fits in 24 GB
    },

    # Good for code/reasoning, does not require token
    "qwen25-7b": {
        "id": "Qwen/Qwen2.5-7B-Instruct",
        "needs_token": False,
        "quant": None,   # FP16 ok
    },

    # Reasoning model (distill of DeepSeek-R1), requires token
    "deepseek-r1-qwen-7b": {
        "id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "needs_token": True,
        "quant": None,   # you can switch to 4-bit if needed
    },

    # Slightly larger model (12B), better load it quantized
    "mistral-nemo-12b": {
        "id": "mistralai/Mistral-Nemo-Instruct-2407",
        "needs_token": True,
        "quant": 4,      # 4-bit to be comfortable in 24 GB
    },
}

# Cache of loaded models
loaded_models = {}          # alias -> { "model": ..., "tokenizer": ... }
active_model_name = None
active_model = None         # reference to active AutoModelForCausalLM
tokenizer = None            # active tokenizer

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def uses_instruct_format(model_name: str) -> bool:
    """
    Fallback to instruct style [INST] ... [/INST] when chat template is unavailable.
    """
    return model_name in AVAILABLE_MODELS.keys()


def build_chat_prompt(user_message: str) -> str:
    """
    Build the prompt using the tokenizer chat template when available.
    Falls back to instruct format or plain text otherwise.
    """
    global tokenizer, active_model_name

    if tokenizer is None:
        raise RuntimeError("Tokenizer not initialized.")

    text = user_message.strip()

    # Single user turn for now
    messages = [
        {"role": "user", "content": text}
    ]

    # 1) Prefer tokenizer.apply_chat_template for instruct/chat models
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return prompt
        except Exception:
            # If something fails, fallback
            pass

    # 2) Manual instruct format fallback
    if active_model_name is not None and uses_instruct_format(active_model_name):
        return f"<s>[INST] {text} [/INST]"

    # 3) Last resort: plain prompt
    return text


# ----------------------------------------------------------------------
# Request/response schemas
# ----------------------------------------------------------------------
class ModelSelection(BaseModel):
    name: str

from typing import Optional

class PredictionRequest(BaseModel):
    message: str
    max_tokens: int = 300              # maximum response tokens
    temperature: float = 0.0           # 0 => deterministic, >0 => sampling
    top_p: Optional[float] = None      # nucleus sampling (optional)
    top_k: Optional[int] = None        # top-k sampling (optional)


# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------
@app.post("/select-model")
def select_model(selection: ModelSelection):
    """
    Download (if needed) and set a model as active.
    """
    global active_model_name, active_model, tokenizer, loaded_models

    name = selection.name

    if name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail=f"Model '{name}' is not available.")

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

        local_tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=auth_token,
            trust_remote_code=True
        )

        if quant == 4:
            quant_config = BitsAndBytesConfig(load_in_4bit=True)
            local_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                quantization_config=quant_config,
                token=auth_token,
                trust_remote_code=True
            )
        else:
            local_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                dtype=torch.float16,
                token=auth_token,
                trust_remote_code=True
            )

        # Asegurar pad_token_id para generación estable
        if getattr(local_model.config, "pad_token_id", None) is None:
            local_model.config.pad_token_id = local_model.config.eos_token_id

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
    """
    Return the current active model name.
    """
    if active_model_name is None:
        raise HTTPException(status_code=404, detail="No model is loaded.")
    return {"active_model": active_model_name}


@app.post("/unload-model")
def unload_model():
    """
    Unload the active model from memory.
    """
    global active_model_name, active_model, tokenizer, loaded_models

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

    return {"message": "Modelo descargado de memoria correctamente."}


@app.post("/predict")
def predict(request: PredictionRequest):
    """
    Generate a response using the active model.
    """
    global active_model_name, active_model, tokenizer

    if active_model_name is None or active_model is None or tokenizer is None:
        raise HTTPException(status_code=400, detail="No model is loaded.")

    # Build prompt for the active model
    prompt = build_chat_prompt(request.message)

    # Tokenize and move to the device where the model lives
    first_param = next(active_model.parameters(), None)
    if first_param is None:
        raise RuntimeError("Active model has no parameters; cannot infer device.")
    device = first_param.device

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Common generation arguments
    gen_kwargs = {
        "max_new_tokens": request.max_tokens,
        "pad_token_id": active_model.config.pad_token_id,
    }

    # Sampling vs deterministic
    if request.temperature > 0.0:
        # Stochastic sampling
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = request.temperature

        # Optional sampling params
        if request.top_p is not None:
            gen_kwargs["top_p"] = request.top_p
        if request.top_k is not None:
            gen_kwargs["top_k"] = request.top_k
    else:
        # Deterministic mode (greedy/beam by default)
        gen_kwargs["do_sample"] = False

    # Generation (no gradients)
    with torch.inference_mode():
        outputs = active_model.generate(
            **inputs,
            **gen_kwargs,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Trim instruct prompt if model echoes it
    if "[/INST]" in decoded:
        decoded = decoded.split("[/INST]", 1)[-1].strip()

    response_text = decoded.strip()

    return {"response": response_text}
