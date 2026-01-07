from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
import torch
import asyncio
import time
import uuid
import os
import gc
from dotenv import load_dotenv
from typing import Optional, List, Dict, Union, Any, Tuple
import json

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
        "quant": None,
        "attn": "flash_attention_2",
    },
    "qwen25-7b": {
        "id": "Qwen/Qwen2.5-7B-Instruct",
        "needs_token": False,
        "quant": None,
        "attn": "flash_attention_2",
    },
    "deepseek-r1-qwen-7b": {
        "id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "needs_token": False,
        "quant": None,
        "attn": "sdpa",
    },

    # ------------------------------------------------------------------
    # NIVEL 2: PUNTO DULCE "SWEET SPOT" (Int8 - 8 bit)
    # ------------------------------------------------------------------
    "mistral-nemo-12b": {
        "id": "mistralai/Mistral-Nemo-Instruct-2407",
        "needs_token": True,
        "quant": 8,
        "attn": "sdpa",
    },
    "gemma2-9b": {
        "id": "google/gemma-2-9b-it",
        "needs_token": True,
        "quant": None,
        "attn": "sdpa"
    },

    # ------------------------------------------------------------------
    # NIVEL 3: INTELIGENCIA SUPERIOR (4-bit en GPU)
    # ------------------------------------------------------------------
    "qwen25-32b": {
        "id": "Qwen/Qwen2.5-32B-Instruct",
        "needs_token": False,
        "quant": 4,
        "attn": "flash_attention_2",
    },

    # ------------------------------------------------------------------
    # NIVEL 4: CLASE "TITAN" (Offloading a CPU RAM)
    # ------------------------------------------------------------------
    "llama31-70b": {
        "id": "meta-llama/Llama-3.1-70B-Instruct",
        "needs_token": True,
        "quant": 4,
        "attn": "flash_attention_2",
    },
    "qwen25-72b": {
        "id": "Qwen/Qwen2.5-72B-Instruct",
        "needs_token": False,
        "quant": 4,
        "attn": "flash_attention_2",
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
# Helper Class: Stop Sequences (NUEVO)
# ----------------------------------------------------------------------
class StringStoppingCriteria(StoppingCriteria):
    """
    Detiene la generación si detecta alguna de las cadenas de parada.
    """
    def __init__(self, tokenizer, stop_strings: List[str], prompt_length: int):
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores, **kwargs):
        # Decodificamos solo lo generado hasta ahora
        generated_tokens = input_ids[0][self.prompt_length:]
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        for stop_str in self.stop_strings:
            if text.endswith(stop_str):
                return True
        return False

# ----------------------------------------------------------------------
# Helpers (must be used under model_lock)
# ----------------------------------------------------------------------
async def _load_and_set_model(name: str):
    """
    Load a model (if needed) and set it as active. Expects caller to hold model_lock.
    """
    global active_model_name, active_model, tokenizer, loaded_models

    if name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail=f"Model '{name}' is not available.")

    if active_model_name is not None and active_model_name != name:
        print(f"[INFO] Auto-unloading '{active_model_name}' to free VRAM...")

        if active_model_name in loaded_models:
            del loaded_models[active_model_name]

        del active_model
        del tokenizer

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        active_model = None
        tokenizer = None
        loaded_models = {}
        active_model_name = None

    model_cfg = AVAILABLE_MODELS[name]
    model_id = model_cfg["id"]
    needs_token = model_cfg["needs_token"]
    quant = model_cfg["quant"]
    attn_impl = model_cfg.get("attn", "sdpa")

    if needs_token and HF_TOKEN is None:
        raise HTTPException(
            status_code=401,
            detail=f"Model '{name}' requires HF_TOKEN set in the environment."
        )

    auth_token = HF_TOKEN if needs_token else None

    if name not in loaded_models:
        print(f"[INFO] Loading model '{model_id}' (alias: {name}) ...")

        local_tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=auth_token,
            trust_remote_code=True
        )

        common_args = {
            "device_map": "auto",
            "token": auth_token,
            "trust_remote_code": True,
            "attn_implementation": attn_impl,
        }

        if quant == 4:
            print(f"[INFO] Loading in 4-bit quantization...")
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            local_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quant_config,
                **common_args
            )
        elif quant == 8:
            print(f"[INFO] Loading in 8-bit quantization...")
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            local_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quant_config,
                **common_args
            )
        else:
            print(f"[INFO] Loading in native FP16...")
            local_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch.float16,
                **common_args
            )

        if local_model.config.pad_token_id is None:
            local_model.config.pad_token_id = local_tokenizer.eos_token_id

        local_model.eval()
        loaded_models[name] = {
            "model": local_model,
            "tokenizer": local_tokenizer,
        }
        gc.collect()

    active_model_name = name
    active_model = loaded_models[name]["model"]
    tokenizer = loaded_models[name]["tokenizer"]


def _generate_chat_response(
    messages: List[Dict[str, str]],
    max_tokens: Optional[int],
    temperature: float,
    top_p: Optional[float],
    top_k: Optional[int],
    seed: Optional[int],
    stop: Optional[Union[str, List[str]]] = None # <--- NUEVO
):
    """
    Run generation using the active model. Expects caller to hold model_lock.
    Returns tuple (response_text, prompt_tokens, completion_tokens).
    """
    global active_model_name, active_model, tokenizer

    if active_model_name is None or active_model is None or tokenizer is None:
        raise HTTPException(status_code=400, detail="No model is loaded.")

    if seed is not None and temperature > 0:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    first_param = next(active_model.parameters(), None)
    device = first_param.device

    try:
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error applying chat template: {str(e)}"
        )

    attention_mask = torch.ones_like(input_ids)
    input_len = input_ids.shape[1]

    stopping_criteria = None
    stop_sequences = []
    
    if stop:
        if isinstance(stop, str):
            stop_sequences = [stop]
        else:
            stop_sequences = stop
        
        criteria = StringStoppingCriteria(tokenizer, stop_sequences, input_len)
        stopping_criteria = StoppingCriteriaList([criteria])

    gen_kwargs = {
        "max_new_tokens": max_tokens if max_tokens is not None else 1024,
        "pad_token_id": active_model.config.pad_token_id,
        "do_sample": temperature > 0.0,
    }

    if stopping_criteria:
        gen_kwargs["stopping_criteria"] = stopping_criteria

    if temperature > 0.0:
        gen_kwargs["temperature"] = temperature
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
        if top_k is not None:
            gen_kwargs["top_k"] = top_k

    with torch.inference_mode():
        outputs = active_model.generate(
            input_ids,
            attention_mask=attention_mask,
            **gen_kwargs
        )

    generated_tokens = outputs[0][input_len:]
    response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    if stop_sequences:
        for stop_str in stop_sequences:
            if response_text.endswith(stop_str):
                response_text = response_text[:-len(stop_str)]
                break # Cortamos solo la última coincidencia
    
    response_text = response_text.strip()

    prompt_tokens = int(input_len)
    completion_tokens = int(generated_tokens.shape[0])

    return response_text, prompt_tokens, completion_tokens


# -------------------------
# Structured output helpers (JSON Schema)
# -------------------------
try:
    import jsonschema  # type: ignore
except Exception:  # pragma: no cover
    jsonschema = None


def _extract_first_json_object(text: str) -> str:
    """Return the first JSON object substring found in text (balanced braces)."""
    if not isinstance(text, str):
        raise ValueError("Expected string")
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object start '{' found")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start: i + 1]
    raise ValueError("Unbalanced JSON braces")


def _minify_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _validate_jsonschema(obj: Any, schema: Dict[str, Any]) -> None:
    if jsonschema is None:
        raise RuntimeError("jsonschema is not installed on server. Install with: pip install jsonschema")
    jsonschema.validate(instance=obj, schema=schema)


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
    max_tokens: int = 1024
    temperature: float = 0.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    seed: Optional[int] = None

class OpenAIChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 1024
    temperature: float = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    seed: Optional[int] = None
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    # Optional: enforce structured outputs (server-side)
    # If provided, the server will extract the first JSON object from the model output,
    # validate it against the schema, and return minified JSON as the assistant content.
    json_schema: Optional[Dict[str, Any]] = None
    schema_max_retries: int = 2

    class Config:
        extra = "ignore" 


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
        await _load_and_set_model(selection.name)
        return {"message": f"Model '{selection.name}' selected and ready."}


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

@app.post("/v1/chat/completions")
async def chat_completions(request: OpenAIChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.
    """
    if request.stream:
        raise HTTPException(status_code=400, detail="stream=True is not supported.")

    async with model_lock:
        await _load_and_set_model(request.model)

        # If a JSON schema is provided, enforce: generate -> extract JSON -> validate -> minify.
        if request.json_schema is not None:
            msgs_base = [m.dict() for m in request.messages]
            max_retries = max(0, int(request.schema_max_retries or 0))
            last_err = ""
            last_text = ""
            prompt_tokens = 0
            completion_tokens = 0

            for k in range(max_retries + 1):
                msgs = list(msgs_base)
                if k > 0:
                    msgs.append({
                        "role": "user",
                        "content": "Reminder: Output ONLY a single JSON object that matches the provided schema. No markdown, no extra text."
                    })

                # IMPORTANT: when enforcing json_schema, ignore stop sequences to avoid truncating JSON mid-object.
                response_text, prompt_tokens, completion_tokens = _generate_chat_response(
                    msgs,
                    request.max_tokens,
                    request.temperature,
                    request.top_p,
                    request.top_k,
                    request.seed,
                    stop=None
                )
                last_text = response_text or ""
                try:
                    json_str = _extract_first_json_object(last_text)
                    obj = json.loads(json_str)
                    _validate_jsonschema(obj, request.json_schema)
                    response_text = _minify_json(obj)
                    last_err = ""
                    break
                except Exception as e:
                    last_err = str(e)
                    continue

            if last_err:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "structured_output_validation_failed",
                        "message": last_err,
                        "last_response_snippet": last_text[:500],
                    },
                )
        else:
            response_text, prompt_tokens, completion_tokens = _generate_chat_response(
                [m.dict() for m in request.messages],
                request.max_tokens,
                request.temperature,
                request.top_p,
                request.top_k,
                request.seed,
                stop=request.stop
            )

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "system_fingerprint": "ark-fastapi",
    }


@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    Generate a response using the active model.
    STRICT MODE: Uses apply_chat_template with return_tensors="pt".
    """
    global active_model_name, active_model, tokenizer

    async with model_lock:
        response_text, _, _ = _generate_chat_response(
            [m.dict() for m in request.messages],
            request.max_tokens,
            request.temperature,
            request.top_p,
            request.top_k,
            request.seed,
            stop=None # Endpoint legacy sin stop
        )

        return {"response": response_text}
