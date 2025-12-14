from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os
import gc
from dotenv import load_dotenv

# ----------------------------------------------------------------------
# Carga variables de entorno (.env opcional)
# ----------------------------------------------------------------------
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN", None)

app = FastAPI()

# ----------------------------------------------------------------------
# Modelos disponibles (pensados para una 3090)
# ----------------------------------------------------------------------
MODELOS_DISPONIBLES = {
    # Buen generalista actual, contexto largo, requiere token
    "llama31-8b": {
        "id": "meta-llama/Llama-3.1-8B-Instruct",
        "needs_token": True,
        "quant": None,   # FP16, cabe bien en 24 GB
    },

    # Fuerte en código/razonamiento, no requiere token
    "qwen25-7b": {
        "id": "Qwen/Qwen2.5-7B-Instruct",
        "needs_token": False,
        "quant": None,   # FP16 ok
    },

    # Modelo de razonamiento (distill de DeepSeek-R1), requiere token
    "deepseek-r1-qwen-7b": {
        "id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "needs_token": True,
        "quant": None,   # puedes cambiar a 4-bit si quieres en el futuro
    },

    # Modelo algo más grande (12B), mejor cargarlo cuantizado
    "mistral-nemo-12b": {
        "id": "mistralai/Mistral-Nemo-Instruct-2407",
        "needs_token": True,
        "quant": 4,      # 4-bit para ir cómodo en 24 GB
    },
}

# Cache de modelos ya cargados
modelos = {}          # nombre -> { "model": ..., "tokenizer": ... }
modelo_activo = None
modelo_base = None    # referencia al modelo activo (AutoModelForCausalLM)
tokenizer = None      # tokenizer activo

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def usa_formato_instruct(nombre: str) -> bool:
    """
    Por si necesitamos un fallback manual tipo [INST] ... [/INST].
    La vía principal es usar apply_chat_template si existe.
    """
    return nombre in MODELOS_DISPONIBLES.keys()


def construir_prompt_chat(mensaje_usuario: str) -> str:
    """
    Construye el prompt usando el chat template del tokenizer si existe.
    Si no, hace fallback a formato instruct o texto plano.
    """
    global tokenizer, modelo_activo

    if tokenizer is None:
        raise RuntimeError("Tokenizer no inicializado.")

    texto = mensaje_usuario.strip()

    # Por ahora: un solo turno de usuario
    messages = [
        {"role": "user", "content": texto}
    ]

    # 1) Intentar usar apply_chat_template (recomendado en modelos instruct/chat)
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return prompt
        except Exception:
            # Si algo falla, seguimos con fallback
            pass

    # 2) Fallback a formato instruct manual
    if modelo_activo is not None and usa_formato_instruct(modelo_activo):
        return f"<s>[INST] {texto} [/INST]"

    # 3) Último recurso: prompt plano
    return texto


# ----------------------------------------------------------------------
# Esquemas de entrada/salida
# ----------------------------------------------------------------------
class ModeloInput(BaseModel):
    nombre: str


class Consulta(BaseModel):
    mensaje: str
    max_tokens: int = 300
    temperature: float = 0.0


# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------
@app.post("/select-model")
def select_model(input: ModeloInput):
    """
    Carga (si hace falta) e instancia un modelo como activo.
    """
    global modelo_activo, modelo_base, tokenizer, modelos

    nombre = input.nombre

    if nombre not in MODELOS_DISPONIBLES:
        raise HTTPException(status_code=404, detail=f"Modelo '{nombre}' no está disponible.")

    cfg = MODELOS_DISPONIBLES[nombre]
    modelo_id = cfg["id"]
    needs_token = cfg["needs_token"]
    quant = cfg["quant"]

    if needs_token and HF_TOKEN is None:
        raise HTTPException(
            status_code=401,
            detail=f"El modelo '{nombre}' requiere HF_TOKEN configurado en el entorno."
        )

    token_arg = HF_TOKEN if needs_token else None

    # Si el modelo no está aún en cache, descargarlo y cargarlo
    if nombre not in modelos:
        print(f"[INFO] Cargando modelo '{modelo_id}' (alias: {nombre}) ...")

        tokenizer_local = AutoTokenizer.from_pretrained(
            modelo_id,
            token=token_arg,
            trust_remote_code=True
        )

        if quant == 4:
            quant_config = BitsAndBytesConfig(load_in_4bit=True)
            modelo_local = AutoModelForCausalLM.from_pretrained(
                modelo_id,
                device_map="auto",
                quantization_config=quant_config,
                token=token_arg,
                trust_remote_code=True
            )
        else:
            modelo_local = AutoModelForCausalLM.from_pretrained(
                modelo_id,
                device_map="auto",
                torch_dtype=torch.float16,
                token=token_arg,
                trust_remote_code=True
            )

        # Asegurar pad_token_id para generación estable
        if getattr(modelo_local.config, "pad_token_id", None) is None:
            modelo_local.config.pad_token_id = modelo_local.config.eos_token_id

        modelo_local.eval()
        modelos[nombre] = {
            "model": modelo_local,
            "tokenizer": tokenizer_local,
        }
        gc.collect()

    # Activar como modelo actual
    modelo_activo = nombre
    modelo_base = modelos[nombre]["model"]
    tokenizer = modelos[nombre]["tokenizer"]

    return {"message": f"Modelo '{nombre}' seleccionado y listo."}


@app.get("/model-status")
def model_status():
    """
    Devuelve el nombre del modelo actualmente activo.
    """
    if modelo_activo is None:
        raise HTTPException(status_code=404, detail="No hay modelo cargado.")
    return {"modelo_activo": modelo_activo}


@app.post("/unload-model")
def unload_model():
    """
    Descarga (desinstancia) el modelo activo de memoria.
    """
    global modelo_activo, modelo_base, tokenizer, modelos

    if modelo_activo is None:
        raise HTTPException(status_code=400, detail="No hay modelo activo para descargar.")

    print(f"[INFO] Descargando modelo '{modelo_activo}' de memoria...")

    try:
        del modelos[modelo_activo]
    except KeyError:
        pass

    modelo_activo = None
    modelo_base = None
    tokenizer = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"message": "Modelo descargado de memoria correctamente."}


@app.post("/predict")
def predict(data: Consulta):
    """
    Genera una respuesta con el modelo activo.
    Esta será la interfaz que usará EconAgent (más adelante).
    """
    global modelo_activo, modelo_base, tokenizer

    if modelo_activo is None or modelo_base is None or tokenizer is None:
        raise HTTPException(status_code=400, detail="No hay modelo cargado.")

    # Construir prompt adecuado para el modelo
    prompt = construir_prompt_chat(data.mensaje)

    # Tokenizar y mover al dispositivo donde vive el modelo
    first_param = next(modelo_base.parameters(), None)
    if first_param is None:
        raise RuntimeError("El modelo activo no tiene parámetros, no se puede inferir el dispositivo.")
    device = first_param.device

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generación (sin gradientes)
    with torch.inference_mode():
        outputs = modelo_base.generate(
            **inputs,
            max_new_tokens=data.max_tokens,
            do_sample=(data.temperature > 0.0),
            temperature=data.temperature,
            pad_token_id=modelo_base.config.pad_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Limpieza ligera en caso de que el modelo devuelva parte del prompt instruct
    if "[/INST]" in decoded:
        decoded = decoded.split("[/INST]", 1)[-1].strip()

    respuesta = decoded.strip()

    return {"respuesta": respuesta}
