# ark_fastapi
API minimal en FastAPI que permite cargar modelos de texto de Hugging Face y exponerlos mediante un endpoint de inferencia ligero.

## Puesta en marcha
1. (Opcional) Define `HF_TOKEN` en `.env` si quieres usar modelos que lo requieran.
2. Instala dependencias: `pip install fastapi uvicorn[standard] transformers accelerate bitsandbytes python-dotenv`.
3. Arranca el servidor: `uvicorn app.main:app --host 0.0.0.0 --port 8000`.

## Endpoints clave
- `POST /select-model` cuerpo `{"nombre": "<alias>"}` para descargar y activar uno de los modelos declarados en `app/main.py`.
- `GET /model-status` devuelve qué modelo está activo.
- `POST /predict` cuerpo `{"mensaje": "...", "max_tokens": 300, "temperature": 0.0}` para generar texto con el modelo activo.
- `POST /unload-model` libera la memoria de la GPU/CPU.

Siempre debes seleccionar un modelo antes de llamar a `/predict`; la primera carga tomará unos minutos según el tamaño del modelo.
