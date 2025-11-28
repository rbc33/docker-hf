import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn

app = FastAPI(title="OpenAI-compatible API")

# --- Configuración del Modelo ---
# Puedes cambiar esto por variables de entorno en HF Spaces
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Cargando modelo: {MODEL_ID} en {device}...")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True
    )
    if device == "cpu":
        model.to(device)
    print("¡Modelo cargado exitosamente!")
except Exception as e:
    print(f"Error cargando el modelo: {e}")
    raise e

# --- Modelos Pydantic (Formato OpenAI) ---

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "default-model"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False

class Choice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage

# --- Endpoint API ---

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    try:
        # 1. Aplicar el Chat Template (convierte la lista de mensajes en el string que entiende el modelo)
        # Esto hace que funcione con Llama, Mistral, Qwen, etc. automáticamente.
        input_text = tokenizer.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in request.messages],
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        # 2. Generar respuesta
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True
            )

        # 3. Decodificar solo la parte nueva (la respuesta)
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # 4. Calcular tokens (aproximado)
        prompt_tokens = len(inputs.input_ids[0])
        completion_tokens = len(generated_ids)

        # 5. Formatear como OpenAI
        return ChatCompletionResponse(
            id="chatcmpl-123",
            created=1234567890,
            model=MODEL_ID,
            choices=[
                Choice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )

    except Exception as e:
        print(f"Error en generación: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"message": "OpenAI-compatible API is running", "model": MODEL_ID}

# Permite ejecutar con `python main.py` para pruebas locales
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)