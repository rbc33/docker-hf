Here is the updated tutorial in English, using `hf auth` commands for Step 3.

---

# Tutorial: Deploying an OpenAI-Compatible API on Hugging Face Spaces (Docker)

This guide will show you how to deploy a custom Large Language Model (LLM) on Hugging Face Spaces using Docker. The result will be an API that behaves exactly like OpenAI's API (`/v1/chat/completions`).

## Prerequisites

1.  A **Hugging Face Account**.
2.  A **Access Token** with **WRITE** permissions (Get it [here](https://huggingface.co/settings/tokens)).
3.  Python installed on your local machine.

---

## Step 1: Create the Space

1.  Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2.  **Space Name:** Enter a name (e.g., `my-custom-api`).
3.  **License:** Choose MIT or Apache 2.0.
4.  **Select the Space SDK:** Choose **Docker**.
5.  **Space Hardware:** Select **CPU Basic** (Free) for small models, or a GPU if you plan to run larger models.
6.  Click **Create Space**.

---

## Step 2: Prepare Your Local Files

Create a folder on your computer (e.g., `hf-api`) and create the following **4 files** inside it.

### 1. `requirements.txt`
```text
fastapi
uvicorn
pydantic
transformers
torch
accelerate
```

### 2. `Dockerfile`
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Create a non-root user for security (recommended by HF)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . .

# Hugging Face Spaces expects the application to run on port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
```

### 3. `main.py`
This code loads a lightweight model (`Qwen/Qwen2.5-0.5B-Instruct`) suitable for CPU.

```python
import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Literal
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn

app = FastAPI(title="OpenAI-compatible API")

# --- Model Configuration ---
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model: {MODEL_ID} on {device}...")

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
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise e

# --- OpenAI Schema ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "default-model"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

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

# --- API Endpoint ---
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    try:
        input_text = tokenizer.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in request.messages],
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True
            )

        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        return ChatCompletionResponse(
            id="chatcmpl-123",
            created=1234567890,
            model=MODEL_ID,
            choices=[Choice(index=0, message=ChatMessage(role="assistant", content=response_text), finish_reason="stop")],
            usage=Usage(prompt_tokens=len(inputs.input_ids[0]), completion_tokens=len(generated_ids), total_tokens=len(inputs.input_ids[0])+len(generated_ids))
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
```

### 4. `README.md`
```markdown
---
title: OpenAI Like API
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---
```

---

## Step 3: Deploy using `hf` CLI

Open your terminal in the folder containing your files.

1.  **Login with Authentication:**
    Run the command below. When prompted for a token, paste your **WRITE** token (input will be invisible).
    ```bash
    hf auth login
    ```

2.  **Upload the files:**
    Replace `YOUR_USERNAME/YOUR_SPACE_NAME` with your actual space details (e.g., `rbc33/docker-hf`).
    ```bash
    hf upload YOUR_USERNAME/YOUR_SPACE_NAME . . --repo-type space
    ```

---

## Step 4: Test Your API

Once the upload finishes, go to your Space URL and wait for the status to change to **Running**.

Your API endpoint is: `https://YOUR_USERNAME-YOUR_SPACE_NAME.hf.space/v1/chat/completions`

### Python Example

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://YOUR_USERNAME-YOUR_SPACE_NAME.hf.space/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="local-model",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)
```