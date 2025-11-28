FROM python:3.10-slim

WORKDIR /app

# Crear usuario no-root por seguridad (recomendado en HF Spaces)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . .

# Hugging Face Spaces espera el puerto 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]