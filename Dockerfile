FROM python:3.9

# Create user with ID 1000 for Hugging Face Spaces
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
