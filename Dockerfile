# Sử dụng Python 3.10
FROM python:3.10

# Thiết lập thư mục làm việc
WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

# Cài đặt thư viện
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy toàn bộ mã nguồn vào
COPY ./app /code/app
COPY ./data /code/data

# Thiết lập biến môi trường mặc định 
# ENV MODEL_NAME="gemini-2.0-flash" 
# ENV DATABASE_URL="..."

# Tạo user non-root 
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Expose port 7860 (Hugging Face Spaces mặc định dùng port này)
EXPOSE 7860

# Lệnh chạy server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
