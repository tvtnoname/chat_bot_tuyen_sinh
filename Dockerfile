# Sử dụng Python 3.10
FROM python:3.10

# Tạo user mới để không chạy dưới quyền root (Yêu cầu của Hugging Face)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Thiết lập thư mục làm việc
WORKDIR /app

# Copy file requirements và cài đặt thư viện
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn vào
COPY --chown=user . .

# Mở port 7860 (Port mặc định của Hugging Face Spaces)
EXPOSE 7860

# Chạy ứng dụng
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
