#!/bin/bash

# 1. Kiểm tra cài đặt
if ! command -v ngrok &> /dev/null; then
    echo "Ngrok chưa được cài đặt. Đang cài đặt qua Homebrew..."
    brew install ngrok/ngrok/ngrok
fi

# 2. Kiểm tra Authentication
echo "--- Kiểm tra trạng thái xác thực ---"
# Chạy thử lệnh config check hoặc đơn giản là prompt luôn nếu muốn chắc chắn
# Vì không có cách dễ để check auth state mà không chạy tunnel, ta sẽ hỏi user nếu họ muốn nhập token.

echo "Nếu bạn chưa nhập Token bao giờ, bạn cần nhập ngay bây giờ."
echo "Lấy token tại: https://dashboard.ngrok.com/get-started/your-authtoken"
read -p "Bạn có muốn nhập Ngrok Authtoken không? (y/n): " answer

if [[ "$answer" =~ ^[Yy]$ ]]; then
    read -p "Nhập Authtoken của bạn: " token
    ngrok config add-authtoken "$token"
    echo "Đã lưu token."
fi

# 3. Chạy Tunnel
echo "---"
echo "Đang khởi động Ngrok Tunnel ở cổng 8080..."
echo "Vui lòng COPY link 'Forwarding' (ví dụ: https://abcd.ngrok-free.app) hiện ra bên dưới."
echo "Sau đó dán vào Hugging Face Secrets (biến SCHOOL_API_URL)."
echo "---"

ngrok http 8080
