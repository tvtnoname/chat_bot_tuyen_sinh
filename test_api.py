import requests
import json

# URL của API trên Hugging Face Spaces
API_URL = "https://tvtnoname01-chatbot-tuyen-sinh.hf.space/api/chat"

def test_chat(question):
    print(f"User: {question}")
    try:
        response = requests.post(
            API_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps({"question": question})
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Bot: {result.get('answer', 'Không có câu trả lời')}")
        else:
            print(f"Lỗi: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"Lỗi kết nối: {e}")
    print("-" * 50)

if __name__ == "__main__":
    print("--- TEST CHATBOT (Hugging Face) ---")
    # Các câu hỏi mẫu để kiểm tra
    test_chat("Trung tâm Thăng Long ở đâu?")
    test_chat("Học phí lớp 12 là bao nhiêu?")
    test_chat("Làm sao để nhận học bổng?")
