import requests
import json

# URL của API (mặc định là localhost:8000)
API_URL = "http://localhost:8000/api/chat"

def test_chat(question):
    """Gửi câu hỏi đến Chatbot và in câu trả lời."""
    payload = {"question": question}
    headers = {"Content-Type": "application/json"}
    
    print(f"User: {question}")
    try:
        response = requests.post(API_URL, data=json.dumps(payload), headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            print(f"Bot: {data['answer']}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except requests.exceptions.ConnectionError:
        print("Error: Không thể kết nối đến server. Hãy chắc chắn rằng bạn đã chạy 'uvicorn main:app --reload'.")
    print("-" * 50)

if __name__ == "__main__":
    print("--- TEST CHATBOT TRUNG TÂM THĂNG LONG ---\n")
    
    # Danh sách các câu hỏi test
    test_questions = [
        "Trung tâm Thăng Long ở đâu?",
        "Thầy dạy Toán là ai?",
        "Học phí lớp 12 là bao nhiêu?",
        "Có khóa học nào cho lớp 9 không?",
        "Làm sao để nhận học bổng?"
    ]

    for q in test_questions:
        test_chat(q)
