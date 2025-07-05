#!/usr/bin/env python3
"""
Client example để test API Server
Demo cách gọi API từ ứng dụng khác
"""

import requests
import json
import time

API_BASE_URL = "http://localhost:8000"

def test_api_endpoints():
    """
    Test tất cả các endpoints của API
    """
    print("🔍 Testing PDF Chatbot API...")
    
    # 1. Health check
    print("\n1. Health Check:")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # 2. Kiểm tra trạng thái
    print("\n2. Status Check:")
    try:
        response = requests.get(f"{API_BASE_URL}/status")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    
    # 3. Load PDF có sẵn
    print("\n3. Load existing PDF:")
    try:
        response = requests.post(f"{API_BASE_URL}/load-pdf", params={"pdf_path": "sample.pdf"})
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Đợi một chút để chatbot khởi tạo
    print("\n⏳ Đợi chatbot khởi tạo...")
    time.sleep(3)
    
    # 4. Đặt câu hỏi đơn
    print("\n4. Ask single question:")
    try:
        question_data = {"question": "AI trong giáo dục có lợi ích gì?"}
        response = requests.post(f"{API_BASE_URL}/ask", json=question_data)
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Answer: {result['answer']}")
        print(f"Sources: {result['source_count']}")
    except Exception as e:
        print(f"Error: {e}")
    
    # 5. Đặt nhiều câu hỏi
    print("\n5. Ask multiple questions:")
    try:
        questions = [
            "Thách thức của AI trong giáo dục là gì?",
            "Tương lai của AI trong giáo dục như thế nào?",
            "AI có thể thay thế giáo viên không?"
        ]
        response = requests.post(f"{API_BASE_URL}/ask-batch", json=questions)
        print(f"Status: {response.status_code}")
        results = response.json()
        for i, result in enumerate(results):
            print(f"\nQ{i+1}: {result['question']}")
            print(f"A{i+1}: {result['answer'][:100]}...")
    except Exception as e:
        print(f"Error: {e}")

def interactive_client():
    """
    Client tương tác để chat với API
    """
    print("\n" + "="*60)
    print("💬 Interactive API Client - Nhập câu hỏi để chat với API")
    print("Gõ 'quit' để thoát")
    print("="*60)
    
    while True:
        try:
            question = input("\n🙋 Câu hỏi: ").strip()
            
            if question.lower() in ['quit', 'exit', 'thoát']:
                print("👋 Kết thúc chat!")
                break
                
            if not question:
                continue
            
            # Gửi request đến API
            question_data = {"question": question}
            response = requests.post(f"{API_BASE_URL}/ask", json=question_data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"\n🤖 Bot: {result['answer']}")
                print(f"📚 Nguồn: {result['source_count']} đoạn văn")
            else:
                error_detail = response.json().get('detail', 'Unknown error')
                print(f"\n❌ Lỗi: {error_detail}")
                
        except KeyboardInterrupt:
            print("\n👋 Kết thúc chat!")
            break
        except Exception as e:
            print(f"\n❌ Lỗi kết nối: {e}")

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════╗
    ║              📱 API CLIENT DEMO                   ║
    ║                                                   ║
    ║  Demo cách gọi PDF Chatbot API từ ứng dụng khác  ║
    ║                                                   ║
    ║  Chọn chế độ:                                     ║
    ║  1. Test tất cả endpoints                        ║
    ║  2. Chat tương tác                               ║
    ╚═══════════════════════════════════════════════════╝
    """)
    
    choice = input("Chọn chế độ (1 hoặc 2): ").strip()
    
    if choice == "1":
        test_api_endpoints()
    elif choice == "2":
        interactive_client()
    else:
        print("Lựa chọn không hợp lệ!")