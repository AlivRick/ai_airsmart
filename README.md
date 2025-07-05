# PDF Chatbot API

API Server cho phép các ứng dụng khác gọi API để trả lời câu hỏi dựa trên nội dung PDF sử dụng Google Gemini AI.

## Tính năng

- ✅ Upload file PDF và xử lý tự động
- ✅ Trả lời câu hỏi dựa trên nội dung PDF
- ✅ Hỗ trợ đặt nhiều câu hỏi cùng lúc
- ✅ API RESTful với FastAPI
- ✅ Tự động tải file PDF mặc định
- ✅ CORS support cho frontend
- ✅ API documentation tự động

## Cài đặt

1. Cài đặt các thư viện cần thiết:
```bash
pip install fastapi uvicorn python-dotenv langchain-community langchain-text-splitters google-generativeai scikit-learn pypdf requests python-multipart
```

2. Tạo file `.env` với Gemini API key:
```
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-flash
```

## Chạy API Server

```bash
python api_server.py
```

Server sẽ chạy tại: `http://localhost:8000`
API Documentation: `http://localhost:8000/docs`

## API Endpoints

### 1. GET /health
Health check endpoint
```bash
curl http://localhost:8000/health
```

### 2. GET /status
Kiểm tra trạng thái chatbot
```bash
curl http://localhost:8000/status
```

### 3. POST /upload-pdf
Upload file PDF mới
```bash
curl -X POST "http://localhost:8000/upload-pdf" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_file.pdf"
```

### 4. POST /load-pdf
Load file PDF có sẵn trong hệ thống
```bash
curl -X POST "http://localhost:8000/load-pdf?pdf_path=sample.pdf" \
     -H "accept: application/json"
```

### 5. POST /ask
Đặt câu hỏi cho chatbot
```bash
curl -X POST "http://localhost:8000/ask" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{"question": "AI trong giáo dục có lợi ích gì?"}'
```

Response:
```json
{
  "answer": "Câu trả lời chi tiết...",
  "source_count": 3,
  "status": "success"
}
```

### 6. POST /ask-batch
Đặt nhiều câu hỏi cùng lúc
```bash
curl -X POST "http://localhost:8000/ask-batch" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '["Câu hỏi 1?", "Câu hỏi 2?", "Câu hỏi 3?"]'
```

## Cách sử dụng từ Python

```python
import requests

# Base URL
API_URL = "http://localhost:8000"

# 1. Kiểm tra trạng thái
response = requests.get(f"{API_URL}/status")
print(response.json())

# 2. Load PDF
response = requests.post(f"{API_URL}/load-pdf", 
                        params={"pdf_path": "sample.pdf"})
print(response.json())

# 3. Đặt câu hỏi
question_data = {"question": "AI trong giáo dục có lợi ích gì?"}
response = requests.post(f"{API_URL}/ask", json=question_data)
result = response.json()
print(f"Answer: {result['answer']}")
```

## Cách sử dụng từ JavaScript

```javascript
// Đặt câu hỏi
const askQuestion = async (question) => {
    const response = await fetch('http://localhost:8000/ask', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: question })
    });
    
    const result = await response.json();
    console.log('Answer:', result.answer);
    return result;
};

// Sử dụng
askQuestion("AI trong giáo dục có lợi ích gì?");
```

## Demo Client

Chạy client demo để test API:
```bash
python api_client_demo.py
```

Chọn chế độ:
- `1`: Test tất cả endpoints
- `2`: Chat tương tác với API

## Cấu trúc Project

```
/workspaces/project/ai/
├── .env                    # API keys
├── api_server.py          # API Server chính
├── api_client_demo.py     # Client demo
├── pdf_chatbot_gemini.py  # Chatbot core
├── sample.pdf             # File PDF mẫu
└── README.md              # Hướng dẫn này
```

## Error Handling

API sẽ trả về các mã lỗi HTTP chuẩn:
- `200`: Thành công
- `400`: Bad Request (câu hỏi trống, file không đúng định dạng)
- `404`: File không tìm thấy
- `500`: Lỗi server (lỗi API Gemini, lỗi xử lý PDF)

Example error response:
```json
{
  "detail": "Chatbot chưa được khởi tạo. Hãy upload file PDF trước."
}
```

## Tích hợp vào ứng dụng khác

### Frontend Web App
```html
<!DOCTYPE html>
<html>
<head>
    <title>PDF Chatbot</title>
</head>
<body>
    <div id="chat"></div>
    <input type="text" id="question" placeholder="Nhập câu hỏi...">
    <button onclick="askQuestion()">Gửi</button>

    <script>
        async function askQuestion() {
            const question = document.getElementById('question').value;
            const response = await fetch('http://localhost:8000/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: question })
            });
            const result = await response.json();
            document.getElementById('chat').innerHTML += 
                `<p><strong>Q:</strong> ${question}</p>
                 <p><strong>A:</strong> ${result.answer}</p>`;
        }
    </script>
</body>
</html>
```

### Mobile App (React Native)
```javascript
const chatWithPDF = async (question) => {
    try {
        const response = await fetch('http://your-server:8000/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question })
        });
        const result = await response.json();
        return result.answer;
    } catch (error) {
        console.error('Error:', error);
    }
};
```

## Deployment

### Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "api_server.py"]
```

### Requirements.txt
```
fastapi==0.104.1
uvicorn==0.24.0
python-dotenv==1.0.0
langchain-community==0.0.6
langchain-text-splitters==0.0.1
google-generativeai==0.3.2
scikit-learn==1.3.2
pypdf==3.17.1
requests==2.31.0
python-multipart==0.0.6
```

## Bảo mật

- Sử dụng HTTPS trong production
- Giới hạn file size upload
- Rate limiting cho API calls
- Validate input data
- Không expose API key trong response