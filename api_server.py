#!/usr/bin/env python3
"""
API Server cho PDF Chatbot sử dụng FastAPI
Cho phép các ứng dụng khác gọi API để trả lời câu hỏi
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import tempfile
import shutil
from typing import Optional
import json

# Import chatbot class
from pdf_chatbot_gemini import GeminiPDFChatbot

app = FastAPI(
    title="PDF Chatbot API",
    description="API để trả lời câu hỏi dựa trên nội dung PDF sử dụng Gemini AI",
    version="1.0.0"
)

# Cho phép CORS để frontend có thể gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global chatbot instance
chatbot = None
current_pdf_path = None

class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str
    source_count: int
    status: str

class StatusResponse(BaseModel):
    status: str
    message: str
    pdf_loaded: bool
    pdf_path: Optional[str] = None
    chunks_count: Optional[int] = None

@app.get("/", response_model=dict)
async def root():
    """
    Endpoint gốc để kiểm tra API
    """
    return {
        "message": "PDF Chatbot API đang hoạt động",
        "version": "1.0.0",
        "endpoints": {
            "POST /upload-pdf": "Upload file PDF",
            "POST /ask": "Đặt câu hỏi",
            "GET /status": "Kiểm tra trạng thái",
            "GET /health": "Health check"
        }
    }

@app.get("/health", response_model=dict)
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "message": "API đang hoạt động bình thường"}

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """
    Kiểm tra trạng thái của chatbot
    """
    global chatbot, current_pdf_path
    
    if chatbot is None:
        return StatusResponse(
            status="not_initialized",
            message="Chatbot chưa được khởi tạo. Hãy upload file PDF trước.",
            pdf_loaded=False
        )
    
    return StatusResponse(
        status="ready",
        message="Chatbot đã sẵn sàng trả lời câu hỏi",
        pdf_loaded=True,
        pdf_path=current_pdf_path,
        chunks_count=len(chatbot.text_chunks) if chatbot.text_chunks else 0
    )

@app.post("/upload-pdf", response_model=StatusResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload file PDF và khởi tạo chatbot
    """
    global chatbot, current_pdf_path
    
    try:
        # Kiểm tra file extension
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File phải có định dạng PDF")
        
        # Tạo thư mục tạm để lưu file
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, file.filename)
        
        # Lưu file PDF
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Khởi tạo chatbot với file PDF mới
        chatbot = GeminiPDFChatbot(temp_file_path)
        current_pdf_path = file.filename
        
        return StatusResponse(
            status="success",
            message=f"Đã upload và xử lý file {file.filename} thành công",
            pdf_loaded=True,
            pdf_path=current_pdf_path,
            chunks_count=len(chatbot.text_chunks)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý file PDF: {str(e)}")

@app.post("/load-pdf", response_model=StatusResponse)
async def load_existing_pdf(pdf_path: str):
    """
    Load file PDF đã có sẵn trong hệ thống
    """
    global chatbot, current_pdf_path
    
    try:
        if not os.path.exists(pdf_path):
            raise HTTPException(status_code=404, detail="File PDF không tồn tại")
        
        # Khởi tạo chatbot với file PDF
        chatbot = GeminiPDFChatbot(pdf_path)
        current_pdf_path = pdf_path
        
        return StatusResponse(
            status="success",
            message=f"Đã load file {pdf_path} thành công",
            pdf_loaded=True,
            pdf_path=current_pdf_path,
            chunks_count=len(chatbot.text_chunks)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi load file PDF: {str(e)}")

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Đặt câu hỏi cho chatbot
    """
    global chatbot
    
    if chatbot is None:
        raise HTTPException(
            status_code=400, 
            detail="Chatbot chưa được khởi tạo. Hãy upload file PDF trước."
        )
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống")
    
    try:
        # Gọi chatbot để trả lời
        result = chatbot.ask_question(request.question)
        
        return QuestionResponse(
            answer=result["answer"],
            source_count=result["source_count"],
            status="success"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý câu hỏi: {str(e)}")

@app.post("/ask-batch", response_model=list)
async def ask_multiple_questions(questions: list[str]):
    """
    Đặt nhiều câu hỏi cùng lúc
    """
    global chatbot
    
    if chatbot is None:
        raise HTTPException(
            status_code=400, 
            detail="Chatbot chưa được khởi tạo. Hãy upload file PDF trước."
        )
    
    if not questions:
        raise HTTPException(status_code=400, detail="Danh sách câu hỏi không được để trống")
    
    try:
        results = []
        for question in questions:
            if question.strip():
                result = chatbot.ask_question(question)
                results.append({
                    "question": question,
                    "answer": result["answer"],
                    "source_count": result["source_count"],
                    "status": "success"
                })
            else:
                results.append({
                    "question": question,
                    "answer": "Câu hỏi không hợp lệ",
                    "source_count": 0,
                    "status": "error"
                })
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý câu hỏi: {str(e)}")

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════╗
    ║              🚀 PDF CHATBOT API SERVER           ║
    ║                                                   ║
    ║  API Endpoints:                                   ║
    ║  • POST /upload-pdf - Upload file PDF            ║
    ║  • POST /load-pdf - Load file PDF có sẵn         ║
    ║  • POST /ask - Đặt câu hỏi                       ║
    ║  • POST /ask-batch - Đặt nhiều câu hỏi           ║
    ║  • GET /status - Kiểm tra trạng thái             ║
    ║  • GET /health - Health check                    ║
    ║                                                   ║
    ║  Server sẽ chạy tại: http://localhost:8000       ║
    ║  API Docs tại: http://localhost:8000/docs        ║
    ╚═══════════════════════════════════════════════════╝
    """)
    
    # Load file PDF mặc định nếu có
    default_pdf = "sample.pdf"
    if os.path.exists(default_pdf):
        try:
            chatbot = GeminiPDFChatbot(default_pdf)
            current_pdf_path = default_pdf
            print(f"✅ Đã load file PDF mặc định: {default_pdf}")
        except Exception as e:
            print(f"⚠️ Không thể load file PDF mặc định: {e}")
    
    # Chạy server
    uvicorn.run(app, host="0.0.0.0", port=8000)