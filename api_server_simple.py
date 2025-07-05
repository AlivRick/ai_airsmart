#!/usr/bin/env python3
"""
API Server cho PDF Chatbot sử dụng FastAPI
Cho phép các ứng dụng khác gọi API để trả lời câu hỏi
Version đơn giản sử dụng requests để gọi Gemini API
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import tempfile
import shutil
import requests
import json
from typing import Optional
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Import các thư viện LangChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class SimplePDFChatbot:
    """
    Lớp PDFChatbot đơn giản sử dụng requests để gọi Gemini API
    """
    
    def __init__(self, pdf_path: str):
        """
        Khởi tạo chatbot với đường dẫn file PDF
        """
        # Load biến môi trường
        load_dotenv()
        
        # Kiểm tra API key
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY không tìm thấy trong file .env")
        
        # Cấu hình Gemini API
        self.model_name = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.gemini_api_key}"
        
        self.pdf_path = pdf_path
        self.text_chunks = []
        self.vectorizer = None
        self.tfidf_matrix = None
        
        # Khởi tạo các thành phần
        self._setup_components()
    
    def _setup_components(self):
        """
        Thiết lập các thành phần của chatbot
        """
        print("🔄 Đang khởi tạo chatbot với Gemini...")
        
        # 1. Đọc file PDF
        print("📖 Đang đọc file PDF...")
        documents = self._load_pdf()
        
        # 2. Chia nhỏ văn bản
        print("✂️ Đang chia nhỏ văn bản...")
        self.text_chunks = self._split_text(documents)
        
        # 3. Tạo vector embeddings với TF-IDF
        print("🔢 Đang tạo vector embeddings với TF-IDF...")
        self._create_tfidf_vectors()
        
        print("✅ Chatbot đã sẵn sàng!")
    
    def _load_pdf(self):
        """
        Đọc nội dung từ file PDF
        """
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"File PDF không tìm thấy: {self.pdf_path}")
        
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        
        print(f"📄 Đã đọc {len(documents)} trang từ file PDF")
        return documents
    
    def _split_text(self, documents):
        """
        Chia nhỏ văn bản thành các chunk
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        text_chunks = text_splitter.split_documents(documents)
        print(f"✂️ Đã chia thành {len(text_chunks)} chunks")
        return text_chunks
    
    def _create_tfidf_vectors(self):
        """
        Tạo TF-IDF vectors từ text chunks
        """
        texts = [chunk.page_content for chunk in self.text_chunks]
        
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        print(f"🔢 Đã tạo TF-IDF vectors với {self.tfidf_matrix.shape[1]} features")
    
    def _retrieve_relevant_chunks(self, question: str, k: int = 3):
        """
        Truy xuất các chunks có liên quan đến câu hỏi
        """
        question_vector = self.vectorizer.transform([question])
        similarities = cosine_similarity(question_vector, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-k:][::-1]
        
        relevant_chunks = []
        for idx in top_indices:
            if similarities[idx] > 0.1:
                relevant_chunks.append(self.text_chunks[idx].page_content)
        
        return relevant_chunks
    
    def _call_gemini_api(self, prompt: str) -> str:
        """
        Gọi Gemini API để tạo câu trả lời
        """
        try:
            headers = {
                'Content-Type': 'application/json',
            }
            
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.3,
                    "topK": 1,
                    "topP": 1,
                    "maxOutputTokens": 2048,
                }
            }
            
            response = requests.post(self.api_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                if 'candidates' in data and len(data['candidates']) > 0:
                    return data['candidates'][0]['content']['parts'][0]['text']
                else:
                    return "Xin lỗi, không thể tạo câu trả lời."
            else:
                return f"Lỗi API: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Lỗi khi gọi Gemini API: {str(e)}"
    
    def ask_question(self, question: str) -> dict:
        """
        Đặt câu hỏi cho chatbot
        """
        try:
            relevant_chunks = self._retrieve_relevant_chunks(question)
            
            if not relevant_chunks:
                return {
                    "answer": "Tôi không tìm thấy thông tin liên quan trong tài liệu để trả lời câu hỏi này.",
                    "source_count": 0
                }
            
            context = "\n\n".join(relevant_chunks)
            
            prompt = f"""Dựa trên thông tin sau đây từ tài liệu PDF, hãy trả lời câu hỏi bằng tiếng Việt một cách chính xác và chi tiết.
Nếu thông tin không đủ để trả lời, hãy nói rằng bạn không tìm thấy thông tin cần thiết trong tài liệu.

THÔNG TIN TỪ TÀI LIỆU:
{context}

CÂU HỎI: {question}

Hãy trả lời một cách rõ ràng và dựa trên thông tin được cung cấp:"""
            
            answer = self._call_gemini_api(prompt)
            
            return {
                "answer": answer,
                "source_count": len(relevant_chunks)
            }
            
        except Exception as e:
            return {
                "answer": f"Xin lỗi, có lỗi xảy ra khi xử lý câu hỏi: {str(e)}",
                "source_count": 0
            }

app = FastAPI(
    title="PDF Chatbot API",
    description="API để trả lời câu hỏi dựa trên nội dung PDF sử dụng Gemini AI",
    version="1.0.0"
)

# Cho phép CORS
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

@app.get("/")
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

@app.get("/health")
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

@app.post("/load-pdf", response_model=StatusResponse)
async def load_existing_pdf(pdf_path: str):
    """
    Load file PDF đã có sẵn trong hệ thống
    """
    global chatbot, current_pdf_path
    
    try:
        if not os.path.exists(pdf_path):
            raise HTTPException(status_code=404, detail="File PDF không tồn tại")
        
        chatbot = SimplePDFChatbot(pdf_path)
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
        result = chatbot.ask_question(request.question)
        
        return QuestionResponse(
            answer=result["answer"],
            source_count=result["source_count"],
            status="success"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý câu hỏi: {str(e)}")

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════╗
    ║              🚀 PDF CHATBOT API SERVER           ║
    ║                                                   ║
    ║  API Endpoints:                                   ║
    ║  • POST /load-pdf - Load file PDF có sẵn         ║
    ║  • POST /ask - Đặt câu hỏi                       ║
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
            chatbot = SimplePDFChatbot(default_pdf)
            current_pdf_path = default_pdf
            print(f"✅ Đã load file PDF mặc định: {default_pdf}")
        except Exception as e:
            print(f"⚠️ Không thể load file PDF mặc định: {e}")
    
    # Chạy server
    uvicorn.run(app, host="0.0.0.0", port=8000)