#!/usr/bin/env python3
"""
PDF Chatbot sử dụng LangChain và Google Gemini API
Tác giả: GitHub Copilot
Mô tả: Chatbot có thể đọc file PDF và trả lời câu hỏi dựa trên nội dung file (sử dụng Gemini miễn phí)
"""

import os
import sys
import json
import requests
from typing import List
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Import các thư viện LangChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class GeminiPDFChatbot:
    """
    Lớp PDFChatbot sử dụng Google Gemini API để xử lý file PDF và trả lời câu hỏi
    """
    
    def __init__(self, pdf_path: str):
        """
        Khởi tạo chatbot với đường dẫn file PDF
        
        Args:
            pdf_path (str): Đường dẫn đến file PDF
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
        
        # 3. Tạo vector embeddings với TF-IDF (miễn phí)
        print("🔢 Đang tạo vector embeddings với TF-IDF...")
        self._create_tfidf_vectors()
        
        print("✅ Chatbot đã sẵn sàng!")
    
    def _load_pdf(self) -> List:
        """
        Đọc nội dung từ file PDF
        
        Returns:
            List: Danh sách các document từ PDF
        """
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"File PDF không tìm thấy: {self.pdf_path}")
        
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        
        print(f"📄 Đã đọc {len(documents)} trang từ file PDF")
        return documents
    
    def _split_text(self, documents: List) -> List:
        """
        Chia nhỏ văn bản thành các chunk
        
        Args:
            documents (List): Danh sách documents từ PDF
            
        Returns:
            List: Danh sách các text chunks
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,      # Kích thước mỗi chunk
            chunk_overlap=200,    # Độ chồng lấp giữa các chunk
            length_function=len,  # Hàm đo độ dài
        )
        
        text_chunks = text_splitter.split_documents(documents)
        print(f"✂️ Đã chia thành {len(text_chunks)} chunks")
        return text_chunks
    
    def _create_tfidf_vectors(self):
        """
        Tạo TF-IDF vectors từ text chunks (thay thế cho OpenAI embeddings)
        """
        # Lấy nội dung text từ chunks
        texts = [chunk.page_content for chunk in self.text_chunks]
        
        # Tạo TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Fit và transform texts
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        print(f"🔢 Đã tạo TF-IDF vectors với {self.tfidf_matrix.shape[1]} features")
    
    def _retrieve_relevant_chunks(self, question: str, k: int = 3) -> List[str]:
        """
        Truy xuất các chunks có liên quan đến câu hỏi
        
        Args:
            question (str): Câu hỏi của người dùng
            k (int): Số lượng chunks cần lấy
            
        Returns:
            List[str]: Danh sách các chunks liên quan
        """
        # Vectorize câu hỏi
        question_vector = self.vectorizer.transform([question])
        
        # Tính cosine similarity
        similarities = cosine_similarity(question_vector, self.tfidf_matrix).flatten()
        
        # Lấy k chunks có similarity cao nhất
        top_indices = similarities.argsort()[-k:][::-1]
        
        relevant_chunks = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Ngưỡng similarity tối thiểu
                relevant_chunks.append(self.text_chunks[idx].page_content)
        
        return relevant_chunks
    
    def _call_gemini_api(self, prompt: str) -> str:
        """
        Gọi Gemini API để tạo câu trả lời
        
        Args:
            prompt (str): Prompt để gửi đến Gemini
            
        Returns:
            str: Câu trả lời từ Gemini
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
        
        Args:
            question (str): Câu hỏi của người dùng
            
        Returns:
            dict: Kết quả bao gồm câu trả lời và nguồn tài liệu
        """
        try:
            # Truy xuất chunks liên quan
            relevant_chunks = self._retrieve_relevant_chunks(question)
            
            if not relevant_chunks:
                return {
                    "answer": "Tôi không tìm thấy thông tin liên quan trong tài liệu để trả lời câu hỏi này.",
                    "source_count": 0
                }
            
            # Tạo ngữ cảnh từ chunks
            context = "\n\n".join(relevant_chunks)
            
            # Tạo prompt cho Gemini
            prompt = f"""Dựa trên thông tin sau đây từ tài liệu PDF, hãy trả lời câu hỏi bằng tiếng Việt một cách chính xác và chi tiết.
Nếu thông tin không đủ để trả lời, hãy nói rằng bạn không tìm thấy thông tin cần thiết trong tài liệu.

THÔNG TIN TỪ TÀI LIỆU:
{context}

CÂU HỎI: {question}

Hãy trả lời một cách rõ ràng và dựa trên thông tin được cung cấp:"""
            
            # Gửi request đến Gemini
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
    
    def save_vectorstore(self, save_path: str):
        """
        Lưu vectorstore để sử dụng lại
        
        Args:
            save_path (str): Đường dẫn lưu vectorstore
        """
        data = {
            'text_chunks': [chunk.page_content for chunk in self.text_chunks],
            'vectorizer': self.vectorizer,
            'tfidf_matrix': self.tfidf_matrix
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"💾 Đã lưu vectorstore tại: {save_path}")
    
    def load_vectorstore(self, load_path: str):
        """
        Tải vectorstore đã lưu
        
        Args:
            load_path (str): Đường dẫn vectorstore đã lưu
        """
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        self.vectorizer = data['vectorizer']
        self.tfidf_matrix = data['tfidf_matrix']
        
        print(f"📂 Đã tải vectorstore từ: {load_path}")


def print_banner():
    """
    In banner cho ứng dụng
    """
    banner = """
    ╔═══════════════════════════════════════════════════╗
    ║              🤖 PDF CHATBOT                      ║
    ║        Sử dụng LangChain & Gemini (MIỄN PHÍ)     ║
    ║                                                   ║
    ║  • Đọc file PDF và trả lời câu hỏi               ║
    ║  • Gõ 'quit' hoặc 'exit' để thoát               ║
    ║  • Gõ 'clear' để xóa màn hình                   ║
    ║  • Gõ 'info' để xem thông tin hệ thống          ║
    ╚═══════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    """
    Hàm main để chạy ứng dụng chatbot
    """
    print_banner()
    
    # Đường dẫn file PDF mặc định
    default_pdf_path = "sample.pdf"
    
    # Kiểm tra xem file PDF có tồn tại không
    pdf_path = input(f"Nhập đường dẫn file PDF (Enter để sử dụng '{default_pdf_path}'): ").strip()
    if not pdf_path:
        pdf_path = default_pdf_path
    
    try:
        # Khởi tạo chatbot
        chatbot = GeminiPDFChatbot(pdf_path)
        
        print("\n" + "="*60)
        print("💬 Bắt đầu trò chuyện! Hãy đặt câu hỏi về nội dung PDF...")
        print("="*60)
        
        # Vòng lặp chat
        while True:
            try:
                # Nhận câu hỏi từ người dùng
                question = input("\n🙋 Bạn: ").strip()
                
                # Kiểm tra lệnh thoát
                if question.lower() in ['quit', 'exit', 'thoát']:
                    print("👋 Cảm ơn bạn đã sử dụng PDF Chatbot!")
                    break
                
                # Lệnh xóa màn hình
                if question.lower() == 'clear':
                    os.system('clear' if os.name == 'posix' else 'cls')
                    print_banner()
                    continue
                
                # Lệnh xem thông tin
                if question.lower() == 'info':
                    print(f"📊 Thông tin hệ thống:")
                    print(f"   • File PDF: {pdf_path}")
                    print(f"   • Số chunks: {len(chatbot.text_chunks)}")
                    print(f"   • Mô hình: {chatbot.model_name}")
                    print(f"   • API: Google Gemini (miễn phí)")
                    continue
                
                # Bỏ qua câu hỏi trống
                if not question:
                    print("⚠️ Vui lòng nhập câu hỏi.")
                    continue
                
                # Xử lý câu hỏi
                print("🤔 Đang suy nghĩ...")
                result = chatbot.ask_question(question)
                
                # In câu trả lời
                print(f"\n🤖 Bot: {result['answer']}")
                
                # In thông tin nguồn
                if result['source_count'] > 0:
                    print(f"\n📚 Nguồn: Tìm thấy {result['source_count']} đoạn văn liên quan")
                
            except KeyboardInterrupt:
                print("\n\n👋 Cảm ơn bạn đã sử dụng PDF Chatbot!")
                break
            except Exception as e:
                print(f"\n❌ Lỗi: {str(e)}")
                continue
    
    except FileNotFoundError as e:
        print(f"❌ Lỗi: {str(e)}")
        print("💡 Hãy đảm bảo file PDF tồn tại và đường dẫn chính xác.")
    except ValueError as e:
        print(f"❌ Lỗi cấu hình: {str(e)}")
        print("💡 Hãy kiểm tra file .env và GEMINI_API_KEY.")
    except Exception as e:
        print(f"❌ Lỗi không mong muốn: {str(e)}")


if __name__ == "__main__":
    main()