#!/usr/bin/env python3
"""
PDF Chatbot sử dụng LangChain, FAISS và OpenAI
Tác giả: GitHub Copilot
Mô tả: Chatbot có thể đọc file PDF và trả lời câu hỏi dựa trên nội dung file
"""

import os
import sys
from typing import List
from dotenv import load_dotenv

# Import các thư viện LangChain với cấu trúc mới
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

class PDFChatbot:
    """
    Lớp PDFChatbot để xử lý file PDF và trả lời câu hỏi
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
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY không tìm thấy trong file .env")
        
        # Thiết lập API key cho OpenAI
        os.environ['OPENAI_API_KEY'] = self.openai_api_key
        
        self.pdf_path = pdf_path
        self.vectorstore = None
        self.qa_chain = None
        
        # Khởi tạo các thành phần
        self._setup_components()
    
    def _setup_components(self):
        """
        Thiết lập các thành phần của chatbot
        """
        print("🔄 Đang khởi tạo chatbot...")
        
        # 1. Đọc file PDF
        print("📖 Đang đọc file PDF...")
        documents = self._load_pdf()
        
        # 2. Chia nhỏ văn bản
        print("✂️ Đang chia nhỏ văn bản...")
        text_chunks = self._split_text(documents)
        
        # 3. Tạo vector embeddings
        print("🔢 Đang tạo vector embeddings...")
        self.vectorstore = self._create_vectorstore(text_chunks)
        
        # 4. Thiết lập QA chain
        print("🔗 Đang thiết lập QA chain...")
        self._setup_qa_chain()
        
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
    
    def _create_vectorstore(self, text_chunks: List):
        """
        Tạo vector store từ text chunks
        
        Args:
            text_chunks (List): Danh sách text chunks
            
        Returns:
            FAISS vectorstore
        """
        # Khởi tạo OpenAI embeddings
        embeddings = OpenAIEmbeddings()
        
        # Tạo FAISS vectorstore
        vectorstore = FAISS.from_documents(
            documents=text_chunks,
            embedding=embeddings
        )
        
        print("🔢 Đã tạo vector store với FAISS")
        return vectorstore
    
    def _setup_qa_chain(self):
        """
        Thiết lập RetrievalQA chain
        """
        # Khởi tạo OpenAI LLM
        llm = OpenAI(
            temperature=0.1,  # Độ sáng tạo thấp để có câu trả lời chính xác
            model_name="gpt-3.5-turbo-instruct"
        )
        
        # Tạo prompt template tiếng Việt
        prompt_template = """
        Sử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói "Tôi không tìm thấy thông tin này trong tài liệu".
        
        Ngữ cảnh: {context}
        
        Câu hỏi: {question}
        
        Câu trả lời bằng tiếng Việt:
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Tạo QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 3}  # Lấy 3 chunks có liên quan nhất
            ),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
    
    def ask_question(self, question: str) -> dict:
        """
        Đặt câu hỏi cho chatbot
        
        Args:
            question (str): Câu hỏi của người dùng
            
        Returns:
            dict: Kết quả bao gồm câu trả lời và nguồn tài liệu
        """
        if not self.qa_chain:
            raise ValueError("QA chain chưa được khởi tạo")
        
        try:
            result = self.qa_chain({"query": question})
            return {
                "answer": result["result"],
                "source_documents": result["source_documents"]
            }
        except Exception as e:
            return {
                "answer": f"Xin lỗi, có lỗi xảy ra khi xử lý câu hỏi: {str(e)}",
                "source_documents": []
            }
    
    def save_vectorstore(self, save_path: str):
        """
        Lưu vectorstore để sử dụng lại
        
        Args:
            save_path (str): Đường dẫn lưu vectorstore
        """
        if self.vectorstore:
            self.vectorstore.save_local(save_path)
            print(f"💾 Đã lưu vectorstore tại: {save_path}")
    
    def load_vectorstore(self, load_path: str):
        """
        Tải vectorstore đã lưu
        
        Args:
            load_path (str): Đường dẫn vectorstore đã lưu
        """
        embeddings = OpenAIEmbeddings()
        self.vectorstore = FAISS.load_local(load_path, embeddings)
        self._setup_qa_chain()
        print(f"📂 Đã tải vectorstore từ: {load_path}")


def print_banner():
    """
    In banner cho ứng dụng
    """
    banner = """
    ╔═══════════════════════════════════════════════════╗
    ║              🤖 PDF CHATBOT                      ║
    ║        Sử dụng LangChain & FAISS & OpenAI        ║
    ║                                                   ║
    ║  • Đọc file PDF và trả lời câu hỏi               ║
    ║  • Gõ 'quit' hoặc 'exit' để thoát               ║
    ║  • Gõ 'clear' để xóa màn hình                   ║
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
        chatbot = PDFChatbot(pdf_path)
        
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
                
                # Bỏ qua câu hỏi trống
                if not question:
                    print("⚠️ Vui lòng nhập câu hỏi.")
                    continue
                
                # Xử lý câu hỏi
                print("🤔 Đang suy nghĩ...")
                result = chatbot.ask_question(question)
                
                # In câu trả lời
                print(f"\n🤖 Bot: {result['answer']}")
                
                # In thông tin nguồn (tùy chọn)
                if result['source_documents']:
                    print(f"\n📚 Nguồn: Tìm thấy {len(result['source_documents'])} đoạn văn liên quan")
                
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
        print("💡 Hãy kiểm tra file .env và OPENAI_API_KEY.")
    except Exception as e:
        print(f"❌ Lỗi không mong muốn: {str(e)}")


if __name__ == "__main__":
    main()