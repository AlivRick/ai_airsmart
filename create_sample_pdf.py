#!/usr/bin/env python3
"""
Script tạo file PDF mẫu để test chatbot
"""

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

def create_sample_pdf():
    """
    Tạo file PDF mẫu với nội dung tiếng Việt
    """
    filename = "sample.pdf"
    doc = SimpleDocTemplate(filename, pagesize=letter)
    
    # Lấy style mặc định
    styles = getSampleStyleSheet()
    story = []
    
    # Tiêu đề
    title = Paragraph("Hướng dẫn sử dụng Trí tuệ Nhân tạo trong Giáo dục", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Nội dung
    content = [
        "1. Giới thiệu về AI trong Giáo dục",
        "",
        "Trí tuệ nhân tạo (AI) đang thay đổi cách chúng ta học tập và giảng dạy. "
        "Các ứng dụng AI có thể giúp cá nhân hóa việc học, tự động hóa việc chấm điểm, "
        "và cung cấp phản hồi ngay lập tức cho học sinh.",
        "",
        "2. Lợi ích của AI trong Giáo dục",
        "",
        "- Học tập cá nhân hóa: AI có thể điều chỉnh nội dung học tập phù hợp với tốc độ và phong cách học của từng học sinh.",
        "- Tự động hóa: Giảm tải công việc của giáo viên bằng cách tự động chấm bài và tạo báo cáo.",
        "- Phân tích dữ liệu: Giúp phân tích hiệu suất học tập và xác định những lĩnh vực cần cải thiện.",
        "- Hỗ trợ 24/7: Chatbot có thể trả lời câu hỏi của học sinh bất cứ lúc nào.",
        "",
        "3. Các ứng dụng phổ biến",
        "",
        "- Hệ thống quản lý học tập thông minh (LMS)",
        "- Chatbot hỗ trợ học tập",
        "- Công cụ tạo nội dung tự động",
        "- Hệ thống đánh giá thông minh",
        "",
        "4. Thách thức và Hạn chế",
        "",
        "Mặc dù AI mang lại nhiều lợi ích, nhưng cũng có những thách thức như:",
        "- Vấn đề quyền riêng tư và bảo mật dữ liệu",
        "- Chi phí triển khai và bảo trì",
        "- Cần đào tạo giáo viên sử dụng công nghệ mới",
        "- Nguy cơ phụ thuộc quá nhiều vào công nghệ",
        "",
        "5. Tương lai của AI trong Giáo dục",
        "",
        "Trong tương lai, AI sẽ trở nên phổ biến hơn trong giáo dục với những tính năng như:",
        "- Thực tế ảo (VR) và thực tế tăng cường (AR) trong học tập",
        "- AI có thể hiểu cảm xúc của học sinh",
        "- Hệ thống gia sư AI thông minh hơn",
        "- Tích hợp AI vào tất cả các môn học",
        "",
        "6. Kết luận",
        "",
        "AI trong giáo dục là một xu hướng không thể tránh khỏi. Việc áp dụng AI một cách "
        "thông minh và có trách nhiệm sẽ giúp nâng cao chất lượng giáo dục và tạo ra "
        "những trải nghiệm học tập tốt hơn cho học sinh."
    ]
    
    for line in content:
        if line:
            p = Paragraph(line, styles['Normal'])
            story.append(p)
        story.append(Spacer(1, 6))
    
    # Tạo PDF
    doc.build(story)
    print(f"✅ Đã tạo file PDF mẫu: {filename}")

if __name__ == "__main__":
    try:
        create_sample_pdf()
    except ImportError:
        print("❌ Cần cài đặt reportlab: pip install reportlab")
    except Exception as e:
        print(f"❌ Lỗi khi tạo PDF: {e}")