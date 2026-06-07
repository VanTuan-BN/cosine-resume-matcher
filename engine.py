import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

MODEL_NAME = 'all-MiniLM-L6-v2'
print("[AI] Đang nạp mô hình mạng Nơ-ron vào bộ nhớ..." ) 
model = SentenceTransformer(MODEL_NAME)
print( "[AI] Nạp mô hình thành công! Sẵn sàng xử lý." )

def extract_text_from_pdf(pdf_path):
    """Trích xuất chữ từ PDF và làm sạch khoảng trắng"""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        return f"Lỗi đọc file PDF: {str(e)}"

def generate_embeddings(text):
    """Sử dụng mô hình Global đã load sẵn để chuyển văn bản thành Vector"""
    return model.encode(text) 

def cosine_similarity(vec1, vec2):
    """Tính toán độ tương đồng Cosine thuần bằng NumPy"""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return float(dot_product / (norm_vec1 * norm_vec2))

# Khu vực chạy thử nghiệm độc lập
if __name__ == "__main__":
    print("\n--- 🏁 BẮT ĐẦU CHẠY THỬ ENGINE ---")
    
    # 1. Giả lập một đoạn JD ngắn
    jd_test = "Tuyển dụng kỹ sư AI chuyên về Computer Vision và mô hình YOLO"
    
    # 2. Giả lập 2 ứng viên (Một ông đúng ngành, một ông lạc đề)
    cv_tot = "Tôi có kinh nghiệm 2 năm làm Deep Learning, tối ưu thị giác máy tính và train mạng YOLOv8"
    cv_lac_de = "Tôi là kế toán trưởng có kinh nghiệm kiểm toán và quản lý dòng tiền doanh nghiệp"
    
    print("\n[MÁY ĐANG DỊCH CHỮ THÀNH VECTOR...]")
    vec_jd = generate_embeddings(jd_test)
    vec_tot = generate_embeddings(cv_tot)
    vec_lac = generate_embeddings(cv_lac_de)
    
    print("\n[MÁY ĐANG ĐO GÓC KẸP COSINE...]")
    score_tot = cosine_similarity(vec_jd, vec_tot)
    score_lac = cosine_similarity(vec_jd, vec_lac)
    
    print("-" * 40)
    print(f"🎯 Điểm CV đúng ngành: {round(score_tot * 100, 2)}%")
    print(f"❌ Điểm CV lạc đề:    {round(score_lac * 100, 2)}%")
    print("-" * 40)