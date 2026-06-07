# 📄 Cosine Similarity Resume Matcher

Hệ thống thông minh hỗ trợ bộ phận tuyển dụng tự động đọc hiểu file CV (PDF), trích xuất kỹ năng và chấm điểm độ tương đồng định lượng về mặt ngữ nghĩa với Mô tả công việc (JD). 

Ứng dụng tách biệt rõ ràng kiến trúc hệ thống: **Backend Engine** xử lý toán học thuần túy và **Frontend UI** tương tác real-time trực quan.

---

## 🚀 Tính năng cốt lõi
- **Trích xuất văn bản (Text Extraction):** Tự động xử lý, đọc hiểu dữ liệu từ cấu trúc file CV định dạng PDF.
- **Mã hóa ngữ nghĩa (Vector Embedding):** Sử dụng mô hình tiền huấn luyện giải quyết bài toán biến đổi các đoạn văn bản dài thành các vector đa chiều trong không gian, giữ nguyên ngữ cảnh.
- **Chấm điểm định lượng (Cosine Similarity):** Áp dụng thuật toán đo góc giữa hai vector CV và JD để đưa ra điểm số trùng khớp từ `0.0` đến `100%`.
- **Giao diện trực quan (Streamlit UI):** Giao diện Web mượt mà, cho phép nhà tuyển dụng kéo thả CV và dán JD để nhận kết quả ngay lập tức.

---

## 🧮 Bản chất toán học của dự án

Hệ thống không chỉ so khớp từ khóa (Keyword Matching) thông thường, mà chuyển đổi toàn bộ văn bản thành các vector và áp dụng công thức **Cosine Similarity** để tính toán khoảng cách góc.

Thuật toán được tối ưu hóa bằng **NumPy** để tính toán tích vô hướng chia cho tích độ dài hình học Euclid, đảm bảo tốc độ xử lý nhanh và kiểm soát chặt chẽ điều kiện biên (tránh lỗi chia cho số 0).

---

## 🛠️ Công nghệ sử dụng
- **Ngôn ngữ:** Python (Tư duy lập trình hướng đối tượng - OOP)
- **Xử lý Toán & Nhúng ngữ nghĩa:** NumPy, Scikit-learn / Sentence-Transformers
- **Giao diện Web:** Streamlit
- **Xử lý file:** PyPDF2

---

## 📂 Cấu trúc thư mục dự án
```text
cosine-resume-matcher/
├── CV_data/            # Thư mục chứa file CV và JD mẫu để test hệ thống
├── video_test/         # Thư mục chứa video test hệ thống
├── .gitignore          # Chặn các tệp tin cấu hình rác và môi trường ảo
├── README.md           # Hướng dẫn và mô tả chi tiết dự án
├── requirements.txt    # Danh sách các thư viện bắt buộc của hệ thống
├── app.py              # Luồng xử lý giao diện người dùng (Streamlit Frontend)
└── engine.py           # Module thuật toán core toán học (NumPy Backend)