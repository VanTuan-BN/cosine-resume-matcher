import streamlit as st
import pandas as pd
import engine  # Nhập file engine.py tối ưu của bạn

# --- GIAO DIỆN ---
st.set_page_config(page_title="AI Resume Matcher", page_icon="🎯")

st.title("🎯 AI Resume Matcher")
st.info("Hệ thống sử dụng cấu trúc tách biệt: UI (app.py) + Core Logic (engine.py)")

# Layout 2 cột
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Yêu cầu Tuyển dụng")
    jd_input = st.text_area("Dán nội dung JD tại đây:", height=250, placeholder="Ví dụ: Cần tuyển kỹ sư AI chuyên về Computer Vision...")

with col2:
    st.subheader("2. Hồ sơ Ứng viên")
    uploaded_files = st.file_uploader("Tải lên các bản CV (PDF):", type="pdf", accept_multiple_files=True)

# Nút xử lý
if st.button("🚀 Thực hiện Chấm điểm", type="primary"):
    if jd_input and uploaded_files:
        
        # Bước 1: Embedding cho JD sử dụng hàm mã hóa của engine.py
        with st.spinner("Đang mã hóa dữ liệu JD..."):
            jd_vector = engine.generate_embeddings(jd_input)
        
        results = []
        
        # Bước 2: Duyệt danh sách CV
        progress_text = "Đang xử lý danh sách CV..."
        my_bar = st.progress(0, text=progress_text)
        
        for index, file in enumerate(uploaded_files):
            # Gọi hàm trích xuất từ engine.py
            cv_text = engine.extract_text_from_pdf(file)
            
            # Kiểm tra: Text phải tồn tại và KHÔNG PHẢI là chuỗi báo lỗi
            if cv_text and not cv_text.startswith("Lỗi"):
                # Sử dụng trực tiếp hàm tạo embedding từ engine.py
                cv_vector = engine.generate_embeddings(cv_text)
                
                # Gọi hàm tính toán Cosine từ engine.py
                score = engine.cosine_similarity(jd_vector, cv_vector)
                
                results.append({
                    "Tên ứng viên": file.name,
                    "Độ tương đồng (%)": round(score * 100, 2)
                })
            else:
                # Nếu file lỗi, thông báo nhẹ cho người dùng biết
                st.error(f"Không thể xử lý file: {file.name}")
            
            # Cập nhật progress bar
            my_bar.progress((index + 1) / len(uploaded_files))
            
        # Hiển thị kết quả
        if results:
            df = pd.DataFrame(results).sort_values(by="Độ tương đồng (%)", ascending=False)
            
            st.divider()
            st.subheader("📊 Bảng xếp hạng ứng viên phù hợp")
            
            # Hiển thị bảng kèm thanh Progress trực quan
            st.dataframe(
                df, 
                use_container_width=True,
                column_config={
                    "Độ tương đồng (%)": st.column_config.ProgressColumn(
                        "Mức độ phù hợp",
                        format="%.2f%%",
                        min_value=0,
                        max_value=100,
                    ),
                },
                hide_index=True
            )
            
            # Biểu đồ cột so sánh trực quan cho HR
            st.bar_chart(df.set_index("Tên ứng viên"))
    else:
        st.warning("Vui lòng điền đầy đủ thông tin JD và chọn ít nhất một file CV!")