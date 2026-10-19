"""
add_entry.py
------------
Thêm nhanh 1 điều luật mới vào database, KHÔNG cần đụng tới file docx hay
chạy lại prepare_data.py + import_data.py (vốn sẽ xoá sạch và nạp lại
TOÀN BỘ bảng). Dùng khi chỉ cần bổ sung thêm vài điều luật lẻ.

Chạy (khi các container đang bật):
    docker compose exec backend python add_entry.py \
        --category "NHÓM 9: LỖI KHÁC" \
        --content "Điều 10 Khoản 1: Người điều khiển xe đạp điện không đội mũ bảo hiểm bị phạt tiền từ 100.000 đồng đến 200.000 đồng."

Hoặc chạy không tham số để nhập tương tác:
    docker compose exec -it backend python add_entry.py

Lưu ý: nội dung "--content" nên bắt đầu bằng "Điều ..." cho đồng nhất với
cách import_data.py nhận diện điều luật (không bắt buộc, script này vẫn
insert bình thường dù không đúng định dạng, chỉ là để nhất quán với dữ
liệu có sẵn).
"""

import argparse
import os

import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from underthesea import word_tokenize

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://chatbot:chatbot@localhost:5432/luat_db"
)
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "keepitreal/vietnamese-sbert")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default=None, help="Tên nhóm luật, ví dụ: 'NHÓM 9: LỖI KHÁC'")
    parser.add_argument("--content", default=None, help="Nội dung điều luật, nên bắt đầu bằng 'Điều ...'")
    args = parser.parse_args()

    category = args.category or input("Nhập tên nhóm (category, Enter để bỏ trống): ").strip() or None
    content = args.content or input("Nhập nội dung điều luật: ").strip()

    if not content:
        print("Nội dung trống, dừng lại, không thêm gì cả.")
        return

    print(f"Đang tải model '{EMBEDDING_MODEL_NAME}' ...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # QUAN TRỌNG: phải tách từ tiếng Việt giống hệt lúc tạo dữ liệu gốc,
    # nếu không vector sẽ không cùng "không gian" với các điều luật khác.
    segmented = word_tokenize(content, format="text")
    vector = model.encode(segmented)

    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    try:
        with conn, conn.cursor() as cur:
            # Truyền thẳng numpy array (KHÔNG .tolist()) để pgvector adapter
            # sinh đúng kiểu "vector", không bị lỗi vector <=> numeric[].
            cur.execute(
                "INSERT INTO legal_documents (category, content, embedding) VALUES (%s, %s, %s)",
                (category, content, vector),
            )
        print("✅ Đã thêm thành công vào database.")
        print(f"   category: {category}")
        print(f"   content : {content}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
