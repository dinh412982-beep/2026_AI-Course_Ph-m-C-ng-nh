-- init.sql
-- Chạy tự động khi container Postgres khởi tạo lần đầu (volume rỗng).

CREATE EXTENSION IF NOT EXISTS vector;

-- Bảng lưu từng đoạn luật + vector embedding SBERT tương ứng.
-- Lưu ý: VECTOR(768) khớp với chiều embedding suy ra từ file
-- du_lieu_luat_sbert.pkl (SBERT tiếng Việt, ví dụ keepitreal/vietnamese-sbert).
-- Nếu bạn dùng model SBERT khác có số chiều khác, hãy sửa số 768 bên dưới
-- cho khớp rồi mới chạy import_data.py.
CREATE TABLE IF NOT EXISTS legal_documents (
    id          SERIAL PRIMARY KEY,
    category    TEXT,              -- ví dụ: "NHÓM 1: CÁC LỖI CỦA XE MÁY..."
    content     TEXT NOT NULL,     -- nội dung điều luật, ví dụ "Điều 6 Khoản 2: ..."
    embedding   VECTOR(768) NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Index ANN cho tìm kiếm cosine similarity nhanh hơn.
-- Với dataset nhỏ (vài chục - vài trăm dòng) thì quét tuần tự đã đủ nhanh,
-- index này chủ yếu để pipeline sẵn sàng scale khi dữ liệu luật lớn hơn.
CREATE INDEX IF NOT EXISTS legal_documents_embedding_idx
    ON legal_documents
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 10);
