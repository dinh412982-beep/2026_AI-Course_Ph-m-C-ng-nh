# Chatbot Vấn đáp Văn bản Pháp luật (Giao thông)

## Pipeline

```
data/btl_nlp.docx                 (nguồn văn bản luật gốc)
        │  backend/prepare_data.py  (chỉ chạy khi cần TẠO LẠI/CẬP NHẬT .pkl)
        │  - đọc từng paragraph trong docx
        │  - encode TOÀN BỘ dòng (cả tiêu đề nhóm lẫn "Điều...") bằng SBERT
        ▼
du_lieu_luat_sbert.pkl            { cac_doan_luat: [text...], vector_luat: Tensor[N,768] }
        │
        │  backend/import_data.py
        │  - đọc pkl, ghép mỗi dòng "Điều ..." với embedding CÙNG INDEX
        │  - dòng tiêu đề nhóm chỉ dùng để cập nhật category, không insert
        ▼
Postgres + pgvector (db/init.sql)  bảng legal_documents(category, content, embedding VECTOR(768))
        │
        │  backend/app.py  (Flask API)
        │  - encode câu hỏi người dùng bằng CÙNG model SBERT
        │  - ORDER BY embedding <=> query_vector  (cosine distance) LIMIT k
        │  - ghép top-k đoạn luật thành câu trả lời
        ▼
frontend/index.html                giao diện chat gọi POST /api/chat
```

### Đã kiểm chứng trực tiếp trên file `du_lieu_luat_sbert.pkl` bạn gửi

Mình viết một trình đọc pickle thủ công (giả lập tối thiểu các lớp
`torch.FloatStorage` bằng numpy, không cần cài torch thật) để mở đúng file
gốc và chạy thử toàn bộ logic trong `import_data.py`, kết quả thực tế:

| Thông số | Giá trị thật |
|---|---|
| Tổng số dòng trong `cac_doan_luat` | 89 |
| Số dòng tiêu đề nhóm | 8 |
| Số dòng "Điều ..." (sẽ insert vào DB) | 81 |
| Chiều embedding (`vector_luat.shape[1]`) | 768 |
| `len(texts) == len(vectors)`? | Có — mọi dòng kể cả tiêu đề đều có vector cùng index |

8 nhóm luật tìm được: xe máy, ô tô, tốc độ, giấy tờ/bằng lái, độ tuổi & xe đạp/đi bộ, độ xe, tình huống nguy hiểm/tai nạn, làn đường/chuyển hướng.

Mình cũng test thử cosine similarity thủ công bằng numpy (mô phỏng đúng
phép toán `embedding <=> query` mà pgvector sẽ chạy): với 1 điều luật làm
câu hỏi mẫu, kết quả top-1 luôn là chính nó (similarity = 1.0) và các kết
quả gần tiếp theo đều cùng chủ đề — xác nhận cách ghép dữ liệu và công
thức truy vấn trong `app.py` là đúng logic.

**Lưu ý:** mình chỉ xác nhận được *cấu trúc* dữ liệu (số dòng, số chiều,
cách ghép category), còn *tên model SBERT chính xác* đã dùng để tạo file
này thì không thể xác nhận vì file pkl không lưu lại tên model. Mặc định
vẫn là `keepitreal/vietnamese-sbert` (768 chiều, khớp số liệu) — bạn xác
nhận lại giúp mình nếu biết chính xác model đã dùng.

### Cập nhật: đã đối chiếu với notebook gốc, sửa 1 bug quan trọng

Sau khi xem code notebook thật (Google Colab) mà bạn dùng để tạo file
`.pkl`, có 2 điều được xác nhận và 1 bug được sửa:

1. ✅ **Tên model đúng**: `keepitreal/vietnamese-sbert` — như dự đoán.
2. 🐛 **Bug đã sửa**: notebook gốc **tách từ tiếng Việt bằng `underthesea.word_tokenize(text, format="text")`** (vd: "mũ bảo hiểm" → "mũ_bảo_hiểm") **trước khi** encode bằng SBERT. Bản `app.py` lúc đầu của mình encode thẳng câu hỏi thô, không tách từ → vector câu hỏi và vector database không cùng không gian ngữ nghĩa, tìm kiếm sẽ kém chính xác. Đã sửa: `app.py` giờ luôn gọi `underthesea.word_tokenize` trước khi encode câu hỏi, và `prepare_data.py` cũng thêm bước này khi tạo lại `.pkl` từ docx.
3. ✨ **Đã đưa cơ chế "buff điểm theo từ khóa"** từ notebook gốc (hàm `tim_kiem_nang_cao`) vào `app.py`: sau khi lấy top ứng viên theo cosine similarity, nếu câu hỏi khớp ≥50% số từ khóa (hoặc câu hỏi 1 từ khớp trọn) với nội dung điều luật, điểm được cộng thêm `KEYWORD_BUFF` (mặc định 0.28, trần 0.98). Chỉ kết quả có điểm > `MIN_SIMILARITY` (mặc định 0.45) mới được trả về — đúng ngưỡng đã test trong notebook.

## Chạy toàn bộ hệ thống

```bash
cp .env.example .env      # chỉnh lại nếu cần (đặc biệt là EMBEDDING_MODEL_NAME)
docker compose up --build -d
```

### (Tuỳ chọn) Tạo lại `du_lieu_luat_sbert.pkl` từ `data/btl_nlp.docx`

Chỉ cần bước này nếu bạn muốn cập nhật nội dung luật từ file docx gốc.
Nếu bạn đã có sẵn `du_lieu_luat_sbert.pkl` và không đổi gì, bỏ qua bước
này và làm luôn bước "nạp dữ liệu" bên dưới.

```bash
docker compose exec backend python prepare_data.py \
    --docx /app/data/btl_nlp.docx \
    --out  /app/du_lieu_luat_sbert.pkl
```

Sau khi 3 container (`chatbot_db`, `chatbot_backend`, `chatbot_frontend`) đã chạy khỏe (`docker compose ps`), nạp dữ liệu vào database (chỉ cần chạy 1 lần, hoặc mỗi khi cập nhật pkl):

```bash
docker compose exec backend python import_data.py
```

Mở trình duyệt tại: **http://localhost:8080**

Backend API đứng độc lập tại `http://localhost:8000` (health check: `GET /api/health`).

## Rất quan trọng: khớp model SBERT

`du_lieu_luat_sbert.pkl` được tạo ra bằng một model SBERT cụ thể (không rõ tên chính xác từ file pkl). Mặc định repo này giả định model là `keepitreal/vietnamese-sbert` (768 chiều — khớp với số chiều suy ra từ file pkl bạn gửi: 89 vector × 768).

Nếu bạn biết chính xác model đã dùng để tạo file `.pkl` (ví dụ một checkpoint SBERT tiếng Việt khác, hoặc `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`), hãy:

1. Sửa `EMBEDDING_MODEL_NAME` trong `.env`.
2. Nếu số chiều embedding khác 768, sửa `VECTOR(768)` trong `db/init.sql` cho khớp.
3. `docker compose down -v` (xóa volume DB cũ) rồi `docker compose up --build -d` và chạy lại `import_data.py`.

Nếu model không khớp, chatbot vẫn chạy nhưng kết quả tìm kiếm sẽ **sai hoàn toàn** vì câu hỏi và dữ liệu luật không còn nằm chung không gian vector.

**Lưu ý thêm về tách từ:** dù model đúng, nếu bạn tự sửa `app.py`/`prepare_data.py` sau này, nhớ **giữ nguyên bước `underthesea.word_tokenize(text, format="text")` trước khi encode** — cả câu hỏi lẫn dữ liệu luật đều phải qua bước này, nếu không kết quả tìm kiếm cũng sẽ sai dù model đúng.

## Cấu trúc thư mục

```
chatbot/
├── docker-compose.yml
├── .env.example
├── data/
│   └── btl_nlp.docx            # (giữ nguyên file gốc của bạn ở đây nếu cần đối chiếu)
├── db/
│   └── init.sql                # tạo extension pgvector + bảng legal_documents
└── backend/
    ├── app.py                  # API /api/chat
    ├── import_data.py          # nạp pkl -> Postgres
    ├── prepare_data.py         # (tuỳ chọn) docx -> pkl, tạo lại embedding
    ├── du_lieu_luat_sbert.pkl  # backup dữ liệu đã encode
    ├── requirements.txt
    ├── Dockerfile
    └── .dockerignore
└── frontend/
    ├── index.html              # giao diện chat
    └── Dockerfile
```

## Gợi ý mở rộng

- **Sinh câu trả lời tự nhiên hơn**: thay `build_answer()` trong `app.py` bằng một lời gọi LLM (OpenAI/Gemini/…) dùng top-k đoạn luật làm ngữ cảnh (RAG) thay vì chỉ liệt kê nguyên văn.
- **Đánh giá độ chính xác**: log lại các câu hỏi có `similarity` thấp (< `MIN_SIMILARITY`) để biết những trường hợp dữ liệu luật còn thiếu.
- **Regenerate embedding từ docx**: nếu cần tạo lại `du_lieu_luat_sbert.pkl` từ `data/btl_nlp.docx`, viết thêm một script dùng `python-docx` để trích văn bản (đã có sẵn trong `requirements.txt`) rồi `SentenceTransformer(...).encode(...)`.
