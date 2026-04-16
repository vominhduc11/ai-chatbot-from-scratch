# Level 2 Generative Chatbot

Nhánh này nâng project từ **intent classification + chọn câu mẫu** lên **seq2seq generative chatbot**.

## Ý tưởng

- Bản cũ: model chỉ đoán intent, sau đó code lấy câu trả lời có sẵn.
- Bản mới: model học từ cặp `input -> output` và **tự sinh câu trả lời theo từng token**.

## File mới

- `src/generative_nlp.py`: xử lý text, vocab và encode/decode cho generative model
- `src/generative_model.py`: mô hình seq2seq GRU
- `train_level2.py`: pipeline train cấp 2
- `src/generative_inference.py`: engine inference cho cấp 2
- `streamlit_level2_app.py`: giao diện demo chatbot cấp 2

## Cách train

```bash
python train_level2.py
```

Sau khi train xong, artifacts mới sẽ được tạo:

- `artifacts/level2_model.pt`
- `artifacts/level2_metadata.json`

## Cách chạy demo

```bash
streamlit run streamlit_level2_app.py
```

## Dữ liệu train

Bản cấp 2 hiện tại **không thêm file dataset lớn mới**.
`train_level2.py` tự sinh cặp `input -> output` từ `data/intents.json` hiện có bằng cách ghép:

- mỗi `pattern`
- với mỗi `response`

Điều này đủ để chuyển project sang kiến trúc generative, nhưng chất lượng vẫn bị giới hạn bởi dữ liệu gốc.

## Giới hạn hiện tại

- Dữ liệu vẫn còn nhỏ
- Model là seq2seq GRU nhỏ, chưa phải LLM
- Chất lượng câu sinh ra chỉ ở mức demo học thuật
- Có thêm `retrieved_fallback` để tránh trả lời quá tệ khi model sinh lỗi hoặc input lệch domain

## Muốn nâng tiếp

Bước tiếp theo hợp lý:

1. Tạo dataset `input -> output` lớn và sạch hơn
2. Thêm test set riêng cho generative model
3. Thử attention hoặc transformer mini
4. Tách inference thành FastAPI service
5. Khi cần chất lượng wording cao hơn, chuyển sang hybrid với pretrained model hoặc LLM
