# Kế hoạch thực hiện — Lightweight Wake Word Detection “Alo GPT à em”

## 1. Sản phẩm cần nộp (theo yêu cầu môn)

| Sản phẩm | Nằm ở đâu | Trạng thái |
|---|---|---|
| Source code + Docker | repo này (`docker/`, `docker-compose.yml`) | ✅ khung hoàn chỉnh, đã test end-to-end trên dữ liệu giả |
| Demo Gradio | `apps/demo_app.py` (phân tích file, mic trực tiếp, thông tin model) | ✅ chạy được, cần model train trên dữ liệu thật |
| MLOps (không bắt buộc) | MLflow, DVC, GitHub Actions CI | ✅ đã cấu hình |
| Technical report | viết khi có kết quả (dàn ý ở mục 7) | ⏳ |
| Slides | viết khi có kết quả (dàn ý ở mục 7) | ⏳ |

## 2. Câu hỏi nghiên cứu (đây là phần “different approaches”)

1. **Kiến trúc:** DS-CNN, TC-ResNet, CRNN và embedding pretrained + head (openWakeWord) khác nhau thế nào về FRR, FA/giờ, số tham số, MACs và latency?
2. **Thiết kế dữ liệu:** mỗi thành phần đóng góp bao nhiêu vào việc giảm false activation? Các bước lần lượt là: người thật → + TTS → + augmentation → + hard negatives → + mined hard negatives.
3. **Synthetic-to-real:** model train phần lớn bằng TTS có tổng quát hóa tốt sang người thật chưa thấy không?
4. **Vì sao không dùng STT:** so sánh với PhoWhisper-tiny + so khớp chuỗi về compute (RTF), FA/giờ và tỷ lệ kích hoạt nhầm trên hard negatives.
5. **Nén mô hình:** INT8 giảm kích thước và latency bao nhiêu, và mất bao nhiêu độ chính xác?
6. **Policy:** khoảng dừng bao lâu thì vẫn coi là wake word? Model xử lý câu tường thuật (challenge set) ra sao?

## 3. Lộ trình (8 tuần, điều chỉnh theo deadline thật)

| Tuần | Việc chính | Kết quả |
|---|---|---|
| 1 | Chốt policy nhãn (các cách đọc “GPT”: “Gi Pi Ti” / “gờ pê tê”?). Cài môi trường, chạy `pytest`. Mỗi người tự thu bằng `collect_app` (chế độ Đầy đủ, môi trường clean) | ≥ 4 người × ~150 bản ghi |
| 2 | Thu thêm trong môi trường có nhiễu, ở xa mic, ở phòng có vang. Chạy TTS (edge + mms) và `compose_pauses`. Tải ESC-50, RIRs và tiếng Việt (VIVOS/FLEURS). Thu ≥ 3–5 giờ audio negative liên tục (TV, podcast, nói chuyện) | `data/raw` đầy đủ, chạy `build_manifest` không còn cảnh báo leakage |
| 3 | Gửi link `collect_app --share` cho ≥ 8–10 người ngoài nhóm (tiền tố `ext_`, chỉ dùng để test). Train các baseline A, B, CRNN | test set người lạ, kết quả baseline |
| 4 | Hướng C (oww_head) và hướng D (STT baseline). Tinh chỉnh window, threshold và smoothing | bảng so sánh mô hình |
| 5 | Ablation dữ liệu a0 → a3, mining hard negative → a4 (V2) | bảng ablation, đường DET V1 vs V2 |
| 6 | Export ONNX/INT8 và đo latency. Phân tích lỗi: nghe lại false accept/reject, xem kết quả theo speaker, vùng miền, môi trường | bảng nén, phần error analysis |
| 7 | Hoàn thiện demo (model tốt nhất, có thể nối GPT thật). Build và kiểm tra Docker. Viết report | bản nháp report |
| 8 | Slides, quay video demo dự phòng, rà soát code và README | nộp bài |

## 4. Phân công gợi ý (4 người, đổi tùy nhóm)

- **Data lead:** policy nhãn, `collect_app`, tổ chức thu người ngoài, TTS, kiểm tra chất lượng (nghe ngẫu nhiên 5%).
- **Model lead A/B:** DS-CNN, TC-ResNet, CRNN, tuning, INT8.
- **Model lead C/D:** oww_head, STT baseline, mining hard negatives.
- **MLOps + demo lead:** MLflow, Docker, CI, Gradio demo, `compare_runs`, hình vẽ cho report.

Tất cả cùng tham gia thu âm, viết report và làm slides.

## 5. Giao thức thu dữ liệu

- **Định dạng:** WAV mono 16 kHz (app tự chuyển). Bỏ mọi bản ghi bị cắt mất chữ “em”.
- **Mỗi thành viên:** chạy chế độ Đầy đủ ≥ 2 lần ở 2 môi trường khác nhau (clean + một loại nhiễu hoặc ở xa). Như vậy mỗi người có khoảng 50 positive, 15 positive kèm lệnh và khoảng 90 hard negative.
- **Người ngoài nhóm (chỉ test):** chế độ Nhanh, càng đa dạng giới tính, vùng miền và thiết bị càng tốt.
- **Audio negative liên tục:** thu bằng điện thoại để trong phòng khi bật TV, podcast hoặc có người nói chuyện. Chia file vào `data/raw/long/{train,val,test}` (khác nguồn hoặc khác ngày). **Không được có wake phrase trong các file này.**
- **Mục tiêu tối thiểu:** khoảng 600 positive người thật, khoảng 2.000 positive TTS, khoảng 3.000 hard negative, ≥ 5 giờ negative liên tục (≥ 2 giờ cho test).
- **Quyền riêng tư:** chỉ lưu tên viết tắt hoặc ID. App có ô đồng ý (consent), giữ dữ liệu trong nhóm.

## 6. Checklist thí nghiệm

- [ ] `build_manifest`: không có speaker xuất hiện ở nhiều split; độ dài positive (p95) < 2 s sau khi trim.
- [ ] A / B / C / CRNN: train → evaluate → export → evaluate INT8 (`run_experiments.py --only models`).
- [ ] D: `stt_baseline.py` trên cùng test split và cùng ≥ 0,5 giờ audio liên tục.
- [ ] Ablation a0–a4 (`run_experiments.py --only ablation`).
- [ ] Error analysis: `clip_scores.csv` sắp theo score, nghe 20 false accept và 20 false reject.
- [ ] Chọn operating point (ví dụ FA/giờ ≤ 0,5) và dùng ngưỡng đó cho demo.
- [ ] `compare_runs.py --out runs/comparison.md` → đưa bảng vào report.

## 7. Dàn ý Technical Report và Slides

**Report:**
1. Giới thiệu & bài toán
2. Vì sao tách wake-word khỏi STT
3. Liên quan: KWS, DS-CNN, TC-ResNet, openWakeWord
4. Dữ liệu: policy nhãn, thu thập, TTS, augmentation, split theo speaker, hard negatives
5. Phương pháp: 4 hướng tiếp cận, streaming detector, metric
6. Thí nghiệm: setup, bảng so sánh mô hình, ablation dữ liệu, V1→V2, INT8, STT baseline
7. Error analysis & challenge set
8. Demo & triển khai (Docker, MLOps)
9. Hạn chế & hướng phát triển
10. Kết luận

**Slides (~12–15 trang):** bài toán và luồng hệ thống → vì sao không dùng STT → policy positive/negative/hard negative → pipeline dữ liệu → các mô hình → metric (FRR, FA/giờ, DET) → bảng kết quả → ablation dữ liệu → V1 vs V2 → INT8 và latency → demo (video hoặc live) → kết luận.

## 8. Rủi ro & cách giảm

| Rủi ro | Cách giảm |
|---|---|
| Ít người nói thật | TTS nhiều voice + augmentation; test vẫn bằng người thật chưa thấy; báo cáo rõ khoảng tin cậy |
| “GPT” có nhiều cách đọc | chốt policy từ tuần 1; mỗi cách đọc được chấp nhận là positive, cách đọc ngoài policy đưa vào hard negative |
| FA/giờ không đo được vì thiếu audio dài | `make_long_negatives.py` làm phương án tạm; ưu tiên thu thật |
| Hard negatives quá khó (ví dụ “Alo em à”) | tăng tỷ trọng `group_weights.hard`, mining V2, nếu cần tăng window hoặc thêm làm mượt |
| Leakage giữa các split | `build_manifest` có kiểm tra; tách người ngoài bằng tiền tố `ext_` |
| Demo live trục trặc khi trình bày | chuẩn bị file audio mẫu cho tab “Phân tích audio” và quay video dự phòng |
