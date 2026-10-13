# NimbusAI — GPU Cost Optimization Report

**Period:** monthly  
**Baseline spend:** $27,133  
**Optimized spend:** $14,639  
**Projected savings:** $12,494  (**46%**)

## Savings by lever

| Lever | Savings (USD) |
|---|---|
| Inference (cascade/cache/batch) | $1,199 |
| Purchasing (spot/reserved) | $10,040 |
| Right-size util-lies | $655 |
| Kill idle GPUs | $600 |

## Sustainability

- Energy per query: 0.24 Wh
- Carbon per query: 0.091 gCO2e
- Cheapest+cleanest region: europe-north1

_Figures are June-2026 as-of snapshots; re-baseline before acting._

---

## Phân tích chi tiết

### 1. Inference: $/1M-token qua từng lever

| Bước (cộng dồn) | $/ngày | $/1M-token |
|---|---|---|
| baseline (all large) | $48.87 | $6.488 |
| + cascade | $11.48 | $1.523 |
| + prompt cache (gated) | $10.72 | $1.423 |
| + batch API | $8.90 | $1.182 |

Cascade đóng góp nhiều nhất: giá model nhỏ rẻ hơn 15× (input) và 37× (output), và phần lớn request chuyển được sang model nhỏ. Cache và batch chỉ chiết khấu một phần hóa đơn còn lại.

### 2. GPU-Util lie

| GPU | Loại | GPU-Util | MFU | MBU | Lãng phí so với MFU 0.40 ($/tháng) |
|---|---|---|---|---|---|
| gpu-h100-4 | H100 | 98.2% | 0.194 | 0.207 | $927 |
| gpu-a10g-1 | A10G | 96.9% | 0.268 | 0.302 | $238 |

`nvidia-smi` đo GPU-Util là tỷ lệ thời gian có ít nhất một kernel đang chạy. Chỉ số này không đo số SM hoặc Tensor Core thực sự làm việc. Một GPU chờ dữ liệu từ HBM (memory stall), chờ dataloader, hoặc chạy nhiều kernel nhỏ (kernel launch overhead) vẫn báo gần 100%.

Hai GPU trên có MBU cũng thấp (~0.2–0.3), nên vấn đề không phải do bị giới hạn băng thông. Nguyên nhân nhiều khả năng là pipeline: batch nhỏ, chờ input hoặc đồng bộ giữa các GPU. Ta vẫn trả đủ giá theo giờ nhưng chỉ nhận khoảng một nửa lượng FLOPs của một GPU khỏe. Tổng lãng phí khoảng **$1,165/tháng**. Trước khi đổi sang GPU nhỏ hơn (lever right-size, $655/tháng), nên profile bằng Nsight hoặc DCGM (`SM_ACTIVE`, `TENSOR_ACTIVE`) và sửa pipeline trước. Cách này rẻ hơn và giữ được throughput.

### 3. Thứ tự hành động đề xuất (theo ROI)

| Ưu tiên | Hành động | Tiết kiệm ($/tháng) | Công sức | Lý do |
|---|---|---|---|---|
| 1 | Tắt GPU idle và đặt lịch auto-shutdown | $600 | Rất thấp | Làm được trong ngày, không rủi ro |
| 2 | Chuyển job interruptible sang spot và mua reserved cho inference 24/7 | $10,040 | Thấp | Lever lớn nhất, chỉ cần checkpoint và cam kết hợp đồng |
| 3 | Cascade + cache có điều kiện + batch cho inference | $1,199 | Trung bình | Giảm $/1M-token 81.8%, cần router và eval chất lượng |
| 4 | Profile và right-size các GPU util-lie | $655 | Trung bình | Cần profile trước để tránh mất throughput |

Purchasing đứng thứ 2 dù tiết kiệm nhiều nhất, vì tắt GPU idle gần như không tốn công. Reserved là cam kết 3 năm. Chỉ mua khi duty cycle ≥ 55% (điểm hòa vốn). Ba job inference được chọn reserved đều chạy 18–24h/ngày nên vượt ngưỡng này.

### 4. Mở rộng 3: cache_is_worth_it()

Ước lượng số lần đọc mỗi prefix theo công thức N = h / (1 − h), với h là tỷ lệ input được cache. Chi phí ghi cache được tính 1.25× giá input, chi phí đọc 0.1×. Điểm hòa vốn là 0.25 / 0.9 ≈ 0.28 lần đọc.

| Project | Tier | Số lần đọc TB | Hòa vốn | Hòa vốn (kiểu Gemini, $1/1M/giờ) | Cache? |
|---|---|---|---|---|---|
| (untagged) | large | 0.39 | 0.28 | 0.37 | Có |
| (untagged) | small | 0.45 | 0.28 | 5.56 | Có |
| chat | large | 1.01 | 0.28 | 0.37 | Có |
| chat | small | 1.03 | 0.28 | 5.56 | Có |
| doc-qa | large | 1.07 | 0.28 | 0.37 | Có |
| doc-qa | small | 1.04 | 0.28 | 5.56 | Có |
| nightly-eval | large | 0.08 | 0.28 | 0.37 | Không |
| nightly-eval | small | 0.08 | 0.28 | 5.56 | Không |
| web-search | large | 0.08 | 0.28 | 0.37 | Không |
| web-search | small | 0.08 | 0.28 | 5.56 | Không |

chat và doc-qa có khoảng 1 lần đọc mỗi prefix, cao hơn điểm hòa vốn, nên giữ cache. web-search và nightly-eval chỉ khoảng 0.08 lần đọc, nên cache ở đây làm tăng chi phí. Chi phí inference mỗi ngày: không tính phí ghi $8.48, cache mọi nơi $9.12, cache có điều kiện $8.90. Nếu tính phí lưu trữ kiểu Gemini, model nhỏ cần khoảng 5.6 lần đọc mới hòa vốn, vì phí lưu trữ cố định trong khi giá input rất rẻ. Với model nhỏ, không nên bật explicit cache.

### 5. Mở rộng 4: Ngân sách reasoning

| Loại | Request | $/ngày | Wh/ngày |
|---|---|---|---|
| Reasoning | 201 | $1.41 | 29,788 |
| Không reasoning | 2199 | $7.49 | 1,888 |

Reasoning chiếm 8.4% traffic nhưng chiếm 15.8% chi phí và **94.0% năng lượng**. Lý do: output dài gấp ~6× (chuỗi suy luận), và mỗi token sinh ra ở giai đoạn decode là memory-bound, nên năng lượng mỗi truy vấn cao hơn ~80×.

**Quy tắc routing đề xuất:** team eval chỉ bật reasoning khi `input_tokens ≥ 3,382` (độ dài ngữ cảnh dùng làm proxy cho độ phức tạp). Quy tắc này giảm tỷ lệ reasoning của eval từ 48% xuống 10%. Kết quả: tiết kiệm $25/tháng và 681 kWh/tháng (160 request/ngày chuyển sang chế độ thường). Lợi ích chính là năng lượng, vì phần lớn traffic reasoning đã chạy trên model nhỏ.

### 6. Mở rộng 5: Lập lịch theo carbon (job interruptible)

5 job interruptible tiêu thụ 4,227 kWh/tháng (công suất board GPU).

| Vùng | $/kWh | gCO2/kWh | Tiền điện ($/tháng) | kgCO2e/tháng | Điểm cân bằng | RTT (ms) |
|---|---|---|---|---|---|---|
| us-east-wa | 0.055 | 90 | $232 | 380 | 0.22 | 15 |
| europe-north1 | 0.09 | 30 | $380 | 127 | 0.27 | 110 |
| us-west-2 | 0.07 | 120 | $296 | 507 | 0.29 | 70 |
| us-east-1 | 0.12 | 380 | $507 | 1,606 | 0.62 | 10 |
| europe-central2 | 0.18 | 660 | $761 | 2,790 | 1.00 | 120 |

- Rẻ nhất: **us-east-wa**. Sạch nhất: **europe-north1**. Cân bằng (50/50 giá và carbon): **us-east-wa**.
- Chuyển từ us-east-1 sang europe-north1: giảm 1,480 kgCO2e/tháng (92.1%) và tiết kiệm $127 tiền điện.
- Chuyển từ us-east-1 sang us-east-wa: giảm 1,226 kgCO2e/tháng và tiết kiệm $275 tiền điện.
- Ở phần Sustainability phía trên, vùng tốt nhất được chọn chỉ theo carbon (europe-north1). Vùng tối ưu thực sự phụ thuộc vào ưu tiên của công ty. Nếu có cam kết ESG, chọn europe-north1. Nếu ưu tiên chi phí, chọn us-east-wa: chỉ cao hơn khoảng 3× carbon so với vùng sạch nhất nhưng tiền điện rẻ hơn ~40%.
- Đánh đổi về latency: europe-north1 cách người dùng Mỹ khoảng 110 ms. Mức này không ảnh hưởng đến training hay batch eval, nhưng không phù hợp cho inference real-time. Vì vậy chỉ di chuyển job interruptible. Cần tính thêm phí egress khi chuyển dữ liệu hoặc checkpoint giữa các vùng.
- Tiền điện (~vài trăm $/tháng) nhỏ hơn nhiều so với tiền thuê GPU (hàng chục nghìn $), nên vùng triển khai chủ yếu là lever carbon. Lever về chi phí lớn nhất vẫn là giá spot và reserved tại vùng đó.
