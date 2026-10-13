# Lab 25: Write-up về tối ưu chi phí GPU cho NimbusAI

Mọi số liệu dưới đây lấy từ `python missions/run_all.py`, dữ liệu sinh với seed = 25, giá tháng 6/2026. Chi tiết xem `outputs/report.md`.

## 1. Baseline và optimized

| Chỉ số | Baseline | Optimized | Thay đổi |
|---|---|---|---|
| Tổng chi phí/tháng | $27,133 | $14,639 | **−$12,494 (−46%)** |
| Inference, $/1M-token | $6.488 | $1.182 | −81.8% |
| Đội GPU (8 job), $/tháng | $25,667 | $15,627 | −39.1% |

Baseline gồm hai phần: mọi request inference chạy trên model lớn, không cache, không batch; và mọi job GPU thuê on-demand.

## 2. Phân tích từng lever

| Lever | Tiết kiệm/tháng | Tỷ trọng |
|---|---|---|
| Purchasing (spot + reserved) | $10,040 | 80% |
| Inference (cascade + cache + batch) | $1,199 | 10% |
| Right-size GPU util-lie | $655 | 5% |
| Tắt GPU idle | $600 | 5% |

- **Tính theo $, purchasing đóng góp nhiều nhất** vì đội GPU chiếm ~95% hóa đơn. Có 5/8 job là interruptible, nên chạy spot kèm checkpoint giảm ~37% chi phí của các job này. Ba job inference chạy 18–24h/ngày, vượt điểm hòa vốn 55%, nên reserved giảm ~40–44%.
- **Tính theo $/1M-token, cascade đóng góp nhiều nhất.** Cascade một mình kéo giá từ $6.49 xuống $1.52 (−77%), vì model nhỏ rẻ hơn 15× (input) và 37× (output), và ~79% request đi qua model nhỏ. Cache có điều kiện giảm tiếp xuống $1.42, batch xuống $1.18.
- Bài học: một lever có % lớn trên unit cost vẫn có thể nhỏ về $ tuyệt đối nếu phần chi phí nó tác động nhỏ. Cần nhìn cả hai.

## 3. GPU-Util lie

| GPU | GPU-Util | MFU | Lãng phí so với MFU 0.40 |
|---|---|---|---|
| gpu-h100-4 | 98.2% | 0.194 | $927/tháng |
| gpu-a10g-1 | 96.9% | 0.268 | $238/tháng |

`nvidia-smi` chỉ báo thời gian có kernel đang chạy, không báo số SM hoặc Tensor Core thực sự làm việc. `gpu-h100-4` báo "bận" 98% nhưng chỉ đạt khoảng một nửa MFU của các H100 khỏe trong cùng cụm (~0.41). MBU của nó cũng thấp (0.21), nên GPU không bị nghẽn băng thông. Nguyên nhân nhiều khả năng là pipeline (batch nhỏ, chờ dataloader, đồng bộ). Tổng lãng phí khoảng **$1,165/tháng**: ta trả đủ giá H100 theo giờ nhưng chỉ nhận một nửa lượng FLOPs.

## 4. Các phần mở rộng đã làm

**Ext 3: `cache_is_worth_it()`** (`finops/pricing.py`, `missions/m2_inference_levers.py`)
- Phí ghi cache là 1.25× giá input, phí đọc 0.1×, nên điểm hòa vốn là **0.28 lần đọc** mỗi prefix.
- Số lần đọc thực tế ước lượng bằng N = h / (1 − h): chat và doc-qa khoảng 1.0 (nên cache); web-search và nightly-eval khoảng 0.08 (không nên cache).
- Chi phí inference mỗi ngày: không tính phí ghi $8.48 (mô hình gốc đánh giá thấp chi phí thật); cache mọi nơi $9.12; **cache có điều kiện $8.90**.
- Insight: với explicit cache kiểu Gemini (~$1/1M-token/giờ), model nhỏ cần **5.6 lần đọc** mới hòa vốn, còn model lớn chỉ cần 0.37. Phí lưu trữ cố định nên ăn hết khoản tiết kiệm trên model rẻ.

**Ext 4: Ngân sách reasoning** (`missions/m2_inference_levers.py`)
- Reasoning chiếm 8.4% traffic, 15.8% chi phí và **94% năng lượng**.
- Quy tắc đề xuất: team eval chỉ bật reasoning khi `input_tokens ≥ 3,382`. Quy tắc này giảm tỷ lệ reasoning của eval từ 48% xuống 10%.
- Kết quả: tiết kiệm $25/tháng và **681 kWh/tháng**. Lợi ích chủ yếu là năng lượng, vì 81% traffic reasoning đã chạy trên model nhỏ.

**Ext 5: Lập lịch theo carbon** (`missions/x5_carbon_scheduling.py`)
- 5 job interruptible tiêu thụ 4,227 kWh/tháng.
- Chuyển từ us-east-1 sang **europe-north1** (sạch nhất): −1,480 kgCO2e/tháng (−92%) và −$127 tiền điện.
- Chuyển sang **us-east-wa** (rẻ nhất, cân bằng nhất): −1,226 kgCO2e/tháng và −$275 tiền điện.
- Insight: vùng "tối ưu" phụ thuộc vào ưu tiên của công ty (ESG hay chi phí). Chỉ nên di chuyển job interruptible vì europe-north1 thêm khoảng 110 ms latency cho người dùng Mỹ.

## 5. Ba hành động đầu tiên nếu tôi là FinOps lead

1. **Tuần 1, quick wins:** tắt GPU idle và đặt auto-shutdown ($600/tháng). Bắt buộc gắn tag ở bước deploy để nâng tag coverage từ 92% lên 100%. Bật showback theo team.
2. **Tháng 1, purchasing ($10,040/tháng):** chuyển 5 job interruptible sang spot kèm checkpoint, ưu tiên vùng us-east-wa. Với 3 job inference, mua reserved 1 năm trước để kiểm chứng duty cycle, sau đó mới cam kết 3 năm.
3. **Quý 1, inference router:** triển khai cascade, cache có điều kiện, batch API và giới hạn reasoning. Song song, profile `gpu-h100-4` và `gpu-a10g-1` bằng DCGM để sửa pipeline trước khi right-size. Theo dõi KPI bằng `$/1M-token` và MFU, không dùng GPU-Util.
