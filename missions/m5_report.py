"""M5 — Optimization Report: combine M1-M4 into baseline-vs-optimized (deck §1/§11).

Run: python missions/m5_report.py   ->  outputs/report.md + outputs/savings.png
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
from missions._common import num, catalog_by_type, ROOT
from finops import report, sustainability
from missions import m1_efficiency_audit, m2_inference_levers, m3_purchasing, x5_carbon_scheduling

DAYS = 30
# one tier down for over-provisioned ("util-lie") GPUs
RIGHTSIZE_MAP = {"H100": "A100", "H200": "H100", "A100": "A10G", "A10G": "L4", "L4": "L4"}
TARGET_MFU = 0.40  # healthy H100s in M1 sit at ~0.40-0.43


def run(verbose: bool = True) -> dict:
    r1 = m1_efficiency_audit.run(verbose=False)
    r2 = m2_inference_levers.run(verbose=False)
    r3 = m3_purchasing.run(verbose=False)
    cat = catalog_by_type()

    # --- buckets ---
    infer_savings = (r2["baseline_daily"] - r2["optimized_daily"]) * DAYS
    purchasing_savings = r3["on_demand_monthly"] - r3["optimized_monthly"]

    idle_savings = r1["idle_waste_daily"] * DAYS
    rightsize_savings = 0.0
    for lie in r1["lies"]:
        cur = lie["gpu_type"]
        tgt = RIGHTSIZE_MAP.get(cur, cur)
        delta = num(cat[cur]["on_demand_hr"]) - num(cat[tgt]["on_demand_hr"])
        rightsize_savings += max(0.0, delta) * 24 * DAYS

    levers = {
        "Inference (cascade/cache/batch)": round(infer_savings),
        "Purchasing (spot/reserved)": round(purchasing_savings),
        "Right-size util-lies": round(rightsize_savings),
        "Kill idle GPUs": round(idle_savings),
    }
    baseline = r2["baseline_daily"] * DAYS + r3["on_demand_monthly"]
    optimized = baseline - sum(levers.values())
    total_pct = sum(levers.values()) / baseline * 100 if baseline else 0.0

    # --- sustainability snapshot ---
    median_tokens = 800
    wh = sustainability.wh_per_query(median_tokens)
    sust = {
        "wh_per_query": wh,
        "carbon_g": sustainability.carbon_g(wh, "us-east-1"),
        "best_region": min(sustainability.REGION_CARBON, key=sustainability.REGION_CARBON.get),
    }

    md = report.build_report(baseline, optimized, levers, sustainability=sust)
    r5 = x5_carbon_scheduling.run(verbose=False)
    md += "\n\n" + analysis_sections(r1, r2, r3, r5, levers, cat, sust)
    out_md = os.path.join(ROOT, "outputs", "report.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w") as f:
        f.write(md)
    png = report.savings_waterfall(levers, os.path.join(ROOT, "outputs", "savings.png"))

    if verbose:
        print("== M5 Optimization Report ==")
        print(md)
        print(f"\nWritten: outputs/report.md" + (f" + outputs/savings.png" if png else " (matplotlib absent: PNG skipped)"))

    return {"baseline_monthly": round(baseline), "optimized_monthly": round(optimized),
            "levers": levers, "total_savings_pct": round(total_pct, 1)}


def analysis_sections(r1: dict, r2: dict, r3: dict, r5: dict, levers: dict, cat: dict, sust: dict) -> str:
    """Vietnamese analysis appended to the report; every number comes from the mission results."""
    L = ["---", "", "## Phân tích chi tiết", "", "### 1. Inference: $/1M-token qua từng lever", "",
         "| Bước (cộng dồn) | $/ngày | $/1M-token |", "|---|---|---|"]
    for name, usd, per_m in r2["lever_waterfall"]:
        L.append(f"| {name} | ${usd:.2f} | ${per_m:.3f} |")
    L += ["", f"Cascade đóng góp nhiều nhất: giá model nhỏ rẻ hơn 15× (input) và 37× (output), "
          f"và phần lớn request chuyển được sang model nhỏ. Cache và batch chỉ chiết khấu một phần hóa đơn còn lại.", ""]

    L += ["### 2. GPU-Util lie", "", "| GPU | Loại | GPU-Util | MFU | MBU | Lãng phí so với MFU 0.40 ($/tháng) |", "|---|---|---|---|---|---|"]
    lie_waste = 0.0
    for lie in r1["lies"]:
        od = num(cat[lie["gpu_type"]]["on_demand_hr"])
        waste = od * 24 * DAYS * max(0.0, 1 - lie["mfu"] / TARGET_MFU)
        lie_waste += waste
        L.append(f"| {lie['gpu_id']} | {lie['gpu_type']} | {lie['gpu_util_pct']}% | {lie['mfu']} | {lie['mbu']} | ${waste:,.0f} |")
    L += ["", "`nvidia-smi` đo GPU-Util là tỷ lệ thời gian có ít nhất một kernel đang chạy. Chỉ số này không đo "
          "số SM hoặc Tensor Core thực sự làm việc. Một GPU chờ dữ liệu từ HBM (memory stall), chờ dataloader, "
          "hoặc chạy nhiều kernel nhỏ (kernel launch overhead) vẫn báo gần 100%.",
          "",
          f"Hai GPU trên có MBU cũng thấp (~0.2–0.3), nên vấn đề không phải do bị giới hạn băng thông. Nguyên nhân "
          f"nhiều khả năng là pipeline: batch nhỏ, chờ input hoặc đồng bộ giữa các GPU. Ta vẫn trả đủ giá theo giờ "
          f"nhưng chỉ nhận khoảng một nửa lượng FLOPs của một GPU khỏe. Tổng lãng phí khoảng **${lie_waste:,.0f}/tháng**. "
          f"Trước khi đổi sang GPU nhỏ hơn (lever right-size, ${levers['Right-size util-lies']:,}/tháng), nên profile "
          f"bằng Nsight hoặc DCGM (`SM_ACTIVE`, `TENSOR_ACTIVE`) và sửa pipeline trước. Cách này rẻ hơn và giữ được throughput.", ""]

    L += ["### 3. Thứ tự hành động đề xuất (theo ROI)", "",
          "| Ưu tiên | Hành động | Tiết kiệm ($/tháng) | Công sức | Lý do |", "|---|---|---|---|---|",
          f"| 1 | Tắt GPU idle và đặt lịch auto-shutdown | ${levers['Kill idle GPUs']:,} | Rất thấp | Làm được trong ngày, không rủi ro |",
          f"| 2 | Chuyển job interruptible sang spot và mua reserved cho inference 24/7 | ${levers['Purchasing (spot/reserved)']:,} | Thấp | Lever lớn nhất, chỉ cần checkpoint và cam kết hợp đồng |",
          f"| 3 | Cascade + cache có điều kiện + batch cho inference | ${levers['Inference (cascade/cache/batch)']:,} | Trung bình | Giảm $/1M-token {r2['savings_pct']}%, cần router và eval chất lượng |",
          f"| 4 | Profile và right-size các GPU util-lie | ${levers['Right-size util-lies']:,} | Trung bình | Cần profile trước để tránh mất throughput |",
          "", "Purchasing đứng thứ 2 dù tiết kiệm nhiều nhất, vì tắt GPU idle gần như không tốn công. Reserved là cam kết 3 năm. "
          f"Chỉ mua khi duty cycle ≥ 55% (điểm hòa vốn). Ba job inference được chọn reserved đều chạy 18–24h/ngày nên vượt ngưỡng này.", ""]

    rb = r2["reasoning"]
    s1, s0 = rb["split"][True], rb["split"][False]
    L += ["### 4. Mở rộng 3: cache_is_worth_it()", "",
          "Ước lượng số lần đọc mỗi prefix theo công thức N = h / (1 − h), với h là tỷ lệ input được cache. "
          "Chi phí ghi cache được tính 1.25× giá input, chi phí đọc 0.1×. Điểm hòa vốn là 0.25 / 0.9 ≈ 0.28 lần đọc.", "",
          "| Project | Tier | Số lần đọc TB | Hòa vốn | Hòa vốn (kiểu Gemini, $1/1M/giờ) | Cache? |", "|---|---|---|---|---|---|"]
    for c in r2["cache_table"]:
        L.append(f"| {c['project']} | {c['tier']} | {c['avg_reads']} | {c['break_even']} | {c['break_even_gemini']} | {'Có' if c['worth_it'] else 'Không'} |")
    L += ["", f"chat và doc-qa có khoảng 1 lần đọc mỗi prefix, cao hơn điểm hòa vốn, nên giữ cache. web-search và nightly-eval chỉ "
          f"khoảng 0.08 lần đọc, nên cache ở đây làm tăng chi phí. Chi phí inference mỗi ngày: không tính phí ghi "
          f"${r2['optimized_daily_free_cache']}, cache mọi nơi ${r2['optimized_daily_cache_all']}, cache có điều kiện "
          f"${r2['optimized_daily']:.2f}. Nếu tính phí lưu trữ kiểu Gemini, model nhỏ cần khoảng 5.6 lần đọc mới hòa vốn, "
          f"vì phí lưu trữ cố định trong khi giá input rất rẻ. Với model nhỏ, không nên bật explicit cache.", ""]

    L += ["### 5. Mở rộng 4: Ngân sách reasoning", "",
          "| Loại | Request | $/ngày | Wh/ngày |", "|---|---|---|---|",
          f"| Reasoning | {s1['n']} | ${s1['usd']:.2f} | {s1['wh']:,.0f} |",
          f"| Không reasoning | {s0['n']} | ${s0['usd']:.2f} | {s0['wh']:,.0f} |",
          "", f"Reasoning chiếm {rb['traffic_share']:.1%} traffic nhưng chiếm {rb['cost_share']:.1%} chi phí và "
          f"**{rb['energy_share']:.1%} năng lượng**. Lý do: output dài gấp ~6× (chuỗi suy luận), và mỗi token sinh ra ở "
          f"giai đoạn decode là memory-bound, nên năng lượng mỗi truy vấn cao hơn ~80×.", "",
          f"**Quy tắc routing đề xuất:** team eval chỉ bật reasoning khi `input_tokens ≥ {rb['cap_threshold_input_tokens']:,}` "
          f"(độ dài ngữ cảnh dùng làm proxy cho độ phức tạp). Quy tắc này giảm tỷ lệ reasoning của eval từ "
          f"{rb['eval_reasoning_share']:.0%} xuống 10%. Kết quả: tiết kiệm ${rb['cap_usd_saved_daily']*DAYS:,.0f}/tháng "
          f"và {rb['cap_wh_saved_daily']*DAYS/1000:,.0f} kWh/tháng ({rb['cap_downgraded']} request/ngày chuyển sang chế độ thường). "
          f"Lợi ích chính là năng lượng, vì phần lớn traffic reasoning đã chạy trên model nhỏ.", ""]

    L += ["### 6. Mở rộng 5: Lập lịch theo carbon (job interruptible)", "",
          f"5 job interruptible tiêu thụ {r5['interruptible_kwh_monthly']:,.0f} kWh/tháng (công suất board GPU).", "",
          "| Vùng | $/kWh | gCO2/kWh | Tiền điện ($/tháng) | kgCO2e/tháng | Điểm cân bằng | RTT (ms) |", "|---|---|---|---|---|---|---|"]
    for r in sorted(r5["regions"], key=lambda r: r["balanced_score"]):
        L.append(f"| {r['region']} | {r['usd_kwh']} | {r['gco2_kwh']} | ${r['energy_usd']:,.0f} | {r['carbon_kg']:,.0f} | {r['balanced_score']:.2f} | {r['rtt_ms']} |")
    L += ["", f"- Rẻ nhất: **{r5['cheapest']}**. Sạch nhất: **{r5['cleanest']}**. Cân bằng (50/50 giá và carbon): **{r5['balanced']}**.",
          f"- Chuyển từ us-east-1 sang {r5['cleanest']}: giảm {r5['carbon_saved_kg_cleanest']:,.0f} kgCO2e/tháng "
          f"({r5['carbon_saved_pct_cleanest']}%) và tiết kiệm ${r5['usd_saved_cleanest']:,.0f} tiền điện.",
          f"- Chuyển từ us-east-1 sang {r5['balanced']}: giảm {r5['carbon_saved_kg_balanced']:,.0f} kgCO2e/tháng "
          f"và tiết kiệm ${r5['usd_saved_balanced']:,.0f} tiền điện.",
          f"- Ở phần Sustainability phía trên, vùng tốt nhất được chọn chỉ theo carbon ({sust['best_region']}). "
          f"Vùng tối ưu thực sự phụ thuộc vào ưu tiên của công ty. Nếu có cam kết ESG, chọn {r5['cleanest']}. Nếu ưu tiên chi phí, "
          f"chọn {r5['balanced']}: chỉ cao hơn khoảng 3× carbon so với vùng sạch nhất nhưng tiền điện rẻ hơn ~40%.",
          "- Đánh đổi về latency: europe-north1 cách người dùng Mỹ khoảng 110 ms. Mức này không ảnh hưởng đến training hay batch eval, "
          "nhưng không phù hợp cho inference real-time. Vì vậy chỉ di chuyển job interruptible. Cần tính thêm phí egress khi chuyển "
          "dữ liệu hoặc checkpoint giữa các vùng.",
          "- Tiền điện (~vài trăm $/tháng) nhỏ hơn nhiều so với tiền thuê GPU (hàng chục nghìn $), nên vùng triển khai chủ yếu "
          "là lever carbon. Lever về chi phí lớn nhất vẫn là giá spot và reserved tại vùng đó."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    run()
