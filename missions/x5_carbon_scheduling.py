"""Ext 5 — Carbon-aware Scheduling: move interruptible jobs to cheap + clean regions (deck §11).

Interruptible jobs (training, fine-tune, batch eval, dev) checkpoint anyway, so they are
not tied to the region closest to users. This compares all regions on electricity $ and
gCO2e for those jobs, using the same monthly GPU-hours as M3.

Run: python missions/x5_carbon_scheduling.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops.sustainability import REGION_CARBON, REGION_PRICE_KWH, carbon_g, energy_cost_usd

DAYS = 30
CURRENT_REGION = "us-east-1"
# Illustrative RTT from US-East users; interruptible batch work does not serve users live.
REGION_RTT_MS = {"us-east-1": 10, "us-east-wa": 15, "us-west-2": 70, "europe-north1": 110, "europe-central2": 120}


def interruptible_energy_wh() -> tuple[float, list[dict]]:
    """Monthly GPU energy (Wh, board power) for every interruptible job."""
    cat = catalog_by_type()
    jobs = []
    for j in load_csv("workloads.csv"):
        if not int(num(j["interruptible"])):
            continue
        gpu_hours = num(j["hours_per_day"]) * DAYS * int(num(j["num_gpus"]))
        jobs.append({"job_id": j["job_id"], "gpu_type": j["gpu_type"],
                     "wh": gpu_hours * num(cat[j["gpu_type"]]["watts"])})
    return sum(j["wh"] for j in jobs), jobs


def region_table(wh: float) -> list[dict]:
    """Electricity $ and carbon of `wh` in every region, plus a balanced score."""
    max_p, max_c = max(REGION_PRICE_KWH.values()), max(REGION_CARBON.values())
    rows = []
    for reg in REGION_CARBON:
        rows.append({
            "region": reg, "usd_kwh": REGION_PRICE_KWH[reg], "gco2_kwh": REGION_CARBON[reg],
            "energy_usd": energy_cost_usd(wh, reg), "carbon_kg": carbon_g(wh, reg) / 1000.0,
            # equal weight on normalized price and carbon intensity (lower is better)
            "balanced_score": 0.5 * REGION_PRICE_KWH[reg] / max_p + 0.5 * REGION_CARBON[reg] / max_c,
            "rtt_ms": REGION_RTT_MS.get(reg),
        })
    return rows


def run(verbose: bool = True) -> dict:
    wh, jobs = interruptible_energy_wh()
    table = region_table(wh)
    by = {r["region"]: r for r in table}
    cheapest = min(table, key=lambda r: r["usd_kwh"])["region"]
    cleanest = min(table, key=lambda r: r["gco2_kwh"])["region"]
    balanced = min(table, key=lambda r: r["balanced_score"])["region"]
    cur = by[CURRENT_REGION]

    per_job = [{"job_id": j["job_id"], "kwh": j["wh"] / 1000,
                "kg_now": carbon_g(j["wh"], CURRENT_REGION) / 1000,
                "kg_clean": carbon_g(j["wh"], cleanest) / 1000} for j in jobs]

    result = {
        "interruptible_kwh_monthly": round(wh / 1000, 1),
        "regions": table, "per_job": per_job,
        "cheapest": cheapest, "cleanest": cleanest, "balanced": balanced,
        "carbon_saved_kg_cleanest": round(cur["carbon_kg"] - by[cleanest]["carbon_kg"], 1),
        "carbon_saved_pct_cleanest": round((1 - by[cleanest]["carbon_kg"] / cur["carbon_kg"]) * 100, 1),
        "usd_saved_cleanest": round(cur["energy_usd"] - by[cleanest]["energy_usd"], 2),
        "carbon_saved_kg_balanced": round(cur["carbon_kg"] - by[balanced]["carbon_kg"], 1),
        "usd_saved_balanced": round(cur["energy_usd"] - by[balanced]["energy_usd"], 2),
    }

    if verbose:
        print("== Ext 5 Carbon-aware Scheduling ==")
        print(f"interruptible jobs: {len(jobs)}  energy: {wh/1000:,.0f} kWh/month (GPU board power)")
        print(f"\n{'job':18}{'kWh/mo':>9}{'kgCO2 '+CURRENT_REGION:>20}{'kgCO2 '+cleanest:>22}")
        for p in per_job:
            print(f"{p['job_id']:18}{p['kwh']:>9,.0f}{p['kg_now']:>20,.1f}{p['kg_clean']:>22,.1f}")
        print(f"\n{'region':17}{'$/kWh':>7}{'gCO2/kWh':>10}{'energy $':>10}{'kgCO2e':>9}{'score':>7}{'RTT ms':>8}")
        for r in sorted(table, key=lambda r: r["balanced_score"]):
            print(f"{r['region']:17}{r['usd_kwh']:>7.3f}{r['gco2_kwh']:>10}{r['energy_usd']:>10,.0f}"
                  f"{r['carbon_kg']:>9,.0f}{r['balanced_score']:>7.2f}{r['rtt_ms']:>8}")
        print(f"\ncheapest $: {cheapest} | cleanest CO2: {cleanest} | balanced: {balanced}")
        print(f"{CURRENT_REGION} -> {cleanest}: -{result['carbon_saved_kg_cleanest']:,} kgCO2e/month "
              f"({result['carbon_saved_pct_cleanest']}%), energy -${result['usd_saved_cleanest']:,}/month")
        print(f"{CURRENT_REGION} -> {balanced}: -{result['carbon_saved_kg_balanced']:,} kgCO2e/month, "
              f"energy -${result['usd_saved_balanced']:,}/month")
    return result


if __name__ == "__main__":
    run()
