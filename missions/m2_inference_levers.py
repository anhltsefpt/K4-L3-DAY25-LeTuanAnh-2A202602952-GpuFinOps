"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Your-Turn extensions implemented here:
  - Ext 3: cache_is_worth_it() gates caching per (project, tier) and charges the write premium.
  - Ext 4: reasoning budget — $ and Wh split by is_reasoning + a capped-routing what-if.

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num
from finops import pricing, sustainability

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}
CACHE_WRITE_PREMIUM = 0.25          # Anthropic-style: cache write billed 1.25x base input
GEMINI_STORAGE_PER_M = 1.00         # Gemini-style explicit cache: ~$1/1M-token/hour storage
REASONING_OUTPUT_MULT = 6           # data/generate.py: reasoning inflates output tokens 6x
REASONING_CAP_SHARE = 0.10          # Ext 4 what-if: reasoning on at most 10% of eval traffic


def _parse(r: dict) -> dict:
    return {
        "project": r["project"] or "(untagged)", "team": r["team"], "tier": r["route_tier"],
        "inp": int(num(r["input_tokens"])), "out": int(num(r["output_tokens"])),
        "cached": int(num(r["cached_input_tokens"])),
        "batch": bool(int(num(r["is_batch"]))), "reasoning": bool(int(num(r["is_reasoning"]))),
    }


def cache_reads_by_group(rows: list[dict]) -> dict:
    """Estimate reads per cached prefix for each (project, tier).

    If a prefix is written once and read N times, the cached share of input is
    h = N / (N + 1), so N = h / (1 - h).
    """
    inp, cached = defaultdict(int), defaultdict(int)
    for r in rows:
        inp[(r["project"], r["tier"])] += r["inp"]
        cached[(r["project"], r["tier"])] += r["cached"]
    reads = {}
    for k in inp:
        h = cached[k] / inp[k] if inp[k] else 0.0
        reads[k] = h / (1.0 - h) if h < 1 else float("inf")
    return reads


def _opt_cost(r: dict, reads: dict, cache_policy: str) -> float:
    """Optimized cost of one request (cascade + batch + cache under a policy).

    cache_policy: 'free'  -> cache hits, write cost ignored (original lab model)
                  'all'   -> cache everything, pay the write premium
                  'gated' -> cache only where cache_is_worth_it(), pay the write premium
    """
    pin, pout = MODEL_PRICES[r["tier"]]
    n = reads[(r["project"], r["tier"])]
    use_cache = cache_policy == "free" or cache_policy == "all" or (
        cache_policy == "gated" and pricing.cache_is_worth_it(n, pin, CACHE_WRITE_PREMIUM))
    cached = r["cached"] if use_cache else 0
    cost = pricing.request_cost(r["inp"], r["out"], pin, pout, cached_in=cached, batch=r["batch"])
    if use_cache and cache_policy != "free" and n > 0:
        write_tokens = cached / n                   # one write amortized over n reads
        cost += (write_tokens / 1e6) * pin * CACHE_WRITE_PREMIUM * (0.5 if r["batch"] else 1.0)
    return cost


def lever_waterfall(rows: list[dict], reads: dict) -> list[tuple[str, float]]:
    """Daily cost after each lever is stacked: baseline -> cascade -> cache -> batch."""
    lin, lout = MODEL_PRICES["large"]
    steps = {"baseline (all large)": 0.0, "+ cascade": 0.0, "+ prompt cache (gated)": 0.0, "+ batch API": 0.0}
    for r in rows:
        pin, pout = MODEL_PRICES[r["tier"]]
        steps["baseline (all large)"] += pricing.request_cost(r["inp"], r["out"], lin, lout)
        steps["+ cascade"] += pricing.request_cost(r["inp"], r["out"], pin, pout)
        no_batch = dict(r, batch=False)
        steps["+ prompt cache (gated)"] += _opt_cost(no_batch, reads, "gated")
        steps["+ batch API"] += _opt_cost(r, reads, "gated")
    return list(steps.items())


def reasoning_budget(rows: list[dict], reads: dict) -> dict:
    """Ext 4: $ and Wh split by reasoning flag, plus a capped-reasoning routing what-if."""
    split = {True: {"n": 0, "tokens": 0, "usd": 0.0, "wh": 0.0}, False: {"n": 0, "tokens": 0, "usd": 0.0, "wh": 0.0}}
    for r in rows:
        tok = r["inp"] + r["out"]
        s = split[r["reasoning"]]
        s["n"] += 1
        s["tokens"] += tok
        s["usd"] += _opt_cost(r, reads, "gated")
        s["wh"] += sustainability.wh_per_query(tok, is_reasoning=r["reasoning"])

    # What-if: in the eval team keep reasoning only for the longest-context tasks
    # (input length = complexity proxy) so reasoning covers <= 10% of eval traffic.
    eval_rows = [r for r in rows if r["team"] == "eval"]
    eval_reasoning = sorted((r for r in eval_rows if r["reasoning"]), key=lambda r: r["inp"], reverse=True)
    keep_n = int(len(eval_rows) * REASONING_CAP_SHARE)
    kept, downgraded = eval_reasoning[:keep_n], eval_reasoning[keep_n:]
    threshold = min((r["inp"] for r in kept), default=0)
    usd_before = usd_after = wh_before = wh_after = 0.0
    for r in downgraded:
        plain = dict(r, out=r["out"] // REASONING_OUTPUT_MULT, reasoning=False)
        usd_before += _opt_cost(r, reads, "gated")
        usd_after += _opt_cost(plain, reads, "gated")
        wh_before += sustainability.wh_per_query(r["inp"] + r["out"], is_reasoning=True)
        wh_after += sustainability.wh_per_query(plain["inp"] + plain["out"], is_reasoning=False)

    total_n = split[True]["n"] + split[False]["n"]
    total_usd = split[True]["usd"] + split[False]["usd"]
    total_wh = split[True]["wh"] + split[False]["wh"]
    return {
        "split": split,
        "traffic_share": split[True]["n"] / total_n if total_n else 0.0,
        "cost_share": split[True]["usd"] / total_usd if total_usd else 0.0,
        "energy_share": split[True]["wh"] / total_wh if total_wh else 0.0,
        "eval_reasoning_share": len(eval_reasoning) / len(eval_rows) if eval_rows else 0.0,
        "cap_threshold_input_tokens": threshold,
        "cap_downgraded": len(downgraded),
        "cap_usd_saved_daily": usd_before - usd_after,
        "cap_wh_saved_daily": wh_before - wh_after,
    }


def run(verbose: bool = True) -> dict:
    rows = [_parse(r) for r in load_csv("token_usage.csv")]
    reads = cache_reads_by_group(rows)
    lin, lout = MODEL_PRICES["large"]

    base_cost = sum(pricing.request_cost(r["inp"], r["out"], lin, lout) for r in rows)
    total_tokens = sum(r["inp"] + r["out"] for r in rows)
    free_cost = sum(_opt_cost(r, reads, "free") for r in rows)     # original lab model
    all_cost = sum(_opt_cost(r, reads, "all") for r in rows)       # cache everywhere, pay writes
    opt_cost = sum(_opt_cost(r, reads, "gated") for r in rows)     # Ext 3 policy

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0

    cache_table = []
    for (project, tier), n in sorted(reads.items()):
        pin = MODEL_PRICES[tier][0]
        cache_table.append({
            "project": project, "tier": tier, "avg_reads": round(n, 2),
            "break_even": round(pricing.cache_break_even_reads(pin, CACHE_WRITE_PREMIUM), 2),
            "break_even_gemini": round(pricing.cache_break_even_reads(pin, 0.0, storage_per_m=GEMINI_STORAGE_PER_M), 2),
            "worth_it": pricing.cache_is_worth_it(n, pin, CACHE_WRITE_PREMIUM),
        })
    waterfall = lever_waterfall(rows, reads)
    rb = reasoning_budget(rows, reads)

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + gated caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")

        print("\n-- $/1M-token by lever (stacked) --")
        for name, cost in waterfall:
            print(f"  {name:26} ${cost:7.2f}/day   ${pricing.dollars_per_million(cost, total_tokens):.3f}/1M-token")

        print("\n-- Ext 3: cache_is_worth_it (write premium 1.25x, read 0.1x) --")
        print(f"  {'project':13}{'tier':7}{'reads':>7}{'BE':>7}{'BE(gemini)':>12}  cache?")
        for c in cache_table:
            print(f"  {c['project']:13}{c['tier']:7}{c['avg_reads']:>7}{c['break_even']:>7}{c['break_even_gemini']:>12}  {'YES' if c['worth_it'] else 'no'}")
        print(f"  cache cost model: free writes ${free_cost:.2f}/day | cache-all ${all_cost:.2f}/day | gated ${opt_cost:.2f}/day")

        s1, s0 = rb["split"][True], rb["split"][False]
        print("\n-- Ext 4: reasoning budget --")
        print(f"  reasoning    : {s1['n']:>5} req  ${s1['usd']:6.2f}/day  {s1['wh']:8.0f} Wh/day")
        print(f"  non-reasoning: {s0['n']:>5} req  ${s0['usd']:6.2f}/day  {s0['wh']:8.0f} Wh/day")
        print(f"  reasoning = {rb['traffic_share']:.1%} of traffic, {rb['cost_share']:.1%} of $, {rb['energy_share']:.1%} of Wh")
        print(f"  eval team reasoning share: {rb['eval_reasoning_share']:.0%} -> cap {REASONING_CAP_SHARE:.0%}"
              f" (reasoning only if input_tokens >= {rb['cap_threshold_input_tokens']:,})")
        print(f"  cap saves ${rb['cap_usd_saved_daily']:.2f}/day (${rb['cap_usd_saved_daily']*30:,.0f}/month)"
              f" and {rb['cap_wh_saved_daily']/1000:.1f} kWh/day ({rb['cap_downgraded']} requests downgraded)")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "optimized_daily_free_cache": round(free_cost, 2), "optimized_daily_cache_all": round(all_cost, 2),
        "lever_waterfall": [(n, round(c, 2), round(pricing.dollars_per_million(c, total_tokens), 3)) for n, c in waterfall],
        "cache_table": cache_table, "reasoning": rb,
    }


if __name__ == "__main__":
    run()
