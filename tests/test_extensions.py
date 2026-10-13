import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finops import pricing
from missions import m2_inference_levers, x5_carbon_scheduling


def test_cache_break_even_reads():
    # write premium 0.25 / read saving 0.9 -> ~0.28 reads, independent of price
    assert abs(pricing.cache_break_even_reads(3.0) - 0.25 / 0.9) < 1e-9
    assert abs(pricing.cache_break_even_reads(0.2) - 0.25 / 0.9) < 1e-9
    # a fixed storage fee hurts the cheap model far more than the expensive one
    assert pricing.cache_break_even_reads(0.2, 0.0, storage_per_m=1.0) > pricing.cache_break_even_reads(3.0, 0.0, storage_per_m=1.0)


def test_cache_is_worth_it():
    assert pricing.cache_is_worth_it(1.0, 3.0)
    assert not pricing.cache_is_worth_it(0.1, 3.0)
    assert not pricing.cache_is_worth_it(1.0, 0.2, write_premium=0.0, storage_per_m=1.0)


def test_m2_gated_cache_beats_cache_all():
    r = m2_inference_levers.run(verbose=False)
    assert r["optimized_daily"] <= r["optimized_daily_cache_all"]
    worth = {(c["project"], c["worth_it"]) for c in r["cache_table"]}
    assert ("chat", True) in worth and ("web-search", False) in worth


def test_m2_reasoning_budget():
    rb = m2_inference_levers.run(verbose=False)["reasoning"]
    assert rb["energy_share"] > rb["cost_share"] > rb["traffic_share"] > 0
    assert rb["cap_usd_saved_daily"] > 0 and rb["cap_wh_saved_daily"] > 0


def test_carbon_scheduling():
    r = x5_carbon_scheduling.run(verbose=False)
    assert r["cleanest"] == "europe-north1" and r["cheapest"] == "us-east-wa"
    assert r["carbon_saved_kg_cleanest"] > 0 and 0 < r["carbon_saved_pct_cleanest"] <= 100
