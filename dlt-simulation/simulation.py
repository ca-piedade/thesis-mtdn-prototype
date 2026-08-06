"""
Simulation study for the ICDLT 2026 paper — Section 5 (Evaluation).

IMPORTANT — honesty of framing:
This is NOT a benchmark of a deployed system. It is a synthetic simulation with
domain-informed parameters (documented below, chosen to be consistent with the
"recipe gap" root cause identified in the dissertation's As-Is process analysis,
not measured from live production data). Its purpose is twofold:

  1. Quantify the operational benefit of the recipe-governance lifecycle
     (Section 3.2): how much individual, case-by-case investigation volume it
     eliminates by consolidating recipe-driven anomaly clusters into a single
     ftFlagged review, versus treating every flagged deviation as an
     independent case (the As-Is / no-recipe-governance baseline).
  2. Translate the resulting simulated transaction volume into an on-chain
     throughput feasibility check against published Hyperledger Fabric
     benchmarks [6] (Zulkarnain et al., 2025), rather than against original
     measurement (no Fabric deployment was performed).

The simulation does NOT claim to evaluate AI detection accuracy (that is the
dissertation's AI layer, out of scope for this DLT-focused paper) — anomaly
ground truth is generated directly, and the ledger's role being evaluated is
deterministic event routing/consolidation, not statistical inference.

Run: python3 simulation.py
"""

import numpy as np

rng = np.random.default_rng(seed=42)

# ---------------------------------------------------------------------------
# Domain-informed simulation parameters (ASSUMPTIONS — not measured data)
# ---------------------------------------------------------------------------
N_MONTHS = 6
N_SECTIONS = 6                  # Kitchen, Bar, Restaurant, Banquets, Room Service, Pool Bar
N_ARTICLES = 300                # distinct F&B articles tracked across the property
LOTS_PER_MONTH = 250            # stock lots processed per month, all sections combined
STALE_RECIPE_RATE = 0.10        # fraction of articles operating under a stale/incorrect
                                 # ficha tecnica at simulation start (dissertation's
                                 # "recipe gap" pain point; no measured baseline exists,
                                 # so a conservative round figure is assumed)
CONSUMPTION_ANOMALY_RATE = 0.04 # baseline probability a lot shows a genuine consumption
                                 # anomaly independent of recipe correctness (assumed)
STALE_RECIPE_TRIGGER_RATE = 0.70  # probability a lot using a stale recipe crosses the
                                   # anomaly-detection threshold (systematic bias affects
                                   # most, not all, transactions using that recipe)
CONSUMPTION_EVENTS_PER_LOT = 4  # average recordConsumption events generated per lot

# ---------------------------------------------------------------------------
# Simulate articles: assign each a "stale recipe" ground-truth flag
# ---------------------------------------------------------------------------
articles = np.arange(N_ARTICLES)
stale_recipe = rng.random(N_ARTICLES) < STALE_RECIPE_RATE
n_stale_articles = int(stale_recipe.sum())

# ---------------------------------------------------------------------------
# Simulate lots over N_MONTHS, each lot tied to a random article
# ---------------------------------------------------------------------------
total_lots = LOTS_PER_MONTH * N_MONTHS
lot_articles = rng.integers(0, N_ARTICLES, size=total_lots)
lot_stale = stale_recipe[lot_articles]

# Ground-truth anomaly generation per lot
consumption_anomaly = rng.random(total_lots) < CONSUMPTION_ANOMALY_RATE
recipe_driven_anomaly = lot_stale & (rng.random(total_lots) < STALE_RECIPE_TRIGGER_RATE)

flagged = consumption_anomaly | recipe_driven_anomaly
pure_consumption = flagged & consumption_anomaly & ~recipe_driven_anomaly
pure_recipe = flagged & recipe_driven_anomaly & ~consumption_anomaly
co_occurring = flagged & recipe_driven_anomaly & consumption_anomaly

n_flagged = int(flagged.sum())
n_pure_consumption = int(pure_consumption.sum())
n_pure_recipe = int(pure_recipe.sum())
n_co_occurring = int(co_occurring.sum())

# ---------------------------------------------------------------------------
# Baseline (As-Is / architecture WITHOUT recipe governance):
# every flagged lot is investigated individually as its own anomalyDetected case.
# ---------------------------------------------------------------------------
baseline_investigations = n_flagged

# ---------------------------------------------------------------------------
# With recipe governance: recipe-driven anomalies (pure + co-occurring, since the
# recipe contribution is still identified even when a genuine anomaly co-occurs)
# collapse to ONE ftFlagged review per distinct stale article, instead of one
# anomalyDetected event per affected lot. Pure consumption anomalies and the
# consumption-side of co-occurring cases are still investigated individually.
# ---------------------------------------------------------------------------
recipe_driven_lots_mask = pure_recipe | co_occurring
affected_articles_with_flagged_lots = np.unique(lot_articles[recipe_driven_lots_mask])
n_ftFlagged_reviews = len(affected_articles_with_flagged_lots)

with_governance_investigations = n_pure_consumption + n_co_occurring + n_ftFlagged_reviews
# (co-occurring lots still need individual consumption-side review; the recipe
#  side of those same lots is folded into the n_ftFlagged_reviews count once per article)

investigation_reduction = 1 - (with_governance_investigations / baseline_investigations)

# ---------------------------------------------------------------------------
# Table III — Simulated anomaly attribution and investigation-volume reduction
# ---------------------------------------------------------------------------
print("=" * 78)
print("TABLE III — Simulated anomaly attribution and investigation-volume")
print(f"reduction ({N_MONTHS}-month horizon, {total_lots} lots, {N_ARTICLES} articles,")
print(f"{n_stale_articles} with a stale/incorrect recipe at simulation start)")
print("=" * 78)
print(f"{'Metric':<62}{'Value':>14}")
print("-" * 78)
print(f"{'Total lots simulated':<62}{total_lots:>14}")
print(f"{'Total flagged anomalies':<62}{n_flagged:>14}")
print(f"{'  - pure consumption anomalies':<62}{n_pure_consumption:>14}")
print(f"{'  - pure recipe-driven anomalies':<62}{n_pure_recipe:>14}")
print(f"{'  - co-occurring (both root causes)':<62}{n_co_occurring:>14}")
print(f"{'Baseline investigations (no recipe governance)':<62}{baseline_investigations:>14}")
print(f"{'Investigations with recipe governance':<62}{with_governance_investigations:>14}")
print(f"{'  of which ftFlagged recipe reviews (distinct articles)':<62}{n_ftFlagged_reviews:>14}")
print(f"{'Investigation-volume reduction':<62}{investigation_reduction*100:>13.1f}%")
print()

# ---------------------------------------------------------------------------
# Table IV — Estimated on-chain transaction volume vs. published Fabric
# throughput [6] (Zulkarnain et al. 2025) — feasibility, not measurement.
# ---------------------------------------------------------------------------
consumption_events = total_lots * CONSUMPTION_EVENTS_PER_LOT
tx_counts_6mo = {
    "stockRegister / validateReceipt": total_lots,
    "confirmReception / allocateStock": total_lots,
    "recordConsumption": consumption_events,
    "anomalyDetected": n_pure_consumption + n_co_occurring,
    "ftFlagged": n_ftFlagged_reviews,
    "correctionRecorded": n_pure_consumption + n_co_occurring,
    "ftValidate / ftActivate / ftSupersede": n_ftFlagged_reviews,
    "inventoryFinalized": N_SECTIONS * N_MONTHS,
}
total_tx_6mo = sum(tx_counts_6mo.values())
days = N_MONTHS * 30
avg_tx_per_day = total_tx_6mo / days
peak_tx_per_day = avg_tx_per_day * 3  # assume a 3x peak-day factor (month-end close)
peak_tx_per_second = peak_tx_per_day / (8 * 3600)  # spread over an 8h operational window

# Published Fabric throughput range from [6], transfer chaincode (closest analogue
# to our write transactions: read + write with validation logic):
FABRIC_LOW_TPS = 18.4    # single-board computer tier, transfer chaincode
FABRIC_MID_TPS = 63.5    # mid-range laptop tier, transfer chaincode
FABRIC_HIGH_TPS = 336.8  # high-performance laptop tier, transfer chaincode

print("=" * 78)
print("TABLE IV — Estimated on-chain transaction volume vs. published Fabric")
print("throughput [6] (feasibility check, not original benchmarking)")
print("=" * 78)
for k, v in tx_counts_6mo.items():
    print(f"{k:<50}{v:>10} tx / {N_MONTHS} mo")
print("-" * 78)
print(f"{'Total write transactions, ' + str(N_MONTHS) + '-month horizon':<50}{total_tx_6mo:>10}")
print(f"{'Average transactions / day':<50}{avg_tx_per_day:>10.1f}")
print(f"{'Assumed peak day (3x average, month-end close)':<50}{peak_tx_per_day:>10.1f}")
print(f"{'Peak-day required throughput (tx/s, 8h window)':<50}{peak_tx_per_second:>10.4f}")
print("-" * 78)
print(f"{'Published Fabric throughput, low tier (SBC) [6]':<50}{FABRIC_LOW_TPS:>10.1f} tx/s")
print(f"{'  -> headroom factor vs. our peak requirement':<50}{FABRIC_LOW_TPS/peak_tx_per_second:>10.0f}x")
print(f"{'Published Fabric throughput, mid tier [6]':<50}{FABRIC_MID_TPS:>10.1f} tx/s")
print(f"{'  -> headroom factor':<50}{FABRIC_MID_TPS/peak_tx_per_second:>10.0f}x")
print(f"{'Published Fabric throughput, high tier [6]':<50}{FABRIC_HIGH_TPS:>10.1f} tx/s")
print(f"{'  -> headroom factor':<50}{FABRIC_HIGH_TPS/peak_tx_per_second:>10.0f}x")
