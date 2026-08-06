"""
Event generator for the Fabric replay harness — ICDLT 2026 paper prototype.

Reuses the EXACT same RNG seed and parameters as simulation.py so that the
underlying scenario (which lots are flagged, which articles are stale, the
45.8% investigation-volume reduction in Table III) is identical. What differs
is the OUTPUT: instead of just counting events, this script emits the full,
ordered sequence of real chaincode calls needed to reproduce that scenario
against a live Hyperledger Fabric network, respecting the chaincode's actual
state-machine preconditions (e.g. validateReceipt requires a prior
stockRegister; ftFlagged requires the recipe to already be Active).

This will produce MORE real transactions than the paper's current Table IV,
because Table IV counts some function pairs (stockRegister/validateReceipt,
confirmReception/allocateStock) as a single combined unit, and does not
separately count the recipe-initialization or ftSubmit calls a real deployment
requires. That is expected and intentional (see paper Section 5.3 update) —
this script produces the numbers Table IV should be updated to once the
replay harness has actually run against the network.

Output: events.json — an ordered list of {contract, function, args} objects,
one per real transaction to submit.

Run: python3 generate_events.py
"""

import json
import numpy as np

rng = np.random.default_rng(seed=42)

# ---------------------------------------------------------------------------
# Same parameters as simulation.py (ASSUMPTIONS — see paper Section 5.2)
# ---------------------------------------------------------------------------
N_MONTHS = 6
N_SECTIONS = 6
N_ARTICLES = 300
LOTS_PER_MONTH = 250
STALE_RECIPE_RATE = 0.10
CONSUMPTION_ANOMALY_RATE = 0.04
STALE_RECIPE_TRIGGER_RATE = 0.70
CONSUMPTION_EVENTS_PER_LOT = 4

articles = np.arange(N_ARTICLES)
stale_recipe = rng.random(N_ARTICLES) < STALE_RECIPE_RATE

total_lots = LOTS_PER_MONTH * N_MONTHS
lot_articles = rng.integers(0, N_ARTICLES, size=total_lots)
lot_stale = stale_recipe[lot_articles]

consumption_anomaly = rng.random(total_lots) < CONSUMPTION_ANOMALY_RATE
recipe_driven_anomaly = lot_stale & (rng.random(total_lots) < STALE_RECIPE_TRIGGER_RATE)

flagged = consumption_anomaly | recipe_driven_anomaly
pure_consumption = flagged & consumption_anomaly & ~recipe_driven_anomaly
pure_recipe = flagged & recipe_driven_anomaly & ~consumption_anomaly
co_occurring = flagged & recipe_driven_anomaly & consumption_anomaly

recipe_driven_lots_mask = pure_recipe | co_occurring
flagged_articles = sorted(set(lot_articles[recipe_driven_lots_mask].tolist()))

# ---------------------------------------------------------------------------
# Build the ordered event list
# ---------------------------------------------------------------------------
events = []


def add(contract, function, args, key):
    # `key` groups events that MUST execute in the order generated (e.g. every
    # event touching the same lotId, or the same recipe/article). Events with
    # DIFFERENT keys touch independent ledger state and are safe to submit
    # concurrently. The replay harness uses this to parallelise safely.
    events.append({"contract": contract, "function": function, "args": [str(a) for a in args], "key": key})


# --- Recipe initialization for the flagged articles only -------------------
# A recipe must exist and be Active before it can ever be flagged. Non-flagged
# articles' recipes are never referenced by a RecipeGovernanceContract call in
# this replay (recordConsumption does not validate ftId existence), so we only
# need to establish the baseline for articles that will later be flagged.
ft_active = {}  # articleId -> current active ftId
for art in flagged_articles:
    key = f"ART-{art}"
    old_ft = f"FT-{art:04d}-v1"
    add("RecipeGovernanceContract", "ftSubmit", [old_ft, art, "1", "system-init"], key)
    add("RecipeGovernanceContract", "ftValidate", [old_ft, "fnb-mgmt-1", f"hash-init-{art}"], key)
    add("RecipeGovernanceContract", "ftActivate", [old_ft, art, "2026-01-01"], key)
    ft_active[art] = old_ft

# --- Stock-lot lifecycle, per lot ------------------------------------------
anomalous_lot_indices = np.where(pure_consumption | co_occurring)[0]
for i in range(total_lots):
    lot_id = f"LOT-{i:05d}"
    supplier = f"SUP-{(i % 15) + 1:02d}"
    po_ref = f"PO-{i:05d}"
    invoice_ref = f"INV-{i:05d}"
    section = f"section-{(i % N_SECTIONS) + 1}"
    req_id = f"REQ-{i:05d}"
    article = int(lot_articles[i])
    ft_id = ft_active.get(article, f"FT-{article:04d}-v1")  # referenced but not necessarily on-ledger
    ts = f"2026-{(i % N_MONTHS) + 1:02d}-{(i % 28) + 1:02d}T12:00:00Z"
    lot_key = lot_id  # every event on this lot must stay strictly ordered

    add("StockLotContract", "stockRegister", [lot_id, supplier, po_ref], lot_key)
    add("StockLotContract", "validateReceipt", [lot_id, invoice_ref], lot_key)
    add("StockLotContract", "confirmReception", [lot_id, "procurement-node-1", ts], lot_key)
    add("StockLotContract", "allocateStock", [lot_id, section, req_id], lot_key)

    for k in range(CONSUMPTION_EVENTS_PER_LOT):
        pos_ref = f"POS-{i:05d}-{k}"
        add("StockLotContract", "recordConsumption", [lot_id, ft_id, pos_ref, ts], lot_key)

    if i in anomalous_lot_indices:
        shap_score = f"{rng.uniform(0.4, 0.95):.3f}"
        add("StockLotContract", "anomalyDetected", [lot_id, shap_score, "consumption"], lot_key)
        add("StockLotContract", "correctionRecorded", [lot_id, "write-off-confirmed", "fnb-mgmt-1", ts], lot_key)

# --- Recipe-drift correction cycle for each flagged article -----------------
# Same key as the initial submission for this article: ftFlagged/ftSupersede
# depend on that earlier sequence having already committed.
for art in flagged_articles:
    key = f"ART-{art}"
    old_ft = ft_active[art]
    add("RecipeGovernanceContract", "ftFlagged", [old_ft, art, f"ev-{art}", "drift"], key)
    new_ft = f"FT-{art:04d}-v2"
    add("RecipeGovernanceContract", "ftSubmit", [new_ft, art, "2", "fnb-ops-1"], key)
    add("RecipeGovernanceContract", "ftValidate", [new_ft, "fnb-mgmt-1", f"hash-v2-{art}"], key)
    add("RecipeGovernanceContract", "ftActivate", [new_ft, art, "2026-04-01"], key)
    add("RecipeGovernanceContract", "ftSupersede", [old_ft, new_ft, "2026-04-01T00:00:00Z"], key)

# --- Monthly inventory finalization -----------------------------------------
# Each (period, section) pair writes an independent key — no ordering needed.
for m in range(N_MONTHS):
    for s in range(N_SECTIONS):
        period = f"2026-{m + 1:02d}"
        section = f"section-{s + 1}"
        add("StockLotContract", "inventoryFinalized", [period, section, f"hash-{m}-{s}"], f"INV-{period}-{section}")

# ---------------------------------------------------------------------------
with open("events.json", "w") as f:
    json.dump(events, f, indent=1)

print(f"Total lots: {total_lots}")
print(f"Flagged articles (recipe-drift correction cycle): {len(flagged_articles)}")
print(f"Anomalous lots (anomalyDetected + correctionRecorded): {len(anomalous_lot_indices)}")
print(f"Total real transactions generated: {len(events)}")
print("Written to events.json")
