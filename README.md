# Prototype Code — MTDN Dissertation "Enhancing Accuracy and Trust in Hospitality F&B Reporting: A Hybrid AI-Blockchain Artefact"

Supplementary code referenced in **Annex E** of the dissertation (Carla Alexandra Viveiros Piedade, ISCTE MTDN). This repository contains the two functional components of the artefact evaluated in Chapter 4 (Demonstration & Evaluation) and in the companion ICDLT 2026 conference paper.

## Repository structure

```
prototype/
├── anomaly-detection/     AI layer — RQ1 evaluation (Isolation Forest vs. Autoencoder)
├── dlt-simulation/         DLT layer — throughput/volume simulation (ICDLT 2026 paper, Section 5)
├── chaincode/fnb-trust/    Hyperledger Fabric chaincode (StockLot + RecipeGovernance contracts)
└── replay-harness/         Node.js harness to replay simulated events as real Fabric transactions
```

### `anomaly-detection/`

`evaluate_anomaly_detection_v2.py` generates a synthetic, labelled F&B lot dataset (same baseline/deviation logic as Chapter 4, Section 4.4.1), fits Isolation Forest and an autoencoder (MLPRegressor reconstruction error) **unsupervised**, and scores both against the known injected labels. This is the script that produced Table 4.3 in the dissertation (precision/recall/F1 for both models, at contamination-based and best-F1 thresholds), plus the SHAP explanations referenced in Chapter 4.

Run:
```bash
pip install scikit-learn pandas numpy shap
python3 evaluate_anomaly_detection_v2.py
```
Outputs `RQ1_results_table_v2.csv` and `RQ1_shap_summary_v2.txt` (included here as already-generated reference outputs).

**Scope note:** validation reported in the dissertation uses synthetic data only. An attempt to validate against real operational data (HOST-vs-PHC reconciliation records) was made and is documented as a limitation and future-work item in Chapter 5 — it was not included as a result because both attempts were methodologically invalid (either circular/leaky by construction, or carrying no usable signal from a single-source feature set). See Chapter 5, Sections 5.3-5.4.

### `dlt-simulation/`

`simulation.py` is the domain-informed simulation behind the ICDLT 2026 paper's throughput/volume-reduction analysis (recipe-governance consolidation benefit). Explicitly documented in-file as a **simulation with assumed parameters**, not a benchmark of a deployed system.

`generate_events.py` reuses the same seed/scenario to emit the ordered sequence of real chaincode calls (`events.json`) needed to reproduce the scenario against a live Fabric network — this is the input consumed by `replay-harness/`.

Run:
```bash
python3 simulation.py
python3 generate_events.py
```

### `chaincode/fnb-trust/`

Hyperledger Fabric chaincode (Node.js, `fabric-contract-api`/`fabric-shim`) implementing the two smart contracts described in the dissertation's design (Chapter 3) and the ICDLT paper (Tables I & II):

- `lib/stockLotContract.js` — stock-lot registration, receipt validation, allocation.
- `lib/recipeGovernanceContract.js` — recipe (ficha técnica) governance lifecycle and flagging.

Deploy with the standard Hyperledger Fabric `test-network` (`fabric-samples`), channel `mychannel`, chaincode name `fnbtrust`.

### `replay-harness/`

`replay.js` reads `dlt-simulation/generate_events.py`'s `events.json` output and submits each event as a real transaction against the deployed `fnbtrust` chaincode, measuring actual latency/throughput with bounded concurrency.

```bash
cd replay-harness
npm install
node replay.js ../dlt-simulation/events.json --concurrency 10
```

## Requirements

| Component | Version |
|---|---|
| Python | >= 3.10 |
| Node.js | >= 18 |
| Docker + Docker Compose | >= 24.x (for Fabric test-network) |
| `fabric-samples` test-network | any recent release |

## Reproducibility

All scripts use fixed random seeds (`seed=42`). Running them in the order above (`anomaly-detection` -> `dlt-simulation` -> deploy `chaincode` -> `replay-harness`) reproduces the figures reported in Chapter 4 and in the ICDLT 2026 paper.
