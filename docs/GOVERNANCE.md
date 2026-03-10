# Governance, Risk, and Ethical AI

This document captures the Responsible AI guardrails for the churn model and maps them to the current implementation status.

---

## 1) Bias Mitigation and Fairness Audit

Risk:
- Models can inadvertently penalize specific demographics (Age, Geography, Gender) if historical data is biased.

Guardrails:
- Feature importance review to ensure protected attributes are not relied upon.
- Prefer behavioral signals (usage frequency, support tickets) over demographic traits.
- Optional exclusion of protected attributes from training and inference.

Status in this repo:
- Implemented: Feature importance visualization for model inspection.
- Not yet implemented: Formal fairness audit by demographic slices.
- Not yet implemented: Explicit exclusion policy for protected attributes.

---

## 2) Data Privacy and Compliance (DPDP / GDPR)

Risk:
- Leaking PII through model features, training logs, or artifacts.

Guardrails:
- Ingestion anonymization or pseudonymization.
- No raw PII used for training; features are transformed into numerical vectors.
- Logging and artifact storage policies that avoid PII.

Status in this repo:
- Implemented: Direct identifier removal (customer ID) during preprocessing.
- Not yet implemented: Dedicated anonymization layer and PII-safe logging policies.

---

## 3) Model Uncertainty and Action Gating

Risk:
- Acting on low-confidence predictions can lead to poor customer experience or wasted retention budget.

Guardrails:
- Predict probability scores rather than only class labels.
- Gate high-cost interventions on a high confidence threshold (e.g., >= 0.90).
- Track calibration and precision-recall behavior to set thresholds.

Status in this repo:
- Implemented: Probability outputs, threshold tuning, PR and calibration curves.
- Not yet implemented: Business rule gating for high-cost interventions.

---

## 4) Human-in-the-Loop Strategy

Risk:
- Full automation without oversight can cause drift and opaque decisioning.

Guardrails:
- For high-value accounts, the model provides reason codes and defers final decision to a human owner.
- Use explainability tooling (e.g., SHAP) for case-level insights.

Status in this repo:
- Not yet implemented: SHAP explanations and decision handoff workflow.

---

## Roadmap

Planned additions:
- Fairness audit metrics by protected groups (precision, recall, FPR, FNR parity).
- Configurable protected-attribute exclusion policy.
- SHAP-based reason codes and human review handoff.
- PII-safe logging and anonymization module at ingestion.
- Business action gating based on confidence thresholds.
