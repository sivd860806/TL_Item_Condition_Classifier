.PHONY: setup eda baseline text-only train train-xgb ablation threshold errors eval report all clean help

help:
	@echo "Targets disponibles:"
	@echo "  setup           Instala deps (uv sync)"
	@echo "  eda             Corre el EDA y genera reports/eda_findings.md + figuras"
	@echo "  baseline        Entrena LR full (tabular + texto)"
	@echo "  text-only       Entrena LR solo TF-IDF sobre title (aislamiento de señal)"
	@echo "  train           Entrena LightGBM con Optuna (modelo principal)"
	@echo "  train-xgb       Entrena XGBoost (comparativo de familias)"
	@echo "  ablation        Ablacion con vs sin listing_type_id"
	@echo "  threshold       Tuning de threshold + curvas PR/ROC"
	@echo "  errors          Top-50 FP + Top-50 FN con SHAP"
	@echo "  eval            Eval final sobre held-out X_test (una sola vez)"
	@echo "  report          Recordatorio: el reporte vive en reports/REPORT.md"
	@echo "  all             Encadena setup + eda + baselines + train + ablation + eval"
	@echo "  clean           Borra modelos, figuras y caches"

setup:
	uv sync

eda:
	uv run python -m src.eda.runner

baseline:
	uv run python -m src.models.lr_baseline

text-only:
	uv run python -m src.models.lr_text_only

train:
	uv run python -m src.models.lightgbm_model

train-xgb:
	uv run python -m src.models.xgboost_model

ablation:
	uv run python -m src.experiments.ablation_listing_type

threshold:
	uv run python -m src.eval.threshold

errors:
	uv run python -m src.eval.error_analysis

eval:
	uv run python -m src.eval.run_test

report:
	@echo "Reporte final: reports/REPORT.md"

all: setup eda baseline text-only train train-xgb ablation threshold errors eval

clean:
	rm -rf models/*.pkl reports/figures/*.png reports/optuna_study.json
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} +
