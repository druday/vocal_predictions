PYTHON ?= python3
PHENOTYPE ?= parkinsons
RUN_ID ?= $(shell date +%Y%m%d-%H%M%S)
BASE_CONFIG ?= configs/base.yaml
PHENO_CONFIG ?= configs/phenotypes/$(PHENOTYPE).yaml

.PHONY: install test lint prepare cluster baseline clustered eval report compile_html pipeline download_data download_data_static frontend

PHYSIONET_USER ?=
PHYSIONET_BASE_URL ?= https://physionet.org/files/b2ai-voice/3.0.0/
DATA_DEST ?= raw_data

install:
	$(PYTHON) -m pip install -e .[dev]

lint:
	ruff check src scripts tests

test:
	pytest

prepare:
	$(PYTHON) scripts/01_prepare_dataset.py --config $(BASE_CONFIG) --phenotype-config $(PHENO_CONFIG) --run-id $(RUN_ID)

cluster:
	$(PYTHON) scripts/02_cluster_features.py --config $(BASE_CONFIG) --phenotype-config $(PHENO_CONFIG) --run-id $(RUN_ID)

baseline:
	$(PYTHON) scripts/03_train_baseline.py --config $(BASE_CONFIG) --phenotype-config $(PHENO_CONFIG) --run-id $(RUN_ID)

clustered:
	$(PYTHON) scripts/04_train_cluster_models.py --config $(BASE_CONFIG) --phenotype-config $(PHENO_CONFIG) --run-id $(RUN_ID)

eval:
	$(PYTHON) scripts/05_patient_level_eval.py --config $(BASE_CONFIG) --phenotype-config $(PHENO_CONFIG) --run-id $(RUN_ID)

report:
	$(PYTHON) scripts/06_generate_figures_tables.py --config $(BASE_CONFIG) --phenotype-config $(PHENO_CONFIG) --run-id $(RUN_ID)

compile_html:
	$(PYTHON) scripts/07_compile_html_report.py --config $(BASE_CONFIG) --phenotype-config $(PHENO_CONFIG) --run-id $(RUN_ID)

pipeline: prepare cluster baseline clustered eval report compile_html
	@echo "Pipeline completed for phenotype=$(PHENOTYPE), run_id=$(RUN_ID)"

download_data:
	PHYSIONET_USER=$(PHYSIONET_USER) scripts/download_physionet_b2ai.sh --mode all --base-url $(PHYSIONET_BASE_URL) --dest $(DATA_DEST)

download_data_static:
	PHYSIONET_USER=$(PHYSIONET_USER) scripts/download_physionet_b2ai.sh --mode static --base-url $(PHYSIONET_BASE_URL) --dest $(DATA_DEST)

frontend:
	$(PYTHON) -m streamlit run app/streamlit_app.py
