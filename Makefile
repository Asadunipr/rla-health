.PHONY: setup lint test data_mitbih data_wesad train_mitbih train_wesad ablate compress latency fade

setup:
	@echo "Installing pre-commit hooks…"
	pre-commit install

lint:
	ruff check --fix .
	isort .
	black .

test:
	pytest -q --maxfail=1 --disable-warnings

data_mitbih:
	python scripts/prep_mitbih.py --out data/mitbih

data_wesad:
	python scripts/prep_wesad.py --out data/wesad

train_mitbih:
	python scripts/train.py --config configs/mitbih.yml

train_wesad:
	python scripts/train.py --config configs/wesad.yml

ablate:
	python scripts/ablate.py --config configs/ablation_default.yml

compress:
	python scripts/compress.py --config configs/mitbih.yml

latency:
	python scripts/measure_latency.py --config configs/mitbih.yml

fade:
	python scripts/run_fade_baseline.py --config configs/mitbih.yml
