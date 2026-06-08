.PHONY: install test build-features serve clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

build-features:
	python src/features/engineer.py

serve:
	uvicorn src.serving.api:app --host 127.0.0.1 --port 8001 --reload

clean:
	rm -rf feature_store/ __pycache__
	find . -name "*.pyc" -delete

help:
	@echo "Commands:"
	@echo "  make install         Install dependencies"
	@echo "  make test            Run 14 unit tests"
	@echo "  make build-features  Build feature store from real data"
	@echo "  make serve           Start feature serving API on port 8001"
