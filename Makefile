.PHONY: help setup run test lint ruff mypy imports-check clean pre_commit_setup pre_commit

help:
	@echo "Comandos disponíveis:"
	@echo "  make setup            - Instala as dependências com uv"
	@echo "  make run              - Executa a aplicação (main.py)"
	@echo "  make test             - Executa os testes com pytest"
	@echo "  make lint             - Verifica o estilo do código"
	@echo "  make pre_commit_setup - Instala os hooks do pre-commit"
	@echo "  make pre_commit       - Executa o pre-commit manualmente"
	@echo "  make clean            - Remove arquivos temporários"

setup:
	uv venv
	uv pip install --upgrade uv
	uv pip install .

pre_commit_setup:
	@echo "Instalando hooks do pre-commit..."
	uv run pre-commit install

pre_commit:
	@echo "Executando pre-commit manualmente..."
	uv run pre-commit run --all-files

run:
	@echo "Executando a aplicação..."
	PYTHONPATH=src uv run python -m ergoview.main --source 0

test:
	@echo "Executando testes com pytest..."
	PYTHONPATH=$(pwd)/src uv run pytest tests --maxfail=3 --disable-warnings -q

ruff:
	@echo "Verificando o código com Ruff..."
	uv run ruff format --check --quiet
	uv run ruff check --quiet

mypy:
	@echo "Executando análise de tipos com mypy..."
	uv run mypy .

imports-check:
	@echo "Verificando imports com isort..."
	uv run isort . --check-only

lint:
	$(MAKE) ruff
	$(MAKE) mypy
	$(MAKE) imports-check

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -delete
