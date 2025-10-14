#!/bin/bash

set -euo pipefail

ENV_DIR="env_ergoview"
ACTIVATE_UNIX="$ENV_DIR/bin/activate"
ACTIVATE_WIN="$ENV_DIR/Scripts/activate"

echo "🚀 Iniciando configuração do ambiente virtual..."

if [ ! -d "$ENV_DIR" ]; then
    echo "🛠️ Criando ambiente virtual com uv..."
    uv venv "$ENV_DIR"
else
    echo "ℹ️ Ambiente virtual '$ENV_DIR' já existe."
fi

if [ ! -d "$ENV_DIR" ]; then
    echo "❌ Erro: o ambiente '$ENV_DIR' não foi criado corretamente."
    exit 1
fi

# Ativa o ambiente virtual
if [ -f "$ACTIVATE_UNIX" ]; then
    echo "🔗 Ativando ambiente-Linux..."
    source "$ACTIVATE_UNIX"
elif [ -f "$ACTIVATE_WIN" ]; then
    echo "🔗 Ativando ambiente-Windows..."
    source "$ACTIVATE_WIN"
else
    echo "❌ Erro: Script de ativação não encontrado em '$ENV_DIR'."
    exit 1
fi

# Instala as dependências
echo "📦 Instalando dependências..."
poetry install

echo ""
echo "✅ Ambiente '$ENV_DIR' configurado com sucesso!"
echo ""
echo "▶️ Para rodar o projeto, use:"
echo "poetry run python src/ergoview/main.py --source 0"
