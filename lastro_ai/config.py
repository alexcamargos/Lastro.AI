#!/usr/bin/env python
# encoding: utf-8

# ------------------------------------------------------------------------------
#  Name: config.py
#  Version: 0.0.1
#
#  Summary: Lastro.AI: Seu Agente de Política Monetária.
#           Transforma relatórios estáticos do BACEN em conhecimento
#           dinâmico e preciso.
#
#  Author: Alexsander Lopes Camargos
#  Author-email: alexcamargos@gmail.com
#
#  License: MIT
# ------------------------------------------------------------------------------

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Base do projeto (cwd)
BASE_DIR: Path = Path(__file__).resolve().parent.parent


# Diretórios de Dados
DATA_DIR: Path = BASE_DIR / 'data'
RAW_DATA_DIR: Path = DATA_DIR / 'raw'
VECTOR_STORE_DIR: Path = DATA_DIR / 'vector_store'

# Endereços dos relatórios do BACEN
BASE_REPORT_URL: str = 'https://www.bcb.gov.br/content/ri/relatorioinflacao/'
# Relatório de Inflação (RI) - formato até 2024.
RI_TEMPLATE: str = '{year}{month}/ri{year}{month}p.pdf'
# Relatório de Política Monetária (RPM) - novo formato após 2024.
RPM_TEMPLATE: str = '{year}{month}/rpm{year}{month}p.pdf'
REPORT_QUARTERS: list[str] = ['03', '06', '09', '12']


# Configurações de IA: Modelo de Embeddings com suporte a Português.
EMBEDDING_MODEL_NAME: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'


# Configurações de Chunking
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 250
BATCH_SIZE: int = 25


# Configurações de Download
DOWNLOAD_TIMEOUT: tuple[int, int] = (15, 60)
DOWNLOAD_CHUNK_SIZE: int = 8192


# Vector Store filenames
VECTOR_STORE_INDEX_NAME: str = 'index.faiss'
VECTOR_STORE_DOCS_NAME: str = 'documents.pkl'


# Diretório dos Prompts (Agora centralizado)
PROMPTS_DIR = BASE_DIR / 'lastro_ai' / 'prompts'
# Perfil de Prompt Ativo: 'slm' (instruções rígidas) ou 'std' (padrão)
PROMPT_PROFILE: str = 'slm'


# Configuração do Provedor de LLM Ativo:
# Exemplos: 'ollama', 'groq, 'openai', etc.
LLM_PROVIDER: str = 'groq'


# Configurações de SLM (Ollama)
OLLAMA_BASE_URL: str = 'http://localhost:11434'
OLLAMA_MODEL_NAME: str = 'deepseek-r1:8b'


# Configurações de Groq
GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME: str = 'qwen/qwen3-32b'


# Caminho para a aplicação Web (Chainlit)
WEB_APP_PATH: Path = BASE_DIR / 'lastro_ai' / 'web' / 'app.py'
