#!/usr/bin/env python
# encoding: utf-8

# ------------------------------------------------------------------------------
#  Name: model_factory.py
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

from typing import Callable, Dict, Optional

from langchain_core.language_models import BaseChatModel

from lastro_ai.agent.services.groq import create_groq_model
from lastro_ai.agent.services.ollama import create_ollama_model


# Mapeamento de provedores para suas funções de fábrica.
_MODEL_FACTORIES: Dict[str, Callable[[Optional[str], float], BaseChatModel]] = {
    'ollama': create_ollama_model,
    'groq': create_groq_model,
}


def get_chat_model(provider: str = "ollama",
                   model_name: Optional[str] = None,
                   temperature: float = 0) -> BaseChatModel:
    """Instancia o modelo de linguagem (LLM/SLM) configurado para o projeto.

    Permite alternar entre provedores (local/nuvem) e modelos via argumentos.

    Args:
        provider (str): O provedor do modelo ('ollama', 'openai', etc).
        model_name (str, optional): Nome específico do modelo. Se None, usa o config.
        temperature (float): Criatividade do modelo (0 = determinístico).

    Returns:
        BaseChatModel: Instância do modelo configurada para uso no LangChain.
    """

    provider = provider.lower()

    if provider not in _MODEL_FACTORIES:
        raise ValueError(f'Provedor de LLM não suportado: {provider}')

    return _MODEL_FACTORIES[provider](model_name, temperature)
