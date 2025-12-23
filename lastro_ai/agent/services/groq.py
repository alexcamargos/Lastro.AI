#!/usr/bin/env python
# encoding: utf-8

# ------------------------------------------------------------------------------
#  Name: groq.py
#  Version: 0.0.1
#
#  Summary: Implementação do provedor Groq para o Lastro.AI.
#
#  Author: Alexsander Lopes Camargos
#  Author-email: alexcamargos@gmail.com
#
#  License: MIT
# ------------------------------------------------------------------------------

from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq

from lastro_ai import config as Cfg


def create_groq_model(model_name: Optional[str], temperature: float = 0) -> BaseChatModel:
    """Cria e configura uma instância do modelo ChatGroq.

    Args:
        model_name (Optional[str]): Nome específico do modelo.
                                    Caso seja None, utiliza o modelo padrão definido em config.py.
        temperature (float): Define a criatividade do modelo.
                             Padrão é 0 (determinístico).

    Returns:
        BaseChatModel: Instância do ChatGroq configurada.
    """

    # Usa o modelo padrão se nenhum nome for fornecido.
    if model_name is None:
        model_name = Cfg.GROQ_MODEL_NAME

    return ChatGroq(model=model_name,
                    temperature=temperature,
                    api_key=Cfg.GROQ_API_KEY)
