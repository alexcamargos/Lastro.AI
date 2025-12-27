#!/usr/bin/env python
# encoding: utf-8

# ------------------------------------------------------------------------------
#  Name: core.py
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

from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from loguru import logger

from lastro_ai import config as Cfg
from lastro_ai.agent.services import get_chat_model
from lastro_ai.agent.tools.retrieval import retrieve_context
from lastro_ai.core.prompts import PromptManager


# pylint: disable=too-few-public-methods
class LastroAgent:
    """
    Agente principal do Lastro.AI.

    Responsável por orquestrar o fluxo de:
    1. Carregamento de prompts (TOML) via PromptManager.
    2. Recuperação de contexto via Vector Store (RAG).
    3. Geração de resposta via LLM.
    """

    def __init__(self, provider: Optional[str] = None) -> None:
        # Inicializa o modelo de linguagem.
        # O provedor é definido na configuração global, mas pode ser sobrescrito.
        if provider is None:
            provider = Cfg.LLM_PROVIDER

        self.chat_model = get_chat_model(provider=provider)

    def _inspect_prompt(self, prompt_value: PromptValue, verbose: bool) -> PromptValue:
        """Helper para exibir o prompt no terminal se o modo verbose estiver ativo.

        Args:
            prompt_value: O prompt gerado.
            verbose (bool): Se True, exibe o prompt no terminal.
        Returns:
            O prompt original.
        """

        if verbose:
            logger.warning('Modo VERBOSE ativado... exibindo o prompt enviado ao modelo:')
            logger.info(
                f"\n{'=' * 40}\n"
                f"{'-' * 40}\n"
                f"{prompt_value.to_string()}\n"
                f"{'=' * 40}"
            )

        return prompt_value

    def run(self, question: str, verbose: bool = False) -> str:
        """Processa uma pergunta do usuário e gera uma resposta fundamentada.

        Args:
            question (str): A pergunta feita pelo usuário.
            verbose (bool): Se True, exibe o prompt gerado no terminal.

        Returns:
            str: A resposta gerada pela IA.
        """

        # Carrega prompts específicos do perfil configurado.
        promtp_profile = Cfg.PROMPT_PROFILE
        system_msg = PromptManager.get(f'agent.{promtp_profile}.system_message')
        rag_template = PromptManager.get(f'agent.{promtp_profile}.qa_template')

        # Fallback para 'slm' se o perfil configurado não retornar nada.
        if not system_msg:
            logger.warning(f"Profile '{promtp_profile}' not found. Falling back to 'slm'.")
            system_msg = PromptManager.get('agent.slm.system_message')
            rag_template = PromptManager.get('agent.slm.qa_template')

        prompt = ChatPromptTemplate.from_messages(
            [
                ('system', system_msg),
                ('human', rag_template),
            ]
        )

        # Definição da Chain usando LCEL com Runnables
        # RunnablePassthrough: Passa a pergunta original adiante.
        # RunnableLambda: Envolve nossa função de retrieval para ser usada na chain.
        chain = (
            {
                'context': RunnableLambda(retrieve_context),
                'question': RunnablePassthrough()
            }
            | prompt
            | RunnableLambda(lambda prompt: self._inspect_prompt(prompt, verbose))
            | self.chat_model
            | StrOutputParser()
        )

        return chain.invoke(question)
