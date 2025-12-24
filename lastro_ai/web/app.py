#!/usr/bin/env python
# encoding: utf-8

# ------------------------------------------------------------------------------
#  Name: app.py
#  Version: 0.0.1
#
#  Summary: Interface Web do Lastro.AI utilizando Chainlit.
#
#  Author: Alexsander Lopes Camargos
#  Author-email: alexcamargos@gmail.com
#
#  License: MIT
# ------------------------------------------------------------------------------

import chainlit as cl

from lastro_ai import config as Cfg
from lastro_ai.agent.core import LastroAgent
from lastro_ai.web.messages import WELCOME_MESSAGE


@cl.on_chat_start
async def on_chat_start() -> None:
    """Inicializa a sessão do chat."""

    # Define as configurações disponíveis na interface (Sidebar).
    settings = await cl.ChatSettings(
        [
            cl.input_widget.Select(
                id="provider",
                label="Provedor de IA",
                values=["groq", "ollama"],
                initial_index=0 if Cfg.LLM_PROVIDER == "groq" else 1,
            )
        ]
    ).send()

    # Instancia o agente e armazena na sessão do usuário.
    # Isso garante que cada aba/usuário tenha seu próprio agente isolado.
    agent = LastroAgent(provider=settings["provider"])
    cl.user_session.set("agent", agent)

    await cl.Message(content=WELCOME_MESSAGE).send()


@cl.on_settings_update
async def setup_agent(settings: dict) -> None:
    """Atualiza o agente quando as configurações são alteradas pelo usuário.

    Args:
        settings (dict): Configurações atualizadas do usuário.
    """

    provider = settings["provider"]

    # Recria o agente com o novo provedor selecionado.
    agent = LastroAgent(provider=provider)
    cl.user_session.set("agent", agent)


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Recebe a mensagem do usuário e gera a resposta.

    Args:
        message (cl.Message): Mensagem enviada pelo usuário.
    """

    agent = cl.user_session.get("agent")

    # Executa o agente. Usamos make_async para envolver a chamada síncrona do agente.
    response = await cl.make_async(agent.run)(message.content, verbose=True)

    await cl.Message(content=response).send()
