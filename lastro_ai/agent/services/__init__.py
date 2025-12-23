#!/usr/bin/env python
# encoding: utf-8

# ------------------------------------------------------------------------------
#  Name: __init__.py
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

from .model_factory import get_chat_model

__all__ = ['get_chat_model']
