#!/usr/bin/env python
# encoding: utf-8

# ------------------------------------------------------------------------------
#  Name: main.py
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

import fire

from lastro_ai.cli import LastroCLI


def main():
    """Ponto de entrada para a CLI do Lastro.AI.

    Utiliza a biblioteca Fire para orquestrar as chamadas da CLI, expondo
    os métodos da classe LastroCLI como comandos de terminal para gerenciar
    o ciclo de vida dos relatórios do BACEN.
    """

    fire.Fire(LastroCLI)


if __name__ == '__main__':
    main()
