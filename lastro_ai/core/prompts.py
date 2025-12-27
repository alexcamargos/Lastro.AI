#!/usr/bin/env python
# encoding: utf-8

# ------------------------------------------------------------------------------
#  Name: ingestion.py
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

import tomllib
from typing import Any

from loguru import logger

from lastro_ai import config as Cfg


# pylint: disable=too-few-public-methods
class PromptManager:
    """Gerenciador centralizado de prompts do sistema.

    Implementa Lazy Loading: carrega arquivos TOML sob demanda (apenas quando solicitado),
    evitando desperdício de memória e I/O na inicialização.
    """

    # Cache interno para armazenar prompts carregados.
    _cache: dict[str, Any] = {}

    @classmethod
    def _load_file(cls, namespace: str) -> None:
        """Carrega um arquivo TOML específico para o cache, se existir.

        Args:
            namespace (str): Nome do arquivo TOML (sem extensão).
        """

        if namespace in cls._cache:
            return

        toml_path = Cfg.PROMPTS_DIR / f'{namespace}.toml'

        # Se o arquivo de prompt não existir, faz cache como vazio, evitando leituras repetidas.
        if not toml_path.exists():
            logger.error(f'Arquivo de prompt não encontrado: {toml_path}')
            cls._cache[namespace] = {}
            return

        try:
            with open(toml_path, 'rb') as file:
                cls._cache[namespace] = tomllib.load(file)
                logger.debug(f'Prompt carregado: {namespace}.toml')
        except tomllib.TOMLDecodeError as error:
            logger.error(f'Erro de sintaxe no arquivo TOML {namespace}.toml: {error}')
            cls._cache[namespace] = {}
        except OSError as error:
            logger.error(f'Erro de E/S ao ler {namespace}.toml: {error}')
            cls._cache[namespace] = {}

    @classmethod
    def get(cls, path: str) -> str:
        """Recupera um prompt usando notação de ponto (arquivo.secao.chave).

        Carrega o arquivo TOML correspondente sob demanda (lazy loading).

        Args:
            path (str): Caminho do prompt no formato 'arquivo.chave'.
                        Exemplo: 'agent.slm.system_message'
                        (Lê 'agent.toml' -> seção [slm] -> chave 'system_message')

        Returns:
            str: O texto do prompt ou string vazia se falhar.
        """

        keys = path.split('.')
        if len(keys) < 2:
            logger.error(f"Caminho de prompt inválido (requer 'arquivo.chave'): {path}")
            return ''

        # O primeiro elemento é o nome do arquivo TOML.
        namespace = keys[0]

        # Lazy Loading: Carrega apenas o arquivo necessário
        if namespace not in cls._cache:
            cls._load_file(namespace)

        value = cls._cache[namespace]

        try:
            # Navega pelas chaves restantes (ex: slm -> system_message)
            for key in keys[1:]:
                value = value[key]

            if not isinstance(value, str):
                logger.warning(f'O caminho do prompt "{path}" não aponta para uma string final.')
                return str(value)

            return value.strip()

        except (KeyError, TypeError):
            logger.error(f'Chave não encontrada no prompt: {path}')
            return ''
