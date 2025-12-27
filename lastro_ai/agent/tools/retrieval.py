#!/usr/bin/env python
# encoding: utf-8

# ------------------------------------------------------------------------------
#  Name: retrieval.py
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

from functools import lru_cache

from lastro_ai import config as Cfg
from lastro_ai.core.embedding import VectorStore


@lru_cache(maxsize=1)
def _get_vector_store() -> VectorStore:
    """Recupera ou inicializa a instância do armazenamento vetorial (Lazy Loading).

    Evita recarregar o índice FAISS a cada chamada da ferramenta.
    """

    vector_store = VectorStore(Cfg.EMBEDDING_MODEL_NAME)
    vector_store.load_vector_store(Cfg.VECTOR_STORE_DIR)

    return vector_store


def retrieve_context(query: str, k: int = 5) -> str:
    """Realiza uma busca semântica no banco vetorial e retorna o contexto formatado.

    Esta função é utilizada pelo Agente para consultar a base de conhecimento
    (RAG) antes de responder ao usuário.

    Args:
        query (str): A pergunta ou termo de pesquisa.
        k (int): Número de trechos de documentos a recuperar.

    Returns:
        str: Uma string contendo os trechos encontrados formatados com metadados.
    """

    vector_store = _get_vector_store()
    results = vector_store.search(query, k=k)

    formatted_results = []
    for index, doc in enumerate(results, 1):
        # Extrai metadados para citar fontes (se disponível)
        source = doc.metadata.get('source', 'Desconhecido')
        page = doc.metadata.get('page', '?')
        content = doc.page_content

        formatted_results.append(
            f'--- Trecho {index} ---\nFonte: {source} (Pág. {page})\nConteúdo: {content}'
        )

    if not formatted_results:
        return "Nenhuma informação relevante encontrada nos documentos."

    return "\n\n".join(formatted_results)
