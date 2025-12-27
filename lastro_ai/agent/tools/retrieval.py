#!/usr/bin/env python
# encoding: utf-8

# ------------------------------------------------------------------------------
#  Name: retrieval.py
#  Version: 0.1.0
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

from sentence_transformers import CrossEncoder

from lastro_ai import config as Cfg
from lastro_ai.core.embedding import VectorStore


@lru_cache(maxsize=1)
def _get_vector_store() -> VectorStore:
    """Recupera ou inicializa a instância do armazenamento vetorial (Lazy Loading).

    Evita recarregar o índice FAISS a cada chamada da ferramenta.

    Returns:
        VectorStore: Instância carregada do gerenciador de vetores.
    """

    vector_store = VectorStore(Cfg.EMBEDDING_MODEL_NAME)
    vector_store.load_vector_store(Cfg.VECTOR_STORE_DIR)

    return vector_store


@lru_cache(maxsize=1)
def _get_reranker() -> CrossEncoder:
    """Recupera ou inicializa o modelo de Cross-Encoder para Re-Ranking (Lazy Loading).

    Returns:
        CrossEncoder: Instância do modelo carregado.
    """

    return CrossEncoder(Cfg.RERANKER_MODEL_NAME)


def retrieve_context(query: str, k: int = 5) -> str:
    """Realiza uma busca semântica com re-ranking (RAG) e retorna o contexto formatado.

    Processo:
        1. Recuperação (Retrieval): Busca K candidatos iniciais (broad recall) no banco vetorial.
        2. Re-ordenação (Re-Ranking): Usa um Cross-Encoder para pontuar a relevância de cada par (Query, Doc).
        3. Seleção: Retorna os top K documentos com maior pontuação.

    Args:
        query (str): A pergunta ou termo de pesquisa.
        k (int, optional): Número de trechos finais a recuperar. Padrão é 5.

    Returns:
        str: Uma string contendo os trechos encontrados formatados com metadados.
    """

    # Recall: Busca um número maior de documentos (padrão 20)
    # para garantir que a resposta esteja no conjunto.
    initial_k = Cfg.INITIAL_RETRIEVAL_K
    vector_store = _get_vector_store()

    # Executa a busca vetorial (FAISS)
    initial_results = vector_store.search(query, k=initial_k)

    if not initial_results:
        return 'Nenhuma informação relevante encontrada nos documentos.'

    # Re-Ranking: Usa o Cross-Encoder para pontuar os documentos.
    reranker = _get_reranker()

    # Prepara os pares (Query, Doc) aplicáveis ao modelo.
    cross_encoder_inputs = [
        [query, doc.page_content] for doc in initial_results
    ]

    # Atribui scores de similaridade/relevância.
    scores = reranker.predict(cross_encoder_inputs)

    # Associa os scores aos documentos originais.
    scored_results = []
    for i, doc in enumerate(initial_results):
        scored_results.append(
            {
                'doc': doc,
                'score': scores[i]
            }
        )

    # Ordena os resultados pelo score em ordem decrescente.
    # documentos mais relevantes primeiro.
    scored_results.sort(key=lambda doc: doc['score'], reverse=True)

    # Seleciona os top K solicitados pelo agente (padrão 5).
    final_results = scored_results[:k]

    # Formatação da saída para o LLM
    formatted_output = []
    for index, item in enumerate(final_results, 1):
        doc = item['doc']
        score = item['score']

        # Extrai metadados para citar fontes.
        source = doc.metadata.get('source', 'Desconhecido')
        page = doc.metadata.get('page', '?')
        content = doc.page_content

        # Formata o trecho com metadados e score (debug only).
        formatted_output.append(
            f'--- Trecho {index} (Relevância: {score:.4f}) ---\n'
            f'Fonte: {source} (Pág. {page})\n'
            f'Conteúdo: {content}'
        )

    return '\n\n'.join(formatted_output)
