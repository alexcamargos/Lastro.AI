#!/usr/bin/env python
# encoding: utf-8

# ------------------------------------------------------------------------------
#  Name: embedding.py
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

import pickle
from pathlib import Path

import faiss
from loguru import logger
from sentence_transformers import SentenceTransformer as st
from langchain_core.documents import Document

from lastro_ai import config as Cfg


class VectorStore:
    """Gerencia o armazenamento vetorial (FAISS) e recuperação de documentos.

    Attributes:
        model (SentenceTransformer): Modelo de embeddings.
        index (faiss.Index): Índice FAISS para busca vetorial.
        documents (list): Lista de documentos associados aos vetores.

    Methods:
        save_vector_store(directory): Salva o índice e os documentos.
        load_vector_store(directory): Carrega o índice e os documentos.
        add_documents(documents): Gera embeddings e adiciona ao índice FAISS.
        search(query, k): Busca os k documentos mais similares.
    """

    def __init__(self, model_name: str) -> None:
        """Inicializa o VectorStore com o modelo de embeddings.

        Args:
            model_name (str): Nome do modelo de embeddings.
        """

        self.model = st(model_name)
        logger.info(f'Modelo carregado no dispositivo: {self.model.device}')
        logger.info(f'Modelo utilizado: {model_name}')

        self.index = None
        self.documents = []

    def save_vector_store(self, directory: Path) -> None:
        """Persiste o estado atual do armazenamento vetorial em disco.

        Salva o índice FAISS (estrutura de busca vetorial) e a lista de documentos
        associados (metadados e conteúdo) no diretório especificado. Isso permite
        que o conhecimento acumulado seja recarregado posteriormente sem necessidade
        de reprocessamento.

        Args:
            directory (Path): Caminho do diretório onde os arquivos de índice e
                              documentos serão salvos.
        """

        # Garante que o diretório de destino exista.
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(directory / Cfg.VECTOR_STORE_INDEX_NAME))

        with open(directory / Cfg.VECTOR_STORE_DOCS_NAME, 'wb') as file:
            pickle.dump(self.documents, file)

        logger.info(f'VectorStore salvo em {directory}')

    def load_vector_store(self, directory: Path) -> None:
        """Restaura o estado do armazenamento vetorial a partir do disco.

        Tenta carregar o índice FAISS e a lista de documentos associados do
        diretório especificado. Se os arquivos necessários existirem, o estado
        interno (índice e documentos) é atualizado; caso contrário, emite um
        aviso e mantém o estado atual.

        Args:
            directory (Path): Caminho do diretório contendo os arquivos de
                              persistência do VectorStore.
        """

        index_path = directory / Cfg.VECTOR_STORE_INDEX_NAME
        docs_path = directory / Cfg.VECTOR_STORE_DOCS_NAME

        if index_path.exists() and docs_path.exists():
            self.index = faiss.read_index(str(index_path))

            with open(docs_path, 'rb') as file:
                self.documents = pickle.load(file)

            logger.info(f'VectorStore carregado de {directory}')
        else:
            logger.warning('Nenhum índice encontrado para carregar.')

    def add_documents(self, documents: list[Document] | list[str]) -> None:
        """Processa e indexa uma lista de documentos no armazenamento vetorial.

        Recebe documentos brutos (texto ou objetos Document), gera seus embeddings
        utilizando o modelo carregado e os adiciona ao índice FAISS. Se o índice
        ainda não existir, ele será inicializado com a dimensão correta.

        Args:
            documents (list[Document] | list[str]): Lista de documentos ou strings
                                                    para serem vetorizados e armazenados.
        """

        if not documents:
            return

        # Garante que documents seja uma lista.
        if not isinstance(documents, list):
            documents = [documents]

        # Extrai o texto para vetorização (suporta str ou Document).
        texts = [
            doc.page_content if hasattr(doc, 'page_content') else str(doc)
            for doc in documents
        ]

        logger.info(f'Gerando embeddings para {len(documents)} documentos...')
        embeddings = self.model.encode(texts, normalize_embeddings=True)

        # Inicializa o índice se necessário.
        dimension = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dimension)

        self.index.add(embeddings)
        self.documents.extend(documents)

        logger.info(f'Total de documentos no índice: {self.index.ntotal}')

    def search(self, query: str, k: int = 3) -> list:
        """Realiza uma busca semântica no índice vetorial.

        Converte a consulta em um vetor de embeddings e utiliza a similaridade
        de cosseno (via vetores normalizados e distância L2) para recuperar
        os documentos mais relevantes do índice.

        Args:
            query (str): Texto da consulta ou pergunta.
            k (int): Número de documentos a serem retornados.
                     Padrão é 3.

        Returns:
            list: Lista dos documentos mais similares encontrados.
        """

        if self.index is None or self.index.ntotal == 0:
            return []

        # Normaliza o vetor da query para corresponder aos vetores
        # normalizados do índice (Simulando Cosine Similarity).
        query_vector = self.model.encode([query], normalize_embeddings=True)
        distances, indices = self.index.search(query_vector, k)

        results = []
        for index in indices[0]:
            if index != -1 and index < len(self.documents):
                results.append(self.documents[index])

        return results
