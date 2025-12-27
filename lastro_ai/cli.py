#!/usr/bin/env python
# encoding: utf-8

# ------------------------------------------------------------------------------
#  Name: cli.py
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

import subprocess
import sys
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter as RCTextSplitter
from loguru import logger

from lastro_ai import config as Cfg
from lastro_ai.agent.core import LastroAgent
from lastro_ai.core.embedding import VectorStore
from lastro_ai.core.evaluator import evaluate_extraction_quality, evaluate_retrieval_performance
from lastro_ai.core.extracting import extract_report_text as extractor
from lastro_ai.core.ingestion import download_report as downloader
from lastro_ai.core.ingestion import download_reports_batch as batch_downloader


class LastroCLI:
    """
    Interface de Linha de Comando (CLI) para o Lastro.AI.

    Esta ferramenta permite gerenciar o ciclo de vida de conhecimento dos relatórios
    do BACEN, incluindo download, extração de texto, vetorização e busca semântica.

    Attributes:
        vector_store (VectorStore): Instância do armazenamento vetorial FAISS.

    Methods:
        _get_vector_store(): Carrega ou inicializa o VectorStore.
        _get_pdf_path(pdf_url: str): Retorna o caminho completo do arquivo PDF local.
        _process_pdf(pdf_path: Path, chunk_size: int): Processa um arquivo PDF (Extração -> Chunking -> Vetorização).
        ingest(pdf_url: str): Baixa um Relatório de Política Monetária do BACEN para a pasta local.
        extract(pdf_path: str): Extrai e exibe o texto bruto de um relatório PDF local.
        vectorize(documents: list): Gera embeddings para uma lista de documentos e atualiza o índice FAISS.
        process_report(pdf_url: str, chunk_size: int = Cfg.CHUNK_SIZE): Executa o fluxo completo de ingestão para um relatório (Download -> Vetorização).
        process_batch(years: int = 1): Baixa e processa automaticamente relatórios dos últimos N anos.
        rebuild_database(): Apaga o índice atual e reconstrói tudo a partir dos PDFs na pasta 'raw'.
        search(question: str, k: int = 5): Realiza uma busca semântica no banco de conhecimento.
        ask(question: str): Envia uma pergunta ao Agente Lastro.AI e exibe a resposta fundamentada.

    Command Examples:
        python main.py ingest <pdf_url>
        python main.py extract <pdf_path>
        python main.py vectorize <documents>
        python main.py process_report <pdf_url> [chunk_size]
        python main.py process_batch [years]
        python main.py rebuild_database
        python main.py inspect [limit]
        python main.py search <question> [k]
        python main.py ask <question> [--verbose]
        python main.py evaluate [num_files]
        python main.py test_retrieval [num_samples]
        python main.py web [--watch]
    """

    def __init__(self) -> None:
        """Inicializa a instância da CLI do Lastro.AI.

        Define o estado inicial da aplicação, preparando os atributos necessários
        para a execução dos comandos. O armazenamento vetorial (vector_store) é
        iniciado como None para permitir o carregamento sob demanda (lazy loading).
        """

        self.vector_store = None

    def _get_vector_store(self) -> VectorStore:
        """Recupera ou inicializa a instância do armazenamento vetorial.

        Implementa um padrão de carregamento preguiçoso (lazy loading) para o
        VectorStore. Se a instância ainda não existir, ela é criada utilizando
        o modelo de embeddings configurado e os dados persistidos são carregados
        do diretório de armazenamento.

        Returns:
            VectorStore: Instância ativa e carregada do gerenciador de vetores.
        """

        if self.vector_store is None:
            self.vector_store = VectorStore(Cfg.EMBEDDING_MODEL_NAME)
            self.vector_store.load_vector_store(Cfg.VECTOR_STORE_DIR)

        return self.vector_store

    def _get_pdf_path(self, pdf_url: str) -> Path:
        """Determina o caminho do arquivo local com base na URL do relatório.

        Extrai o nome do arquivo da URL fornecida e o combina com o diretório
        configurado para dados brutos (RAW_DATA_DIR), retornando o caminho
        completo onde o PDF deve ser salvo ou encontrado.

        Args:
            pdf_url (str): URL do relatório PDF.

        Returns:
            Path: Caminho completo do arquivo PDF no sistema de arquivos local.
        """

        filename = pdf_url.split('/')[-1]

        return Cfg.RAW_DATA_DIR / filename

    def _process_pdf(self, pdf_path: Path, chunk_size: int) -> None:
        """Orquestra o processamento completo de um arquivo PDF local.

        Este método interno encapsula a lógica de transformação de um documento PDF
        em vetores pesquisáveis. As etapas incluem:
        1. Extração do texto bruto e metadados das páginas.
        2. Segmentação (chunking) do texto em partes menores e sobrepostas.
        3. Envio dos segmentos para vetorização e indexação no FAISS.

        Args:
            pdf_path (Path): Caminho absoluto ou relativo para o arquivo PDF.
            chunk_size (int): Número máximo de caracteres por segmento de texto.
        """

        # Extração de texto do PDF do relatório.
        raw_documents = extractor(pdf_path)

        # Sanitização: Remove caracteres de substituição () gerados por falhas de encoding no PDF.
        for doc in raw_documents:
            if doc.page_content:
                doc.page_content = doc.page_content.replace('\ufffd', '')

        # Chunking do texto extraído, para facilitar a vetorização.
        logger.info('Segmentando texto (Chunking)...')
        text_splitter = RCTextSplitter(chunk_size=chunk_size,
                                       chunk_overlap=Cfg.CHUNK_OVERLAP,
                                       length_function=len,
                                       separators=['\n\n', '\n', ' ', ''])

        # Usa split_documents para preservar metadados (página, fonte) nos chunks.
        chunked_docs = text_splitter.split_documents(raw_documents)

        # Construção do índice FAISS com base nos chunks.
        self.vectorize(chunked_docs)

    def ingest(self, pdf_url: str) -> None:
        """Realiza o download de um Relatório de Política Monetária do BACEN.

        Baixa o arquivo PDF da URL especificada e o salva no diretório de dados
        brutos ('raw') para processamento posterior. Este comando apenas adquire
        o arquivo, sem realizar extração ou vetorização imediata.

        Args:
            pdf_url (str): URL do relatório PDF a ser baixado.
        """

        save_path = self._get_pdf_path(pdf_url)

        # Download do Relatório de Política Monetária do BACEN.
        downloader(pdf_url, save_path)

    def extract(self, pdf_path: str) -> list[str]:
        """Extrai e retornar o conteúdo textual bruto de um relatório PDF local.

        Processa o arquivo PDF especificado, extraindo o texto de cada página
        sem realizar chunking ou vetorização.

        Args:
            pdf_path (str): Caminho absoluto ou relativo para o arquivo PDF.

        Returns:
            list[str]: Lista contendo o texto extraído de cada página.
        """

        # Extração de texto do PDF.
        documents = extractor(Path(pdf_path))

        # Retorna apenas o conteúdo textual para manter compatibilidade visual na CLI
        return [document.page_content for document in documents]

    def vectorize(self, documents: list) -> None:
        """Gera embeddings para uma lista de documentos e atualiza o índice FAISS.

        Recebe uma lista de documentos (texto ou objetos Document), gera os vetores
        correspondentes utilizando o modelo de embeddings configurado e persiste
        o índice atualizado no disco para uso posterior.

        Args:
            documents (list): Lista de objetos Document ou strings para indexar.
        """

        logger.info('Iniciando vetorização dos documentos...')

        vector_store = self._get_vector_store()
        vector_store.add_documents(documents)
        vector_store.save_vector_store(Cfg.VECTOR_STORE_DIR)

        logger.info('Vetorização concluída com sucesso!')

    def process_report(self, pdf_url: str, chunk_size: int = Cfg.CHUNK_SIZE) -> None:
        """Executa o fluxo completo de ingestão de um relatório individual.

        Verifica se o relatório já existe localmente; caso contrário, realiza o download.
        Em seguida, processa o arquivo PDF extraindo o texto, segmentando-o em chunks
        e gerando os embeddings para indexação no banco vetorial.

        Args:
            pdf_url (str): URL do relatório PDF a ser processado.
            chunk_size (int): Tamanho dos chunks para segmentação de texto.
                              Padrão definido em config.CHUNK_SIZE.
        """

        # Download do Relatório de Política Monetária do BACEN.
        pdf_path = self._get_pdf_path(pdf_url)

        # Checa se o arquivo já foi baixado.
        if not pdf_path.exists():
            logger.info(f'Baixando {pdf_path.name}...')
            downloader(pdf_url, pdf_path)

        self._process_pdf(pdf_path, chunk_size)

    def process_batch(self, years: int = 1) -> None:
        """Executa o processamento em lote de relatórios dos últimos N anos.

        Automatiza a busca, download e ingestão de múltiplos relatórios do BACEN
        com base no período especificado. Para cada relatório encontrado, realiza
        o fluxo completo de processamento (extração, chunking e vetorização).

        Args:
            years (int): Número de anos para retroceder na busca de relatórios.
                         Padrão é 1 ano.
        """

        logger.info(f'Iniciando processamento em lote dos últimos {years} anos...')
        reports = batch_downloader(years, Cfg.RAW_DATA_DIR)

        for report in reports:
            self._process_pdf(report, Cfg.CHUNK_SIZE)

    def rebuild_database(self) -> None:
        """Reconstrói integralmente o banco de dados vetorial.

        Remove o índice e os documentos existentes e reprocessa todos os arquivos PDF
        disponíveis no diretório de dados brutos ('raw'). Útil para aplicar alterações
        nas configurações de chunking, modelo de embeddings ou após limpeza de dados.
        """

        logger.warning('Iniciando reconstrução do banco de dados vetorial...')

        # Remoção dos arquivos físicos do índice existente.
        index_path = Cfg.VECTOR_STORE_DIR / Cfg.VECTOR_STORE_INDEX_NAME
        docs_path = Cfg.VECTOR_STORE_DIR / Cfg.VECTOR_STORE_DOCS_NAME

        if index_path.exists():
            index_path.unlink()

        if docs_path.exists():
            docs_path.unlink()

        # Reinicializa o VectorStore vazio.
        self.vector_store = None

        # Reprocessa todos os PDFs na pasta de dados brutos.
        pdf_files = list(Cfg.RAW_DATA_DIR.glob('*.pdf'))

        if not pdf_files:
            logger.warning('Nenhum PDF encontrado para processar.')
            return

        logger.info(f'Encontrados {len(pdf_files)} arquivos para reprocessar.')

        for pdf_path in pdf_files:
            try:
                self._process_pdf(pdf_path, Cfg.CHUNK_SIZE)
            except Exception:  # pylint: disable=broad-except
                logger.exception(f'Erro ao processar {pdf_path.name}')

        logger.info('Reconstrução concluída com sucesso!')

    def inspect(self, limit: int = 3) -> None:
        """Exibe uma amostra dos documentos vetorizados para auditoria.

        Permite visualizar como o texto foi segmentado (chunking) e armazenado,
        incluindo metadados e formatação de tabelas. Útil para verificar se
        o chunk_size e chunk_overlap estão adequados.

        Args:
            limit (int): Número de chunks a serem exibidos. Padrão é 3.
        """
        vector_store = self._get_vector_store()
        docs = vector_store.documents

        if not docs:
            logger.warning("O banco vetorial está vazio. Execute 'ingest' ou 'process_batch' primeiro.")
            return

        print(f'\n=== Inspeção do Vector Store ({len(docs)} chunks totais) ===')

        for i, doc in enumerate(docs[:limit]):
            print(f'\n[Chunk #{i}] Metadata: {doc.metadata}')
            print('-' * 60)
            print(doc.page_content)
            print('-' * 60)

    def search(self, question: str, k: int = 5) -> None:
        """Realiza uma busca semântica no banco de conhecimento vetorial.

        Utiliza o modelo de embeddings para converter a pergunta em um vetor e
        encontra os trechos de texto mais similares armazenados no índice FAISS.
        Exibe os resultados encontrados juntamente com seus metadados.

        Args:
            question (str): Pergunta ou consulta em linguagem natural.
            k (int): Número de trechos mais relevantes a serem retornados.
                     Padrão é 5.
        """

        vector_store = self._get_vector_store()
        results = vector_store.search(question, k=k)

        print(f"Encontrados {len(results)} resultados para: '{question}'")
        for index, text in enumerate(results, 1):
            print('\n--- Resultado(s) ---')
            if hasattr(text, 'page_content'):
                print(f'# {index}: {text.page_content}')
                print(f'Metadata: {text.metadata}')
            else:
                print(f'# {index}: {text}')
            print('-' * 40)

    def ask(self, question: str, verbose: bool = False) -> None:
        """Envia uma pergunta ao Agente Lastro.AI e exibe a resposta.

        Utiliza o modelo de linguagem configurado (SLM/LLM) e a base de conhecimento
        vetorial para gerar uma resposta fundamentada nos relatórios do BACEN.

        Args:
            question (str): A pergunta a ser respondida pelo agente.
            verbose (bool): Se True, exibe detalhes do prompt enviado ao modelo.
        """

        logger.info('Iniciando interação com Lastro.AI para a pergunta:')
        logger.info(question)

        try:
            agent = LastroAgent()
            response = agent.run(question, verbose=verbose)

            print('\n' + '=' * 60)
            print('Lastro.AI Responde:\n')
            print(response)
            print('=' * 60 + '\n')

        # Broad-except é usado intencionalmente para capturar qualquer erro
        # durante a interação do agente, garantindo que o usuário receba
        # uma mensagem de erro amigável em vez de uma falha abrupta.
        except Exception:  # pylint: disable=broad-except
            logger.exception('Ocorreu um erro inesperado ao processar a pergunta.')

    def evaluate(self, num_files: int = 3) -> None:
        """Avalia a qualidade da extração de texto usando LLM-as-a-Judge.

        Seleciona PDFs aleatórios, extrai texto e solicita ao LLM uma avaliação
        de coerência, formatação de tabelas e presença de ruído.

        Args:
            num_files (int): Número de arquivos a testar. Padrão é 3.
        """

        evaluate_extraction_quality(num_files)
    
    def test_retrieval(self, num_samples: int = 5) -> None:
        """Avalia a performance de recuperação (Retrieval) do banco vetorial.

        Calcula a métrica de 'Hit Rate' gerando perguntas sintéticas para chunks
        aleatórios e verificando se o sistema consegue recuperar o chunk original.

        Args:
            num_samples (int): Número de amostras para o teste. Padrão é 5.
        """

        evaluate_retrieval_performance(num_samples)

    def web(self, watch: bool = False) -> None:
        """Inicia a interface web do Lastro.AI.

        Inicia o servidor Chainlit, tornando o agente acessível via navegador.

        Args:
            watch (bool): Se True, ativa o modo de recarregamento automático (desenvolvimento).
        """

        logger.info(f'Iniciando interface web a partir de: {Cfg.WEB_APP_PATH}')

        cmd = [sys.executable, '-m', 'chainlit', 'run', str(Cfg.WEB_APP_PATH)]

        if watch:
            cmd.append("-w")

        try:
            # Executa o módulo chainlit como script (-m chainlit run ...)
            subprocess.run(cmd, check=True)
        except KeyboardInterrupt:
            logger.info('Servidor web interrompido pelo usuário.')
        except subprocess.CalledProcessError as error:
            logger.error(f'O servidor web foi encerrado com erro (código {error.returncode}).')
