#!/usr/bin/env python
# encoding: utf-8

# ------------------------------------------------------------------------------
#  Name: evaluator.py
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

import random
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from loguru import logger

from lastro_ai import config as Cfg
from lastro_ai.agent.tools.retrieval import _get_vector_store, retrieve_context
from lastro_ai.core.extracting import extract_report_text
from lastro_ai.core.prompts import PromptManager


def _get_random_samples(raw_dir: Path, num_files: int = 3) -> list[Path]:
    """Seleciona aleatoriamente arquivos PDF para avaliação.

    Args:
        raw_dir (Path): Diretório contendo os PDFs brutos.
        num_files (int): Número máximo de arquivos a serem selecionados.
                         Padrão é 3.

    Returns:
        list[Path]: Lista de caminhos completos para os arquivos PDF selecionados.
    """

    all_pdfs = list(raw_dir.glob('*.pdf'))
    if not all_pdfs:
        logger.error('Nenhum PDF encontrado para avaliação.')
        return []

    return random.sample(all_pdfs, min(len(all_pdfs), num_files))


def evaluate_extraction_quality(num_files: int = 3) -> None:
    """Executa a avaliação de qualidade da extração usando LLM-as-a-Judge.

    Seleciona PDFs aleatórios, extrai uma página de cada e submete ao LLM
    para análise de coerência, formatação e limpeza.

    Args:
        num_files (int): Número de arquivos a serem avaliados.
                         Padrão é 3.
    """

    logger.info('--- Iniciando Avaliação de Qualidade de Extração (LLM-as-a-Judge) ---')

    # Configuração do LLM (Groq)
    try:
        llm = ChatGroq(model_name=Cfg.GROQ_MODEL_NAME,
                       temperature=0,
                       api_key=Cfg.GROQ_API_KEY)
    except (ValueError, TypeError) as error:
        logger.error(f'Erro de configuração ao inicializar LLM Groq: {error}')
        return

    judge_template = PromptManager.get('evaluation.extraction.judge_template')
    evaluation_prompt = ChatPromptTemplate.from_template(judge_template)

    chain = evaluation_prompt | llm

    pdf_files = _get_random_samples(Cfg.RAW_DATA_DIR, num_files)

    for pdf_path in pdf_files:
        logger.info(f'\nProcessando arquivo: {pdf_path.name}...')
        try:
            # Extrai todas as páginas (necessário para escolher uma aleatória)
            # Nota: Isso pode ser lento para PDFs gigantes, mas aceitável para teste.
            docs = extract_report_text(pdf_path)
            if not docs:
                logger.warning('Nenhuma página extraída.')
                continue

            # Escolhe uma página aleatória
            random_page_index = random.randint(0, len(docs) - 1)
            target_doc = docs[random_page_index]
            page_num = target_doc.metadata.get("page", random_page_index + 1)

            logger.info(f'Avaliado página {page_num} de {len(docs)}...')

            # Executa a avaliação
            response = chain.invoke({'text': target_doc.page_content})

            # Exibe o resultado (usando print para garantir visualização limpa do JSON no terminal)
            print(f'\nRelatório para {pdf_path.name} (Pág. {page_num}):')
            print(response.content)
            print('-' * 40)

        except (IOError, ValueError) as error:
            logger.error(f'Falha ao ler/processar PDF {pdf_path.name}: {error}')
        except Exception as error:  # pylint: disable=broad-except
            # Captura erros inesperados do LangChain/Groq para não interromper o lote.
            logger.error(f'Erro inesperado na avaliação de {pdf_path.name}: {error}')


def evaluate_retrieval_performance(num_samples: int = 5) -> None:
    """Avalia a performance de recuperação (Retrieval) usando métrica de Hit Rate.

    1. Seleciona N chunks aleatórios do banco vetorial.
    2. Gera uma pergunta sintética para cada chunk usando o LLM.
    3. Realiza a busca vetorial usando essa pergunta.
    4. Verifica se o chunk original está entre os resultados encontrados (Hit Rate).

    Args:
        num_samples (int): Número de amostras a testar.
    """

    logger.info('Iniciando Avaliação de Recuperação (Hit Rate/Synthetic QA)')

    vector_store = _get_vector_store()
    if not vector_store.documents:
        logger.error('Banco vetorial vazio. Execute "ingest" ou "process_batch" primeiro.')
        return

    # Garante que não tentaremos pegar mais amostras do que existem.
    k = min(num_samples, len(vector_store.documents))
    samples = random.sample(vector_store.documents, k)

    logger.info(f'Selecionando {k} chunks aleatórios...')

    # Config LLM para gerar perguntas
    try:
        llm = ChatGroq(model_name=Cfg.GROQ_MODEL_NAME,
                       temperature=0,
                       api_key=Cfg.GROQ_API_KEY)
    except (ValueError, TypeError) as error:
        logger.error(f'Erro ao inicializar LLM: {error}')
        return

    qa_gen_template = PromptManager.get('evaluation.retrieval.synthetic_qa_template')
    qa_gen_prompt = ChatPromptTemplate.from_template(qa_gen_template)

    qa_chain = qa_gen_prompt | llm

    hits = 0

    for index, doc in enumerate(samples, 1):
        original_text = doc.page_content[:200].replace('\n', ' ') + "..." # Snippet for comparison/log

        try:
            # Geração de Pergunta.
            question_resp = qa_chain.invoke({'text': doc.page_content})
            question = question_resp.content.strip()

            # Exibe o resultado (usando print para garantir visualização limpa do JSON no terminal).
            print(f'\n[Teste {index}/{k}]')
            print(f'Chunk Alvo: "{original_text}"')
            print(f'Pergunta Gerada: "{question}"')

            # Busca (Retrieval): Recuperamos top 5 para ser generoso.
            retrieved_text = retrieve_context(question, k=5)

            # Verificar Hit (se o texto original está no contexto retornado).
            # NOTE: Comparação exata pode falhar devido a formatação, mas retrieve_context
            # retorna trechos. # O retrieve_context retorna uma string formatada.
            # Vamos verificar se um trecho significativo do original está lá.
            # Uma heurística melhor seria checar o ID se tivéssemos, mas vamos checar substring.

            # Vamos pegar os primeiros 50 chars do original para buscar.
            search_signature = doc.page_content[:150].strip()

            if search_signature in retrieved_text:
                print('Resultado: HIT (Encontrado)')
                hits += 1
            else:
                print('Resultado: MISS (Não encontrado)')
                # logger.warning(f"Esperado: {search_signature[:50]}...")

        except Exception as e:
            logger.error(f'Erro no teste {index}: {e}')

    hit_rate = hits / k
    logger.info('Resultado Final')
    logger.info(f'Hit Rate: {hit_rate:.2%} ({hits}/{k})')
