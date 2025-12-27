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
from lastro_ai.core.extracting import extract_report_text


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

    evaluation_prompt = ChatPromptTemplate.from_template("""
        Você é um avaliador especialista em processamento de documentos (OCR/PDF Parsing).
        Analise o texto extraído abaixo de uma página de um documento oficial e dê uma nota de 0 a 10.

        Critérios de Avaliação:
        1. **Coerência Textual**: O texto faz sentido ou há quebras de linha/palavras unidas incorretamente?
        2. **Formatação de Tabelas**: Se houver tabelas, elas estão em Markdown estruturado ou "achatadas"?
        3. **Limpeza**: O texto contém cabeçalhos/rodapés repetitivos ("Banco Central do Brasil", "Pág 10")? (Menos é melhor)

        <texto_extraido>
        {text}
        </texto_extraido>

        Responda APENAS no seguinte formato JSON:
        {{
            "score": <nota_0_10>,
            "reasoning": "<breve explicação dos pontos positivos e negativos>",
            "has_table": <true/false>,
            "has_noise": <true/false>
        }}
    """
    )

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
