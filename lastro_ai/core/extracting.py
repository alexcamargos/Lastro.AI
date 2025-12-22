#!/usr/bin/env python
# encoding: utf-8

# ------------------------------------------------------------------------------
#  Name: extracting.py
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

import logging
from pathlib import Path

import pdfplumber
from langchain_core.documents import Document
from loguru import logger


def extract_report_text(pdf_path: Path) -> list[Document]:
    """Extrai o conteúdo textual e tabular de um arquivo PDF.

    Processa o documento página por página, extraindo o texto principal e
    convertendo tabelas encontradas para o formato Markdown. O conteúdo é
    encapsulado em objetos Document do LangChain, preservando metadados
    como a fonte do arquivo e o número da página.

    Nota:
        Silencia logs de erro do pdfminer para evitar poluição visual no console.

    Args:
        pdf_path (Path): Caminho completo para o arquivo PDF a ser processado.

    Returns:
        list[Document]: Lista de documentos contendo o texto e tabelas extraídos.
    """

    logger.info(f'Extraindo texto de {pdf_path.name}...')

    # Silencia warnings do pdfminer (usado pelo pdfplumber) sobre cores inválidas.
    logging.getLogger('pdfminer').setLevel(logging.ERROR)

    documents = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Extrai texto completo da página.
            text = page.extract_text() or ''

            # Extrai as tabelas da página, se houver, e as formata em Markdown.
            tables = page.extract_tables()
            tables_content = []
            for table in tables:
                # Remove linhas vazias e nulas.
                clean_table = [
                    [cell or '' for cell in row] for row in table if any(row)
                ]
                if not clean_table:
                    continue

                # Formata a tabela em Markdown.
                header = clean_table[0]
                rows = clean_table[1:]

                md_table = (
                    f'| {" | ".join(header)} |\n| {" | ".join(["---"] * len(header))} |'
                )
                for row in rows:
                    md_table += f'\n| {" | ".join(row)} |'

                tables_content.append(md_table)

            # Adiciona o conteúdo das tabelas ao final do texto da página.
            if tables_content:
                text += '\n\n' + '\n\n'.join(tables_content)

            metadata = {'source': str(pdf_path), 'page': page.page_number}
            documents.append(Document(page_content=text, metadata=metadata))

    logger.info(f'Extração concluída: {len(documents)} páginas extraídas de {pdf_path.name}.')

    return documents
