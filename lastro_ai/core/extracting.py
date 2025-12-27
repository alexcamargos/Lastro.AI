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


def _is_valid_line(line: str) -> bool:
    """Verifica se a linha contém informação relevante.

    Args:
        line (str): Linha de texto a ser avaliada.
    Returns:
        bool: True se a linha for relevante, False caso contrário.
    """

    # Filtros simples de cabeçalho/rodapé comuns.
    _DISCARD_PATTERNS = [
        'Banco Central do Brasil',
        'Relatório de Inflação',
        'Relatório de Política Monetária'
    ]

    stripped = line.strip()

    # Remove números de página isolados que estejam isolados.
    if len(stripped) < 5 and stripped.isdigit():
        return False

    # Remove padrões de ruído conhecidos.
    if any(pattern in stripped for pattern in _DISCARD_PATTERNS):
        return False

    return True


def _clean_text(text: str) -> str:
    """Realiza limpeza básica do texto extraído.

    Remove cabeçalhos e rodapés comuns, bem como linhas muito curtas
    que não agregam valor ao conteúdo principal.

    Args:
        text (str): Texto a ser limpo.

    Returns:
        str: Texto limpo.
    """

    # Remove linhas vazias.
    if not text:
        return ''

    cleaned_lines = [
        line for line in text.split('\n') if _is_valid_line(line)
    ]

    return '\n'.join(cleaned_lines)


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
            raw_text = page.extract_text() or ''
            # Limpa o texto extraído.
            text = _clean_text(raw_text)

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
