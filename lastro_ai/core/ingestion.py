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


import datetime
from pathlib import Path

import requests
from loguru import logger

from lastro_ai import config as Cfg


def download_report(url: str,
                    save_path: Path,
                    timeout: tuple[int, int] = Cfg.DOWNLOAD_TIMEOUT,
                    chunk_size: int = Cfg.DOWNLOAD_CHUNK_SIZE) -> Path:
    """Realiza o download de um Relatório de Política Monetária do BACEN.

    Executa uma requisição HTTP GET para a URL fornecida e salva o conteúdo
    no caminho especificado. Utiliza streaming para lidar eficientemente com
    arquivos grandes, escrevendo em chunks no disco.

    Args:
        url (str): URL do relatório PDF a ser baixado.
        save_path (Path): Caminho completo onde o arquivo será salvo.
        timeout (tuple[int, int], optional): Tupla definindo timeouts de conexão e leitura.
                                             Padrão definido em config.DOWNLOAD_TIMEOUT.
        chunk_size (int, optional): Tamanho do buffer de leitura/escrita em bytes.
                                    Padrão definido em config.DOWNLOAD_CHUNK_SIZE.

    Returns:
        Path: Caminho absoluto do arquivo salvo no disco.

    Raises:
        requests.exceptions.RequestException: Caso ocorra erro na conexão ou download.
    """

    logger.info(f"Baixando relatório de: {url}...")

    with requests.get(url, timeout=timeout, stream=True) as response:
        # Levanta um erro para códigos de status HTTP ruins.
        response.raise_for_status()

        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                file.write(chunk)

    file_size = save_path.stat().st_size / (1024 * 1024)
    logger.info(f"Relatório salvo em: {save_path} ({file_size:.2f} MB)")

    return save_path


def download_reports_batch(years: int, save_dir: Path) -> list[Path]:
    """Automatiza o download em lote dos relatórios do BACEN para um período histórico.

    Itera sobre os anos e trimestres especificados, construindo as URLs corretas
    com base na mudança de nomenclatura dos relatórios (RI até 2024, RPM após).
    Gerencia falhas de download individualmente, garantindo que o processo continue
    para os demais arquivos mesmo se um falhar.

    Args:
        years (int): Número de anos a retroceder a partir do ano atual.
        save_dir (Path): Diretório de destino para salvar os arquivos PDF.

    Returns:
        list[Path]: Lista contendo os caminhos completos dos arquivos baixados com sucesso.
    """

    today = datetime.date.today()
    current_year = today.year
    start_year = current_year - years

    downloaded_count = 0
    downloaded_files = []
    errors = 0

    for year in range(start_year, current_year + 1):
        for month in Cfg.REPORT_QUARTERS:
            # Evitar tentar baixar relatórios futuros
            if year == current_year and int(month) > today.month:
                continue

            # Lógica para correção da nomeação dos arquivos (RI vs RPM).
            # Entre 1999 e 2024: Relatório de Inflação (RI).
            # Após 2024: Relatório de Política Monetária (RPM).
            template = Cfg.RI_TEMPLATE if year <= 2024 else Cfg.RPM_TEMPLATE

            url = Cfg.BASE_REPORT_URL + template.format(year=year, month=month)
            filename = url.split("/")[-1]
            save_path = Path(save_dir) / filename

            try:
                download_report(url, save_path)
                downloaded_count += 1
                downloaded_files.append(save_path)
            except requests.exceptions.HTTPError as error:
                if error.response.status_code == 404:
                    logger.warning(f"Relatório não encontrado (ou não publicado): {url}")
                else:
                    logger.error(f"Erro ao baixar {url}: {error}")
                errors += 1
            except requests.exceptions.RequestException as error:
                logger.error(f"Erro inesperado em {url}: {error}")
                errors += 1

    logger.info(f"Download concluído: {downloaded_count} arquivos baixados com {errors} erros.")

    return downloaded_files
