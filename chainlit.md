# Bem-vindo ao Lastro.AI

**Análise Inteligente de Política Monetária**

O **Lastro.AI** não é apenas um chatbot; é um analista econômico sintético. Ele combina a capacidade de raciocínio de grandes modelos de linguagem (LLMs) com a precisão dos documentos oficiais do Banco Central do Brasil.


## Como o Sistema Funciona?

Diferente de IAs genéricas (como o ChatGPT padrão) que podem "alucinar" ou usar dados desatualizados, o Lastro.AI opera em um ciclo rigoroso de **RAG (Retrieval-Augmented Generation)**:

1.  **Busca Semântica & Re-Ranking**: Quando você faz uma pergunta, o sistema varre milhares de páginas dos Relatórios de Inflação e Atas do COPOM. Ele usa um modelo Cross-Encoder avançado para selecionar os 5 trechos mais relevantes, mesmo que não contenham as palavras exatas da sua pergunta.
2.  **Análise Fundamentada**: O Agente recebe *apenas* esses trechos oficiais como sua única fonte de verdade.
3.  **Resposta Citada**: A resposta gerada cita explicitamente quais documentos e páginas fundamentaram a conclusão (ex: *"Conforme Relatório de Inflação 03/2024, Trecho 2"*).


## O que você pode perguntar?

Tente perguntas complexas que exigiriam leitura de vários documentos:

- *"Qual a diferença na projeção de inflação entre o cenário de referência e o alternativo?"*
- *"Quais foram os principais riscos fiscais citados na última Ata do COPOM?"*
- *"O que o relatórios diz sobre o impacto do cenário externo na taxa de câmbio?"*
- *"Compare a visão sobre o hiato do produto em 2024 versus 2025."*


## Fonte de Dados Oficial

A base de conhecimento é composta exclusivamente por documentos públicos do **Banco Central do Brasil**:
- **Relatório de Inflação (RI)**
- **Relatório de Política Monetária (RPM)**


Acesse os originais: [bcb.gov.br/publicacoes/rpm](https://www.bcb.gov.br/publicacoes/rpm)


## Sobre

Este projeto é um portfólio de Engenharia de IA focado em **precisão e transparência**.
*   **Desenvolvedor:** [Alexsander Lopes Camargos](https://github.com/alexcamargos)
*   **Licença:** MIT
