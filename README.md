# Lastro.AI

> _**Lastro.AI**: Onde a complexidade da política monetária encontra a clareza da IA, transformando relatórios estáticos em conhecimento dinâmico e preciso._

[![LinkedIn](https://img.shields.io/badge/%40alexcamargos-230A66C2?style=social&logo=LinkedIn&label=LinkedIn&color=white)](https://www.linkedin.com/in/alexcamargos)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

---

## Sobre o Projeto

O **Lastro.AI** é um agente de Inteligência Artificial especializado projetado para decodificar e democratizar o acesso aos complexos _Relatórios de Política Monetária (RPM) e Relatórios de Inflação (RI)_ do Banco Central do Brasil.

Em um cenário onde a informação econômica é crucial mas frequentemente inacessível devido à linguagem técnica e formato PDF, este projeto atua como uma ponte. Por meio de **Processamento de Linguagem Natural (NLP)** e **Retrieval-Augmented Generation (RAG)**, o sistema ingere documentos oficiais, cria uma base de conhecimento vetorial e permite que pesquisadores, economistas e estudantes façam perguntas em linguagem natural, obtendo respostas fundamentadas exclusivamente nos dados oficiais.

### Principais Funcionalidades
- **Ingestão Automatizada**: Download e processamento automático de relatórios históricos do BACEN.
- **Busca Semântica Avançada**: Utiliza embeddings e re-ranking para encontrar trechos relevantes não apenas por palavras-chave, mas pelo significado da pergunta.
- **Rastrabilidade**: Toda resposta gerada cita a fonte exata (documento e página), eliminando alucinações.
- **Interface Conversacional**: Chatbot interativo disponível via web (Chainlit) ou terminal.

## Instalação e Uso

Este projeto utiliza o gerenciador de pacotes `uv` para garantir reprodutibilidade e velocidade.

### Pré-requisitos
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) instalado.
- Chave de API da Groq (para o LLM).

### 1. Clonar o Repositório
```bash
git clone https://github.com/alexcamargos/lastro.ai.git
cd lastro.ai
```

### 2. Configurar o Ambiente
Crie um arquivo `.env` na raiz do projeto com suas credenciais:
```env
# .env
GROQ_API_KEY="sua_chave_api_aqui"
```

### 3. Instalar Dependências
```bash
uv sync
```

### 4. Construir a Base de Conhecimento
Antes de perguntar, o sistema precisa "ler" os relatórios. O comando abaixo baixa e processa documentos do último ano:
```bash
uv run main.py process_batch 1
```

### 5. Executar a Aplicação
Inicie a interface web:
```bash
uv run main.py web
```
Ou faça perguntas diretas via terminal:
```bash
uv run main.py ask "Qual a projeção de inflação para 2025?"
```

**Exemplo de Resposta Real:**
> "No Relatório de Inflação 2024/12 (Trecho 4), as projeções foram revisadas indicando uma inflação de **4,5%** para 2025... A projeção mais recente reflete uma atualização condicional às trajetórias da taxa Selic e do câmbio."
*(O sistema cruza fontes diferentes, identifica a data mais recente e justifica a resposta)*

## Demonstração e Exemplos

### 1. Terminai (CLI)
A interface de linha de comando é ideal para integrações e perguntas rápidas.

```bash
# Pergunta
uv run main.py ask "Por que o COPOM decidiu manter a taxa Selic?"

# Resposta do Agente (Output Real)
"""
O COPOM decidiu manter a taxa Selic em 15,00% a.a. por três razões principais, conforme destacado nos relatórios:

1. **Convergência da inflação à meta**: A manutenção da Selic busca assegurar a convergência da inflação para o redor da meta de 3,00% ao longo do horizonte relevante. O Comitê avalia que uma política monetária contracionista por período prolongado é necessária para consolidar o processo de desinflação e reancorar as expectativas de inflação (Trechos 1, 2 e 4).

2. **Cenário de incertezas e desancoragem de expectativas**: O cenário atual é marcado por expectativas desancoradas, projeções de inflação elevadas, resiliência da atividade econômica e pressões no mercado de trabalho. Esses fatores exigem cautela na condução da política monetária, com manutenção de juros em patamar significativamente contracionista para mitigar riscos (Trechos 1, 2 e 4).

3. **Avaliação dos impactos acumulados**: O Comitê interrompeu temporariamente o ciclo de alta de juros para examinar os efeitos acumulados do ajuste anterior. A manutenção da Selic permite observar se o nível atual é suficiente para garantir a convergência da inflação à meta, mantendo flexibilidade para ajustes futuros (Trechos 1 e 4).

Além disso, o COPOM enfatiza que seguirá vigilante, podendo retomar ajustes na taxa se julgar necessário, mantendo o foco na estabilidade de preços e no equilíbrio entre desinflação e suavização das flutuações da atividade econômica (Trechos 1, 2 e 4).
"""
```

### 2. Interface Web (Chainlit)
Para uma experiência visual completa, a interface web oferece histórico de chat, visualização de fontes e *thought process* (cadeia de pensamento).

![Demonstração da Interface Web](assets/demo_web.gif)
*Acima: O sistema analisando um relatório de 80 páginas em segundos para responder sobre o 'Hiato do Produto'.*

## Stack Tecnológico

O projeto foi construído sobre um stack moderno de Python, priorizando performance e tipagem estática.

*   **Linguagem**: Python 3.12 (com Type Hints rigorosos).
*   **Orquestração de LLM**: [LangChain](https://www.langchain.com/) para construção das cadeias de RAG.
*   **Interface Web**: [Chainlit](https://chainlit.io/) para uma experiência de chat rápida e reativa.
*   **Banco Vetorial**: [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search) para busca de alta performance.
*   **CLI**: [Fire](https://github.com/google/python-fire) para criação automática de interfaces de linha de comando.
*   **Gerenciamento de Dependências**: [uv](https://github.com/astral-sh/uv) (substituto moderno ao Pip/Poetry).

## Decisões Arquiteturais e Técnicas

Como um projeto focado em **Data Science e Engenharia de IA**, cada componente foi escolhido para resolver um problema específico de escalabilidade ou precisão.

### 1. Retrieval-Augmented Generation (RAG)
Optou-se por uma arquitetura RAG em vez de fine-tuning.
*   **Por quê?** Relatórios do BACEN mudam trimestralmente. RAG permite atualizar o conhecimento do modelo instantaneamente apenas adicionando novos documentos ao índice, sem o custo e tempo de re-treinar o modelo. Além disso, RAG mitiga alucinações ao forçar o modelo a usar o contexto recuperado.

### 2. Modelo de LLM: Llama 3 / Mixtral (via Groq)
O sistema utiliza a API da Groq para inferência.
*   **Por quê?** A Groq oferece a menor latência de inferência do mercado (LPU), essencial para uma experiência de chat fluida. Os modelos Open Source (Llama 3 70B ou Mixtral 8x7B) demonstraram capacidade de raciocínio comparável ao GPT-4 para tarefas de sumarização e extração em português, com custo zero ou muito reduzido.

### 3. Embeddings e Re-Ranking (Dois Estágios)
A recuperação de informação é feita em duas etapas para maximizar a precisão (Hit Rate).
*   **Estágio 1 (Recall)**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Um modelo bi-encoder rápido que busca os 20 documentos topologicamente mais próximos.
*   **Estágio 2 (Precision)**: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`. Um modelo Cross-Encoder que re-avalia e pontua os documentos recuperados.
*   **Por quê?** Busca vetorial simples falha em nuances linguísticas. O Cross-Encoder atua como um "juiz" detalhista, garantindo que o contexto entregue ao LLM seja extremamente relevante, dobrando a precisão do sistema.

## Estudo de Caso: Problema, Solução e Aprendizado

### Problema
Profissionais que dependem de dados do Banco Central perdem horas lendo PDFs de 80 páginas em busca de parágrafos específicos sobre "metas de inflação" ou "cenário externo". A busca por palavras-chave (Ctrl+F) é falha pois o vocabulário técnico varia ("inflação" vs "IPCA" vs "nível de preços").

### Solução
Desenvolvimento de um pipeline de ETL não estruturado que converte PDFs em vetores semânticos. O sistema não busca por palavras, mas por **conceitos**. Se o usuário pergunta sobre "custo de vida", o sistema recupera trechos sobre "IPCA", mesmo que a palavra "custo" não apareça, graças aos embeddings multilingues.

### Aprendizado
*   **Qualidade dos Dados > Modelo**: A maior melhoria de performance não veio de trocar o LLM, mas de melhorar o "chunking" (segmentação) dos PDFs e limpar cabeçalhos/rodapés que poluíam a busca.
*   **Avaliação Quantitativa**: Implementar métricas como "Hit Rate" foi essencial para sair do "acho que melhorou" para "melhorou 40%". A engenharia de IA exige rigor científico tanto quanto engenharia de software tradicional.

---

## Autor

Feito com ❤️ por [Alexsander Lopes Camargos](https://github.com/alexcamargos) 👋 Entre em contato!

[![GitHub](https://img.shields.io/badge/-AlexCamargos-1ca0f1?style=flat-square&labelColor=1ca0f1&logo=github&logoColor=white&link=https://github.com/alexcamargos)](https://github.com/alexcamargos)
[![Twitter Badge](https://img.shields.io/badge/-@alcamargos-1ca0f1?style=flat-square&labelColor=1ca0f1&logo=twitter&logoColor=white&link=https://twitter.com/alcamargos)](https://twitter.com/alcamargos)
[![Linkedin Badge](https://img.shields.io/badge/-alexcamargos-1ca0f1?style=flat-square&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/alexcamargos/)](https://www.linkedin.com/in/alexcamargos/)
[![Gmail Badge](https://img.shields.io/badge/-alcamargos@vivaldi.net-1ca0f1?style=flat-square&labelColor=1ca0f1&logo=Gmail&logoColor=white&link=mailto:alcamargos@vivaldi.net)](mailto:alcamargos@vivaldi.net)

## Licença
Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.
