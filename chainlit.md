# Bem-vindo ao Lastro.AI

**Lastro.AI** é o seu Agente de Política Monetária Inteligente.

Esta aplicação utiliza Inteligência Artificial Generativa e técnicas de RAG (*Retrieval-Augmented Generation*) para transformar relatórios estáticos do Banco Central do Brasil em conhecimento dinâmico, permitindo análises precisas sobre inflação, juros e cenários econômicos.

---

## Engenharia de Prompt

O agente opera com uma **persona especializada** em análise econômica, configurada para garantir a máxima fidelidade aos dados oficiais.

### Diretrizes do Sistema:
1.  **Contexto Fechado:** As respostas baseiam-se **exclusivamente** nos trechos recuperados dos relatórios (RAG).
2.  **Precisão:** Citação exata de dados numéricos, datas e projeções.
3.  **Tom:** Técnico, analítico e direto, evitando opiniões ou conhecimento externo não verificado.

---

## Fonte de Dados

A base de conhecimento é alimentada diretamente pelos documentos oficiais do **Banco Central do Brasil (BACEN)**:

O **Relatório de Política Monetária (RPM)** apresenta as diretrizes das políticas adotadas pelo Copom e sua avaliação da evolução recente e das perspectivas da economia, especialmente as projeções de inflação. Conjuntamente com o Comunicado e a Ata do Copom, o RPM é um dos principais documentos da política monetária. Essas publicações promovem a transparência das decisões do Comitê, fazendo parte do processo de prestação de contas à sociedade e contribuindo para a eficácia da política monetária. Entre 1999 e 2024, este documento era denominado **Relatório de Inflação (RI)**.

Acesse os relatórios originais em: https://www.bcb.gov.br/publicacoes/rpm

Os documentos são processados, segmentados e indexados vetorialmente para permitir buscas semânticas de alta relevância.

---

## Sobre o Projeto

- **Desenvolvedor:** Alexsander Lopes Camargos
- **Licença:** MIT
