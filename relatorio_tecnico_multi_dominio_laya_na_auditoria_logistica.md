# Relatório Técnico Executivo: Implementação e Escalabilidade Multi-Domínio de Modelos de Decisão System 1 (Laya) em Auditoria Logística de Grande Porte

**Autoridade Técnica:** Auditoria Interna Especializada / Engenharia de Soluções de IA  
**Contexto Operacional:** Varejo e Logística de Grande Escala (Magalog / Magazine Luiza)  
**Escopo Geral:** Gestão de Estoque (WMS - FEFO), Auditoria de Faturamento e Fretes (TMS / Financeiro), Conformidade de SLA, Reversa e Logística Reversa

---

## 1. Sumário Executivo e Contexto Estratégico

Nas operações de varejo e logística de alta densidade, o volume transacional diário gera um montante massivo de exceções operacionais, divergências de inventário e desvios de faturamento (como quebras de FEFO, sobrepreços em faturas de transportadoras e falhas de cumprimento de SLA). A auditoria tradicional, baseada em amostragem humana ou em regras determinísticas rígidas em SQL, esbarra em barreiras intransponíveis: incapacidade de processar o contexto de dados textuais e semi-estruturados, alto custo de manutenção de código e limitação de cobertura de escopo.

A introdução de motores de decisão do tipo System 1 (como o **Laya**) representa uma mudança de paradigma. Tratando-se de uma rede neural compacta de $421\text{M}$ parâmetros baseada em arquitetura codificadora (estilo BERT), o Laya executa classificações e pontuações tipadas em milissegundos ($\approx 33\text{ms}$), operando 100% localmente com custo marginal zero de API e garantia de sigilo absoluto dos dados corporativos. Este relatório consolida o planejamento estrutural, o gerenciamento de ambiente, as diretrizes de *fine-tuning*, a adaptabilidade para múltiplos domínios de auditoria e a argumentação executiva para a adoção da tecnologia.

---

## 2. Arquitetura e Fundamentos do Laya

Diferente de Modelos de Linguagem Grande (LLMs) autoregressivos tradicionais (que geram prosa token a token e apresentam riscos latentes de alucinação), o Laya opera como um **motor de decisão estruturada**. 

* **Mecanismo de Leitura Paralela**: Utiliza camadas de atenção baseadas em transformadores para ler toda a linha de entrada (combinando estado, pergunta e opções) em uma única passada.
* **Saída Probabilística Tipada**: Responde estritamente a primitivas predefinidas (`choice`, `score`, `noul`), retornando pontuações e probabilidades calibradas via softmax.
* **Restrições Arquiteturais**: Por alocar todas as opções em um slot compartilhado de tokens (tipicamente otimizado para até $20$ opções por consulta), o modelo exige estruturação rigorosa das hipóteses de auditoria.
* **Acurácia Base vs. Especializada**: O modelo base pré-treinado apresenta acurácia *zero-shot* limitada ($\approx 36\%$), tornando mandatório o processo de *fine-tuning* com o histórico real da operação logística para alcançar patamares corporativos superiores a $90\%$.

---

## 3. Gerenciamento de Ambiente Virtual com `uv`

Para garantir reprodutibilidade, isolamento e alta performance na execução dos scripts de preparação de dados e treinamento do Laya, a infraestrutura deve ser gerenciada utilizando o `uv` (gerenciador de pacotes e ambientes virtuais ultrarrápido em Rust).

### Vantagens do `uv` na Engenharia de IA:
* **Velocidade de Resolução**: Criação de ambientes virtuais e instalação de dependências até $10\times$ mais rápida que o `pip` tradicional.
* **Determinismo de Versões**: Gestão estrita de dependências corporativas, garantindo que o ambiente de treinamento local espelhe perfeitamente o ambiente de homologação ou nuvem externa (como o Google Colab via túneis seguros).

### Padrão de Inicialização do Ambiente:
```bash
# Criação do ambiente virtual isolado para o domínio de auditoria
uv venv .venv --python 3.11

# Ativação do ambiente
source .venv/bin/activate

# Instalação das dependências essenciais do framework e processamento de dados
uv pip install polars torch transformers scikit-learn
```

---

## 4. Estrutura de Dados e Engenharia de `state`

O sucesso do Laya depende diretamente da conversão de tabelas relacionais complexas (WMS e TMS) em uma representação semântica padronizada. Cada instância de treinamento e inferência é composta por cinco pilares estritos:

1. **`state`**: O espelho textual da transação. Deve combinar chaves imutáveis e delimitadas (ex: `|`).
   * *Exemplo (FEFO)*: `"SKU: 789102-X | Lote: L-20260310 | Entrada WMS: 2026-03-12 | Validade: 2026-06-15 | Dias de Vida Útil Total: 90 | Dias Restantes na Chegada: 15 | Giro Médio Diário: 2.3 unidades | Estoque Atual: 450 un | Zona WMS: Pulmão A3 | Ocorrência: Vencido no Estoque sem giro."`
2. **`question`**: A indagação analítica direcionada ao auditor digital.
   * *Exemplo*: `"Qual foi o fator determinante para a perda deste lote por violação de FEFO?"`
3. **`q_type`**: A primitiva de decisão estrutural (ex: `"choice"`).
4. **`options`**: O conjunto fechado de hipóteses ou causas-raiz (máximo de $20$ opções mutuamente exclusivas).
   * *Exemplo*: `["fornecedor_lote_curto_recebido", "falha_rotacao_operador_picking", "sazonalidade_quebra_demanda", "omissao_alerta_sistema_wms"]`
5. **`target`**: O índice numérico correspondente à resposta correta validada pelo auditor humano no histórico (utilizado apenas no treinamento).

---

## 5. Escalabilidade e Adaptação do Laya para Outras Situações na Auditoria

A flexibilidade do Laya permite que ele seja replicado para diferentes escopos transacionais da cadeia de suprimentos da Magalog e Magazine Luiza. Para cada novo problema, a estrutura lógica é mantida, alterando-se apenas o conteúdo semântico do `state`, a `question` e as `options`.

### A. Domínio 1: Auditoria Preventiva de Faturas de Frete (TMS / Financeiro)
* **Objetivo**: Identificar cobranças indevidas, sobrepreços de peso cubado e taxas adicionais duplicadas antes da liquidação financeira.
* **Exemplo de `state`**: `"Transportadora: ExpressLog | Manifesto: M-99823 | Rota: SP-RJ | Peso Real: 120kg | Peso Cubado: 145kg | Valor Cobrado: R$ 1.250,00 | Valor Tabela TMS: R$ 980,00 | Ocorrência: Divergência de cubagem em taxa de reentrega."`
* **Exemplo de `question`**: `"Qual é a tratativa regulamentar correta para esta fatura de frete?"`
* **Exemplo de `options`**: `["aprovar_integral", "bloquear_divergencia_peso", "estornar_taxa_reentrega", "encaminhar_contestacao_juridica"]`

### B. Domínio 2: Conformidade de SLA e Riscos de Entrega no Ultramel
* **Objetivo**: Triar falhas críticas de entrega e desvios de rota para acionamento de contingência.
* **Exemplo de `state`**: `"Pedido: 88372-9 | Destino: Capital-SP | Prazo Acordado: 2026-03-20 | Status Atual: Retido na Base Avançada | Dias de Atraso: 4 | Ocorrência: Tentativa de entrega frustrada por restrição de circulação municipal."`
* **Exemplo de `question`**: `"Qual o nível de criticidade e ação imediata requerida para este atraso?"`
* **Exemplo de `options`**: `["risco_baixo_reprogramacao_padrao", "risco_medio_contato_cliente", "risco_alto_sinistro_ou_avaria", "extravio_confirmado_acionar_seguro"]`

---

## 6. Metodologia de Fine-Tuning e Amostragem Estratificada

Em corporações de grande porte, alimentar a inteligência artificial com milhões de registros brutos gera ruído e degrada a atenção do modelo.

* **Volume Ideal**: O ponto ótimo de treinamento para cada domínio especialista situa-se entre $15.000$ e $30.000$ exemplos altamente curados.
* **Curadoria de Ground-Truth**: O dataset deve ser extraído do histórico de auditorias sêniores validadas, aplicando amostragem estratificada para equilibrar classes minoritárias (ex: fraudes ou desvios graves) e evitar o vício de predição majoritária.
* **Isolamento de Domínios**: Cada escopo de auditoria (ex: WMS-Estoque vs. TMS-Fretes vs. SLA) deve possuir adaptadores de pesos independentes e ambientes virtuais separados para evitar interferência catastrófica (*catastrophic forgetting*).

---

## 7. Justificativa Corporativa: Laya vs. Regras SQL / Python

Ao defender o projeto perante a diretoria de TI e finanças, os contrapontos estruturais em relação aos métodos tradicionais devem ser claros:

| Critério Comparativo | Regras Tradicionais (SQL / Python Determinístico) | Modelo System 1 (Laya Especialista) |
| :--- | :--- | :--- |
| **Processamento de Texto Livre** | Cego a variações textuais; exige pipelines complexos de PLN e Regex. | **Nativo**: Lê descrições informais de operadores e logs em conjunto com dados numéricos. |
| **Manutenibilidade** | Acúmulo de código espaguete (*if-else* / *CASE WHEN*), exigindo reescrita constante. | **Orgânico**: Atualização de comportamento via fine-tuning baseado em novos exemplos históricos. |
| **Resiliência a Variáveis** | Dificuldade em correlacionar fatores não-lineares sem engenharia exaustiva de *features*. | **Holístico**: Avalia multivariáveis simultaneamente em milissegundos via atenção transformer. |
| **Governança de Incerteza** | Resposta estritamente binária (Verdadeiro/Falso), gerando falsos positivos rígidos. | **Probabilística**: Retorna grau de certeza calibrado, permitindo limiares de triagem para humanos. |

---

## 8. Plano de Implementação e Governança

Para mitigar riscos operacionais e assegurar conformidade com normas de controle interno (IIA / COSO), a implantação seguirá um protocolo rigoroso:

1. **Fase de Homologação Isolada (PoC)**: Seleção de um domínio prioritário (ex: Quebra por FEFO) e extração de amostra histórica validada.
2. **Operação em Shadow Mode (30 Dias)**: O Laya especialista executa em paralelo com a operação real, registrando suas decisões de auditoria sem aplicar bloqueios automáticos, permitindo auditoria humana integral da matriz de confusão (falsos positivos e falsos negativos).
3. **Calibração de Limiar de Confiança**: Configuração de abstenção automática ($confidence < 0.85$), direcionando obrigatoriamente casos ambíguos para a equipe sênior.