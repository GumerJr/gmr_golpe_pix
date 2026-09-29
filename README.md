# 🛡️ gmr_golpe_pix — Fine-Tuning do Laya para Detecção de Golpe do Pix

Fine-tuning do **[Laya](https://github.com/NandhaKishorM/laya)** (motor de decisão tipado
System 1, não-autoregressivo) no domínio de **detecção de golpe do Pix**, reproduzindo com
Python + Laya o experimento [Pix Race](https://github.com/sandeco/pix-golpe) do professor
[Sandeco](https://www.youtube.com/watch?v=YGuLBJ6af_o) — originalmente Rust + Jev vs
Python + DeepSeek — usando **apenas o nosso modelo fine-tunado**, rodando **100% local em CPU**.

> Dataset: 1000 mensagens sintéticas em português (75 golpes / 925 legítimas), semente fixa 42,
> geradas por templates — nenhum dado bancário real.

---

## 📊 Resultados (validação — 200 mensagens, 800 decisões)

| Pergunta tipada | Acurácia |
|---|---|
| `golpe` (noul) | **1.000** |
| `tipo` (choice, 8 classes) | **0.930** |
| `urgencia` (score 0–2) | **0.990** (MAE 0.027) |
| `pede_pix` (noul) | **1.000** |
| **GLOBAL** | **0.980** · ECE 0.012 |

**Veredito em 3 faixas** (protocolo do paper: `p ≥ 0,6` → golpe, `p ≤ 0,4` → ok, senão revisar):
**acurácia 1.000 · precisão 1.000 · recall 1.000** — paridade com o teacher DeepSeek (1.000)
e acima do Jev zero-shot (99,9%), contra ~36% de acurácia zero-shot do checkpoint base.

## 🔄 Pipeline

```
pix-golpe (clone, referência)                 Google Colab (T4)                Notebook local (CPU)
─────────────────────────────                ──────────────────────           ─────────────────────────────
data/mensagens.jsonl  ──┐
data/raw/decisions.csv ─┼─► src/preparar_dataset.py ─► data/processed/*.jsonl ─► treino RLCD (4 épocas)
 (teacher DeepSeek)     │      (labels: golpe_real + teacher)                    + calibração de temperatura
                        │                                                        + exportação ONNX INT8
                        │                                                              │ zip
                        ▼                                                              ▼
              4 perguntas tipadas do paper:                                  models/laya_pix_golpe_finetuned/
              golpe · tipo · urgencia · pede_pix                                     │
                        ▲                                              src/inferir.py (CLI) · src/servidor_web.py
                        └───────────────────────── respostas do modelo ◄── http://localhost:8080 (demo web)
```

1. **`src/preparar_dataset.py`** — cruza `mensagens.jsonl` (verdade de referência) com
   `decisions.csv` (teacher DeepSeek, 100% de acurácia) e gera o dataset no formato oficial de
   fine-tuning do Laya (`state` / `questions` / `gold`), com split estratificado 800/200.
2. **`notebooks/treinamento_laya_colab.ipynb`** — treino no Google Colab (1× T4): loop RLCD
   (policy gradient GRPO + cross-entropy), calibração de temperatura e exportação **ONNX fp32**
   para inferência em CPU sem PyTorch.
3. **`src/servidor_web.py`** + **`static/index.html`** — demo web (FastAPI + WebSocket) que
   percorre as 1000 mensagens ao vivo e exibe o relatório de acurácia no protocolo do paper.
4. **`src/inferir.py`** — CLI de classificação de mensagens (única ou interativo).

## 🚀 Quickstart

```bash
# 1. Ambiente (uv)
uv venv --prompt gmr_pix_golpe
uv pip install colorama "fastapi>=0.115" "uvicorn[standard]" "laya[onnx]>=0.3.21"

# 2. Dataset (já versionado em data/processed/; para regerar)
.venv/bin/python src/preparar_dataset.py

# 3. Treino: abrir notebooks/treinamento_laya_colab.ipynb no Google Colab (T4),
#    fazer upload dos dois JSONL de data/processed/ e executar todas as células.
#    A última célula baixa laya_pix_golpe_finetuned.zip.

# 4. Modelo local
mkdir -p models/laya_pix_golpe_finetuned
unzip ~/Downloads/laya_pix_golpe_finetuned.zip -d models/laya_pix_golpe_finetuned/

# 5. Demo web  →  http://localhost:8080
.venv/bin/python src/servidor_web.py

# 6. CLI interativo
.venv/bin/python src/inferir.py
```

## 🗂️ Estrutura

```
├── config/laya_config.json      # ⚙️ caminhos, checkpoint, hiperparâmetros
├── data/raw/                    # 📥 mensagens.jsonl + decisions.csv (do pix-golpe)
├── data/processed/              # 🏗️ dataset no formato Laya (treino/validação/resumo)
├── docs/ARQUITETURA.md          # 🏛️ decisões e fluxo de dados
├── logs/                        # 📝 logging centralizado (fuso America/Sao_Paulo)
├── models/                      # 🧠 checkpoint fine-tunado (não versionado — ver .gitignore)
├── notebooks/                   # 📓 treinamento_laya_colab.ipynb
├── src/                         # 🐍 laya_comum · preparar_dataset · inferir · servidor_web
├── static/index.html            # 🌐 UI da demo
└── relatorio_tecnico_multi_dominio_laya_na_auditoria_logistica.md  # 📄 contexto do relatório
```

Padrões de código: tipagem estrita PEP 484 · regras **ANN** do Ruff (`ruff check src/`) ·
logging centralizado com horário de Brasília (UTC-3) · colorama.

## 📚 Referências

- [sandeco/pix-golpe](https://github.com/sandeco/pix-golpe) — dataset sintético, experimento
  Pix Race (Rust + Jev × Python + DeepSeek), protocolo de veredito em 3 faixas e `paper/` de
  reprodutibilidade. Fonte dos dados e das 4 perguntas tipadas (`src/jev.rs`).
- [Vídeo: JEV vs LAYA — RETREINEI o LAYA](https://www.youtube.com/watch?v=YGuLBJ6af_o) — aula
  do professor Sandeco que originou este projeto.
- [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) — motor de decisão System 1
  (Apache-2.0), checkpoints [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya)
  (subpasta `multilingual`, mmBERT-base) e
  [notebook oficial de fine-tuning](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb)
  do qual nosso treino foi adaptado (2×T4 → 1×T4).
- [`scripts/export_onnx.py`](https://github.com/NandhaKishorM/laya/blob/main/scripts/export_onnx.py) —
  exportação ONNX usada para inferência local em CPU (usamos fp32; a variante INT8 degradou
  as decisões neste checkpoint — ver `docs/ARQUITETURA.md` §3.4).
