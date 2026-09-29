# 🏛️ Arquitetura — gmr_golpe_pix

Documento técnico da solução de fine-tuning e inferência do **Laya** no domínio de detecção
de golpe do Pix. Complementar ao [README.md](../README.md) e ao
[relatório técnico multi-domínio](../relatorio_tecnico_multi_dominio_laya_na_auditoria_logistica.md).

---

## 1. Visão geral

O sistema reproduz o experimento **Pix Race** do professor Sandeco — originalmente
Rust + Jev vs Python + DeepSeek — usando **exclusivamente Python + Laya fine-tunado**,
rodando 100% local em CPU. O pipeline tem 3 grandes blocos:

| Bloco | Onde | O que faz |
|---|---|---|
| **Preparação de dados** | local (`src/preparar_dataset.py`) | Reconstrói o dataset no formato oficial de fine-tuning do Laya |
| **Treinamento** | Google Colab, 1× T4 (`notebooks/treinamento_laya_colab.ipynb`) | RLCD + calibração + exportação ONNX fp32 |
| **Inferência/demo** | local CPU (`src/servidor_web.py`, `src/inferir.py`) | Decisão tipada em 1 forward pass + demo web |

## 2. Fluxo de dados de ponta a ponta

```
┌─ pix-golpe (clone, somente leitura — fonte de dados e do protocolo) ─────────┐
│ data/mensagens.jsonl   1000 msgs · id, ts, remetente, texto, chave, golpe_real│
│ paper/results/.../decisions.csv   2000 decisões (jev + deepseek)             │
│ src/jev.rs::questions()   schema das 4 perguntas tipadas                     │
└───────────────────────────────────────────────────────────────────────────────┘
                 │ mensagens (ground truth)        │ decisions (labels teacher DeepSeek=100% · fallback Jev)
                 ▼                                 ▼
        ┌─────────────────────────────────────────────────────┐
        │ src/preparar_dataset.py  (local · logging UTC-3)    │
        │  state   = {"remetente": ..., "mensagem": ...}      │
        │  questions = 4 tipadas idênticas ao jev.rs          │
        │  gold    = golpe (verdade dura) + tipo/urgencia/    │
        │            pede_pix (teacher, one-hot)              │
        │  split estratificado por golpe_real, semente 42     │
        └─────────────────────────────────────────────────────┘
                 │ data/processed/laya_pix_golpe_treino.jsonl   (800 msgs × 4 perguntas = 3200 decisões)
                 │ data/processed/laya_pix_golpe_validacao.jsonl (200 msgs)
                 ▼   upload manual no Colab
┌─ notebooks/treinamento_laya_colab.ipynb  (Google Colab T4) ───────────────────┐
│ 1 GPU check · 2 pip laya · 3 upload dataset · 4 tokenização (checkpoint      │
│   convaiinnovations/laya → subpasta multilingual, mmBERT-base 322M)          │
│ 5 treino RLCD: GRPO (σ 0.4→0.1, grupo=4) + CE · AdamW (2.5e-5/1e-4) ·        │
│   4 épocas · batch efetivo 64 · checkpoint rolante/época                     │
│ 6 avaliação: acurácia por pergunta · ECE · veredito 3 faixas                 │
│ 7 exportação ONNX fp32 (scripts/export_onnx.py oficial, sem --quantize)      │
│ 8 zip + download                                                             │
└───────────────────────────────────────────────────────────────────────────────┘
                 │ models/laya_pix_golpe_finetuned/  (weights + tokenizer + rl_agent_config + laya.int8.onnx)
                 ▼
┌─ Inferência local (CPU, sem GPU) ─────────────────────────────────────────────┐
│ src/laya_comum.py    carregar_agente: prefere ONNX fp32 (onnxruntime) →       │
│                      fallback PyTorch fp16 · veredito 3 faixas · logging      │
│ src/servidor_web.py  FastAPI + WebSocket · percorre as 1000 mensagens ·       │
│                      relatório (matriz 3×2, acurácia, precisão, recall)       │
│ static/index.html    UI dark: feed ao vivo, placar, relatório                 │
│ src/inferir.py       CLI single-shot / interativo                             │
└───────────────────────────────────────────────────────────────────────────────┘
```

## 3. Decisões de arquitetura

### 3.1 Reconstrução do dataset (distilação do teacher)
O repositório `dados-laya` do professor não foi publicado. O dataset de treino foi
reconstruído a partir de artefatos públicos do `pix-golpe`:

- **`golpe`**: rótulo duro da verdade de referência (`golpe_real`) — não do teacher;
- **`tipo` / `urgencia` / `pede_pix`**: labels do teacher **DeepSeek** (acurácia 1.000 na
  corrida de referência), com fallback para o **Jev** (99,9%) no único id sem resposta (id 7);
- `state` e `questions` replicam **byte a byte** o que era enviado aos modelos no experimento.

Motivação: fine-tuning por distilação de um professor perfeito no dataset + ground truth real
para a decisão crítica. É o mesmo espírito do notebook oficial (targets derivados de teacher).

### 3.2 Checkpoint multilingual
As mensagens são em português: usamos `convaiinnovations/laya` subpasta **`multilingual`**
(mmBERT-base, 322M, 100+ idiomas, até 8192 tokens) em vez do checkpoint inglês
(ModernBERT-large), que degrada em texto não-inglês.

### 3.3 Treino em GPU gratuita (1× T4 em vez de 2× T4 do notebook oficial)
Adaptação do `train_ddp.py` oficial: removido o `DistributedDataParallel`, `GRAD_ACCUM=8`
mantém o **batch efetivo em 64** sequências. Hiperparâmetros 1:1 com o oficial
(épocas 4, LRs 2.5e-5/1e-4, σ 0.4→0.1, `w_sph=0.75`, CE peso 1.0). Slice de calibração
(10%, máx. 400) retido **antes** do treino — temperaturas honestas, sem vazamento.

### 3.4 ONNX fp32 para CPU doméstica
A máquina-alvo local tem ~5,5 GB de RAM; o load PyTorch fp16 (~2 GB de pico) era morto pelo
OOM killer. A exportação **ONNX** (`scripts/export_onnx.py`, oficial do Laya) roda com
`onnxruntime` sem carregar o grafo em PyTorch.

Decisão empírica (29/09/2026): testamos primeiro a quantização dinâmica INT8 por canal
(`--quantize`) — drifto desprezível no checkpoint base segundo o autor do Laya, **mas no
nosso checkpoint fine-tunado ela achatou os logits da head** (temperaturas calibradas ≈ 5,5
indicam gaps de logit extremos): todas as probabilidades `noul` colapsaram para p≈0,5 na
validação local. Exportamos então em **fp32** (~1,3 GB), que preserva as decisões.
`carregar_agente` prefere `laya.onnx` e cai no PyTorch apenas como fallback.

### 3.5 Protocolo de veredito em 3 faixas (idêntico ao paper)
`golpe ≥ 0,6` → **GOLPE** · `golpe ≤ 0,4` → **OK** · entre → **REVISAR**.
Na métrica binária, só GOLPE conta como positivo (revisar = golpe perdido quando era golpe).
Isso torna nossos números comparáveis 1:1 com o `paper/` do repositório original.

## 4. Contratos

### 4.1 Linha do dataset (JSONL) — formato de fine-tuning do Laya
```json
{
  "id": "815",
  "workflow": "pix_golpe",
  "state": "{\"remetente\": \"...\", \"mensagem\": \"...\"}",
  "questions": "{\"golpe\": {\"type\": \"noul\", ...}, \"tipo\": {\"type\": \"choice\", ...}}",
  "gold": "{\"golpe\": {\"label\": \"false\", \"probabilities\": {...}}, ...}"
}
```

### 4.2 WebSocket da demo (`src/servidor_web.py`, porta 8080)
- Cliente → servidor: `{"cmd": "iniciar", "intervalo": 0.0, "limite": 0}` · `{"cmd": "parar"}`
- Servidor → cliente: `{"type": "iniciado", "total": N}` ·
  `{"type": "decisao", id, remetente, texto, golpe_real, prob_golpe, veredito, tipo,
  tipo_confianca, urgencia, pede_pix, latencia_ms, i, placar}` ·
  `{"type": "fim", "motivo", "relatorio": {matriz, acuracia, precisao, recall, ...}}`

## 5. Qualidade e observabilidade

- **Tipagem estrita PEP 484** + regras **ANN** do Ruff em todo `src/` (`ruff check src/`).
- **Logging centralizado** com fuso oficial de Brasília (`America/Sao_Paulo`, UTC-3):
  console colorido (colorama) + arquivo por execução em `logs/`.
- **Configuração única** em `config/laya_config.json` (caminhos, checkpoint, hiperparâmetros).
- Reprodutibilidade: semente 42 (dataset e split), gerador ChaCha8 semente 42 no upstream.

## 6. Limitações conhecidas

- Dataset **sintético templated**: a acurácia de validação (0,980 global / 1,000 no veredito)
  mede especialização no domínio sintético, não generalização para golpes reais inéditos.
- Temperaturas calibradas ~5 são truncadas ao teto 5,0 do Laya no load (aviso benigno;
  ECE medido 0,012).
- Classes raras de `tipo` (ex.: `pix_errado`, 6 exemplos) concentram os poucos erros —
  alavanca futura: data augmentation/oversampling.

## 7. Referências

- [sandeco/pix-golpe](https://github.com/sandeco/pix-golpe) — dados, protocolo, perguntas tipadas;
- [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) — motor System 1 (Apache-2.0),
  [notebook oficial de fine-tuning](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb)
  e [`scripts/export_onnx.py`](https://github.com/NandhaKishorM/laya/blob/main/scripts/export_onnx.py);
- [Vídeo do professor Sandeco](https://www.youtube.com/watch?v=YGuLBJ6af_o) — origem do projeto;
- [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) — checkpoints no Hugging Face.
