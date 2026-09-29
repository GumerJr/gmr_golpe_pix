"""Preparação do dataset de fine-tuning do Laya para detecção de golpe do Pix.

Reconstrói o dataset no formato oficial de fine-tuning do Laya
(id, workflow, state, questions, gold) a partir de:

- ``data/raw/mensagens.jsonl``: 1000 mensagens com a verdade de referência
  (``golpe_real``) do repositório sandeco/pix-golpe;
- ``data/raw/decisions.csv``: decisões do experimento Pix Race. O lado
  ``python`` (DeepSeek, 100% de acurácia no benchmark) atua como teacher
  para os rótulos de ``tipo``, ``urgencia`` e ``pede_pix``; o lado ``rust``
  (Jev) é o fallback quando o teacher não respondeu uma mensagem.

As 4 perguntas tipadas replicam exatamente ``questions()`` de
``pix-golpe/src/jev.rs``:

- ``golpe`` (``noul``): rótulo duro da verdade de referência;
- ``tipo`` (``choice``): rótulo one-hot do teacher entre 8 classes;
- ``urgencia`` (``score`` 0-2): nível arredondado do teacher;
- ``pede_pix`` (``noul``): rótulo binário do teacher (p >= 0.5).

Uso::

    .venv/bin/python src/preparar_dataset.py [--config config/laya_config.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from colorama import init as colorama_init

from laya_comum import (
    FUSO_BRASILIA,
    LIMIAR_NOUL,
    NIVEIS_URGENCIA,
    QUESTIONS,
    TIPOS_CRITERIA,
    configurar_logging,
)


@dataclass(frozen=True)
class Mensagem:
    """Mensagem sintética do dataset Pix Race com a verdade de referência."""

    id: int
    ts: int
    remetente: str
    texto: str
    chave: str
    golpe_real: bool


@dataclass(frozen=True)
class DecisaoProfessor:
    """Decisão de um lado do experimento (teacher DeepSeek ou fallback Jev)."""

    lado: str
    tipo: str
    urgencia: float
    pede_pix: float


def carregar_config(caminho_config: Path, logger: logging.Logger) -> dict[str, Any]:
    """Carrega o JSON de configuração central do projeto.

    Args:
        caminho_config: Caminho do ``laya_config.json``.
        logger: Logger central.

    Returns:
        Dicionário de configuração validado.

    Raises:
        FileNotFoundError: Se o arquivo de configuração não existir.
    """
    if not caminho_config.is_file():
        raise FileNotFoundError(f"Configuração ausente: {caminho_config}")
    config: dict[str, Any] = json.loads(caminho_config.read_text(encoding="utf-8"))
    logger.info("⚙️ Configuração carregada de %s", caminho_config)
    return config


def carregar_mensagens(caminho: Path, logger: logging.Logger) -> list[Mensagem]:
    """Lê o ``mensagens.jsonl`` do Pix Race preservando a ordem original.

    Args:
        caminho: Caminho do arquivo JSONL de mensagens.
        logger: Logger central.

    Returns:
        Lista de mensagens tipadas.

    Raises:
        FileNotFoundError: Se o dataset bruto não existir.
    """
    if not caminho.is_file():
        raise FileNotFoundError(f"Dataset de mensagens ausente: {caminho}")
    mensagens: list[Mensagem] = []
    with caminho.open(encoding="utf-8") as handle:
        for linha in handle:
            if not linha.strip():
                continue
            bruto: dict[str, Any] = json.loads(linha)
            mensagens.append(
                Mensagem(
                    id=int(bruto["id"]),
                    ts=int(bruto["ts"]),
                    remetente=str(bruto["remetente"]),
                    texto=str(bruto["texto"]),
                    chave=str(bruto["chave"]),
                    golpe_real=bool(bruto["golpe_real"]),
                )
            )
    golpes = sum(1 for m in mensagens if m.golpe_real)
    logger.info(
        "📥 Mensagens carregadas: %d (%d golpes / %d legítimas)", len(mensagens), golpes, len(mensagens) - golpes
    )
    return mensagens


def carregar_decisoes(
    caminho: Path, lado_teacher: str, lado_fallback: str, logger: logging.Logger
) -> dict[int, DecisaoProfessor]:
    """Lê o ``decisions.csv`` e indexa as decisões por id de mensagem.

    O lado teacher tem prioridade; ids ausentes nele usam o lado fallback
    para manter as 1000 mensagens cobertas.

    Args:
        caminho: Caminho do CSV de decisões do experimento.
        lado_teacher: Lado prioritário (``python`` = DeepSeek, 100% de acurácia).
        lado_fallback: Lado de contingência (``rust`` = Jev, 99,9% de acurácia).
        logger: Logger central.

    Returns:
        Mapa ``id_mensagem -> decisão`` já com fallback aplicado.

    Raises:
        FileNotFoundError: Se o CSV de decisões não existir.
    """
    if not caminho.is_file():
        raise FileNotFoundError(f"CSV de decisões ausente: {caminho}")
    teacher: dict[int, DecisaoProfessor] = {}
    fallback: dict[int, DecisaoProfessor] = {}
    with caminho.open(newline="", encoding="utf-8") as handle:
        for linha in csv.DictReader(handle):
            decisao = DecisaoProfessor(
                lado=str(linha["side"]),
                tipo=str(linha["tipo"]),
                urgencia=float(linha["urgencia"]),
                pede_pix=float(linha["pede_pix"]),
            )
            destino = teacher if decisao.lado == lado_teacher else fallback
            destino[int(linha["id"])] = decisao
    logger.info(
        "🏷️ Decisões carregadas: %d teacher (%s) | %d fallback (%s)",
        len(teacher),
        lado_teacher,
        len(fallback),
        lado_fallback,
    )

    decisoes: dict[int, DecisaoProfessor] = {}
    ids_fallback = sorted(set(fallback) - set(teacher))
    decisoes.update(teacher)
    for id_msg in ids_fallback:
        decisoes[id_msg] = fallback[id_msg]
    if ids_fallback:
        logger.warning("⚠️ Mensagens sem decisão do teacher; fallback aplicado nos ids: %s", ids_fallback)
    return decisoes


def construir_gold(mensagem: Mensagem, decisao: DecisaoProfessor) -> dict[str, Any]:
    """Monta o bloco ``gold`` (alvo) no formato oficial de fine-tuning do Laya.

    Args:
        mensagem: Mensagem com a verdade de referência.
        decisao: Decisão do professor para os rótulos auxiliares.

    Returns:
        Dicionário gold por pergunta com ``label`` e ``probabilities``.

    Raises:
        ValueError: Se o tipo ou o nível de urgência forem inválidos.
    """
    if decisao.tipo not in TIPOS_CRITERIA:
        raise ValueError(f"Tipo inválido na mensagem {mensagem.id}: {decisao.tipo}")
    nivel_urgencia = min(NIVEIS_URGENCIA - 1, max(0, round(decisao.urgencia)))
    prob_golpe = 1.0 if mensagem.golpe_real else 0.0
    prob_pede_pix = 1.0 if decisao.pede_pix >= LIMIAR_NOUL else 0.0
    return {
        "golpe": {
            "label": "true" if mensagem.golpe_real else "false",
            "noul": prob_golpe,
            "probabilities": {"false": 1.0 - prob_golpe, "true": prob_golpe},
        },
        "tipo": {
            "label": decisao.tipo,
            "probabilities": {tipo: 1.0 if tipo == decisao.tipo else 0.0 for tipo in TIPOS_CRITERIA},
        },
        "urgencia": {
            "label": nivel_urgencia,
            "score": float(nivel_urgencia),
            "probabilities": {str(i): 1.0 if i == nivel_urgencia else 0.0 for i in range(NIVEIS_URGENCIA)},
        },
        "pede_pix": {
            "label": "true" if prob_pede_pix == 1.0 else "false",
            "noul": prob_pede_pix,
            "probabilities": {"false": 1.0 - prob_pede_pix, "true": prob_pede_pix},
        },
    }


def construir_registro(mensagem: Mensagem, decisao: DecisaoProfessor, workflow: str) -> dict[str, str]:
    """Serializa uma linha do dataset de fine-tuning (state/questions/gold em JSON).

    O ``state`` replica o enviado ao Jev/DeepSeek no experimento original:
    ``{"remetente": ..., "mensagem": ...}``.

    Args:
        mensagem: Mensagem fonte.
        decisao: Decisão do professor.
        workflow: Nome do domínio do dataset.

    Returns:
        Linha pronta para o JSONL de treino/validação.
    """
    state = {"remetente": mensagem.remetente, "mensagem": mensagem.texto}
    gold = construir_gold(mensagem, decisao)
    return {
        "id": str(mensagem.id),
        "workflow": workflow,
        "state": json.dumps(state, ensure_ascii=False),
        "questions": json.dumps(QUESTIONS, ensure_ascii=False),
        "gold": json.dumps(gold, ensure_ascii=False),
    }


def dividir_estratificado(
    registros: list[dict[str, str]],
    mensagens_por_id: dict[int, Mensagem],
    proporcao_treino: float,
    semente: int,
    logger: logging.Logger,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Divide treino/validação com estratificação por ``golpe_real``.

    Args:
        registros: Linhas já construídas do dataset.
        mensagens_por_id: Mapa id -> mensagem (fonte da classe de estratificação).
        proporcao_treino: Fração destinada ao treino.
        semente: Semente do embaralhamento reproduzível.
        logger: Logger central.

    Returns:
        Tupla ``(treino, validacao)`` embaralhada.
    """
    golpes = [r for r in registros if mensagens_por_id[int(r["id"])].golpe_real]
    legitimas = [r for r in registros if not mensagens_por_id[int(r["id"])].golpe_real]
    sorteador = random.Random(semente)
    sorteador.shuffle(golpes)
    sorteador.shuffle(legitimas)

    corte_golpes = round(len(golpes) * proporcao_treino)
    corte_legitimas = round(len(legitimas) * proporcao_treino)
    treino = golpes[:corte_golpes] + legitimas[:corte_legitimas]
    validacao = golpes[corte_golpes:] + legitimas[corte_legitimas:]
    sorteador.shuffle(treino)
    sorteador.shuffle(validacao)
    logger.info(
        "✂️ Split estratificado (semente %d): treino=%d (%d golpes) | validação=%d (%d golpes)",
        semente,
        len(treino),
        corte_golpes,
        len(validacao),
        len(golpes) - corte_golpes,
    )
    return treino, validacao


def salvar_jsonl(registros: list[dict[str, str]], caminho: Path, logger: logging.Logger) -> None:
    """Grava uma lista de linhas em JSONL UTF-8.

    Args:
        registros: Linhas do dataset.
        caminho: Arquivo de destino.
        logger: Logger central.
    """
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", encoding="utf-8") as handle:
        for registro in registros:
            handle.write(json.dumps(registro, ensure_ascii=False) + "\n")
    logger.info("💾 Dataset gravado: %s (%d linhas)", caminho, len(registros))


def salvar_resumo(
    treino: list[dict[str, str]],
    validacao: list[dict[str, str]],
    mensagens_por_id: dict[int, Mensagem],
    caminho: Path,
    logger: logging.Logger,
) -> None:
    """Gera o resumo estatístico do dataset produzido.

    Args:
        treino: Linhas de treino.
        validacao: Linhas de validação.
        mensagens_por_id: Mapa id -> mensagem.
        caminho: Arquivo JSON de saída.
        logger: Logger central.
    """
    distribuicao_tipo: dict[str, int] = {tipo: 0 for tipo in TIPOS_CRITERIA}
    for registro in treino + validacao:
        tipo = str(json.loads(registro["gold"])["tipo"]["label"])
        distribuicao_tipo[tipo] += 1
    resumo: dict[str, Any] = {
        "gerado_em": datetime.now(tz=FUSO_BRASILIA).isoformat(),
        "total_mensagens": len(treino) + len(validacao),
        "treino": {
            "n": len(treino),
            "golpes": sum(1 for r in treino if mensagens_por_id[int(r["id"])].golpe_real),
        },
        "validacao": {
            "n": len(validacao),
            "golpes": sum(1 for r in validacao if mensagens_por_id[int(r["id"])].golpe_real),
        },
        "distribuicao_tipo": distribuicao_tipo,
        "perguntas_tipadas": list(QUESTIONS.keys()),
    }
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("📊 Resumo do dataset: %s", caminho)
    logger.info("📊 Distribuição das 8 classes de tipo: %s", distribuicao_tipo)


def main() -> int:
    """Ponto de entrada do pipeline de preparação do dataset.

    Returns:
        Código de saída do processo (0 = sucesso).
    """
    colorama_init(autoreset=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/laya_config.json", help="caminho do JSON de configuração")
    args = parser.parse_args()
    raiz = Path(__file__).resolve().parent.parent

    logger = configurar_logging(raiz / "logs", "preparar_dataset")
    logger.info("🚀 Iniciando preparação do dataset Laya | projeto gmr_pix_golpe")

    try:
        config = carregar_config(raiz / args.config, logger)
        paths: dict[str, str] = config["paths"]
        cfg_dataset: dict[str, Any] = config["dataset"]

        mensagens = carregar_mensagens(raiz / paths["mensagens_raw"], logger)
        decisoes = carregar_decisoes(
            raiz / paths["decisoes_raw"],
            lado_teacher=str(cfg_dataset["lado_teacher"]),
            lado_fallback=str(cfg_dataset["lado_fallback"]),
            logger=logger,
        )
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
        logger.error("❌ Falha na carga dos insumos: %s", exc)
        return 1

    ausentes = [m.id for m in mensagens if m.id not in decisoes]
    if ausentes:
        logger.error("❌ Mensagens sem nenhuma decisão (teacher ou fallback): %s", ausentes)
        return 1

    registros = [construir_registro(m, decisoes[m.id], str(cfg_dataset["workflow"])) for m in mensagens]
    mensagens_por_id = {m.id: m for m in mensagens}
    logger.info("🏗️ Registros construídos: %d (state + 4 perguntas tipadas + gold)", len(registros))

    treino, validacao = dividir_estratificado(
        registros,
        mensagens_por_id,
        proporcao_treino=float(cfg_dataset["proporcao_treino"]),
        semente=int(cfg_dataset["semente"]),
        logger=logger,
    )
    salvar_jsonl(treino, raiz / paths["arquivo_treino"], logger)
    salvar_jsonl(validacao, raiz / paths["arquivo_validacao"], logger)
    salvar_resumo(treino, validacao, mensagens_por_id, raiz / paths["arquivo_resumo"], logger)

    logger.info(
        "✅ Dataset Laya pronto para upload no Colab: %s e %s",
        paths["arquivo_treino"],
        paths["arquivo_validacao"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
