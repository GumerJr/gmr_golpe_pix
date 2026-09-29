"""Inferência local (CPU) com o Laya fine-tunado no domínio pix_golpe.

Carrega o checkpoint treinado no Colab (``models/laya_pix_golpe_finetuned/``)
e classifica mensagens com as 4 perguntas tipadas do experimento Pix Race,
aplicando o veredito em 3 faixas do protocolo original.

Uso::

    # mensagem única via argumentos
    .venv/bin/python src/inferir.py --remetente "Banco X" --mensagem "Sua conta foi bloqueada..."

    # modo interativo (digite 'sair' para encerrar)
    .venv/bin/python src/inferir.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from colorama import Fore, Style
from colorama import init as colorama_init

from laya_comum import NIVEIS_URGENCIA, QUESTIONS, AgenteSystemOne, carregar_agente, configurar_logging, veredito

CORES_VEREDITO: dict[str, str] = {"golpe": Fore.RED, "revisar": Fore.YELLOW, "ok": Fore.GREEN}


def carregar_config(raiz: Path) -> dict[str, Any]:
    """Carrega a configuração central do projeto.

    Args:
        raiz: Raiz do projeto.

    Returns:
        Dicionário de configuração.

    Raises:
        FileNotFoundError: Se ``config/laya_config.json`` não existir.
    """
    caminho = raiz / "config" / "laya_config.json"
    if not caminho.is_file():
        raise FileNotFoundError(f"Configuração ausente: {caminho}")
    config: dict[str, Any] = json.loads(caminho.read_text(encoding="utf-8"))
    return config


def classificar(agente: AgenteSystemOne, remetente: str, mensagem: str) -> dict[str, Any]:
    """Executa a decisão tipada das 4 perguntas em uma única passada.

    Args:
        agente: Agente Laya carregado.
        remetente: Remetente da mensagem.
        mensagem: Texto da mensagem.

    Returns:
        Dicionário com as 4 respostas, o veredito e a latência em ms.
    """
    state = {"remetente": remetente, "mensagem": mensagem}
    inicio = time.perf_counter()
    resultado = agente.predict(state, QUESTIONS)
    latencia_ms = (time.perf_counter() - inicio) * 1000
    answers: dict[str, Any] = resultado["answers"]
    return {
        "answers": answers,
        "prob_golpe": float(answers["golpe"]["noul"]),
        "veredito": veredito(float(answers["golpe"]["noul"])),
        "latencia_ms": latencia_ms,
    }


def imprimir_decisao(remetente: str, mensagem: str, decisao: dict[str, Any], logger: logging.Logger) -> None:
    """Apresenta a decisão tipada completa com cores e emojis no console.

    Args:
        remetente: Remetente avaliado.
        mensagem: Texto avaliado.
        decisao: Saída de :func:`classificar`.
        logger: Logger central (o veredito também vai para o arquivo de log).
    """
    answers = decisao["answers"]
    probs_tipo: dict[str, float] = answers["tipo"]["probabilities"]
    probs_urgencia: dict[str, float] = {
        str(i): float(answers["urgencia"]["probabilities"].get(str(i), 0.0)) for i in range(NIVEIS_URGENCIA)
    }

    print(f"\n{'=' * 64}")
    print(f"📨 Remetente : {remetente}")
    print(f"💬 Mensagem  : {mensagem}")
    print(f"{'-' * 64}")
    for qid in ("golpe", "tipo", "urgencia", "pede_pix"):
        resposta = answers[qid]
        if resposta.get("noul") is not None:
            print(f"   {qid:<10}: p={resposta['noul']:.3f}")
        elif resposta.get("choice") is not None:
            print(f"   {qid:<10}: {resposta['choice']:<18} (confiança {resposta['confidence']:.3f})")
        else:
            print(f"   {qid:<10}: score={resposta['score']:.2f} | níveis {probs_urgencia}")
    print(f"{'-' * 64}")
    cor = CORES_VEREDITO[decisao["veredito"]]
    emoji = {"golpe": "🚨", "revisar": "⚠️", "ok": "✅"}[decisao["veredito"]]
    print(
        f"{cor}{emoji} VEREDITO: {decisao['veredito'].upper()} (p_golpe={decisao['prob_golpe']:.3f}){Style.RESET_ALL}"
    )
    print(f"⏱️ Latência: {decisao['latencia_ms']:.0f} ms | tipo pmax={max(probs_tipo.values()):.3f}")
    logger.info(
        "🔎 Decisão registrada | remetente=%r | veredito=%s | p_golpe=%.3f | tipo=%s | latencia=%.0fms",
        remetente,
        decisao["veredito"],
        decisao["prob_golpe"],
        answers["tipo"]["choice"],
        decisao["latencia_ms"],
    )


def modo_interativo(agente: AgenteSystemOne, logger: logging.Logger) -> None:
    """Loop de classificação interativa no terminal.

    Args:
        agente: Agente Laya carregado.
        logger: Logger central.
    """
    print("\n🛡️ Modo interativo — detecção de golpe do Pix (digite 'sair' para encerrar)")
    while True:
        remetente = input("\n👤 Remetente: ").strip()
        if remetente.lower() == "sair":
            break
        mensagem = input("💬 Mensagem: ").strip()
        if mensagem.lower() == "sair":
            break
        if not remetente or not mensagem:
            print("⚠️ Remetente e mensagem são obrigatórios.")
            continue
        imprimir_decisao(remetente, mensagem, classificar(agente, remetente, mensagem), logger)
    logger.info("👋 Sessão interativa encerrada")


def main() -> int:
    """Ponto de entrada da inferência local.

    Returns:
        Código de saída do processo (0 = sucesso).
    """
    colorama_init(autoreset=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remetente", help="remetente da mensagem a classificar")
    parser.add_argument("--mensagem", help="texto da mensagem a classificar")
    args = parser.parse_args()
    raiz = Path(__file__).resolve().parent.parent

    logger = configurar_logging(raiz / "logs", "inferir")
    try:
        config = carregar_config(raiz)
        agente = carregar_agente(raiz / config["paths"]["diretorio_modelo_finetunado"], logger)
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
        logger.error("❌ %s", exc)
        return 1

    if args.remetente and args.mensagem:
        imprimir_decisao(args.remetente, args.mensagem, classificar(agente, args.remetente, args.mensagem), logger)
    else:
        modo_interativo(agente, logger)
    return 0


if __name__ == "__main__":
    sys.exit(main())
