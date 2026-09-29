"""Servidor web da demo Pix Race com o Laya fine-tunado (local, CPU).

Sobe uma página em ``http://localhost:8080`` que percorre as 1000 mensagens de
``data/raw/mensagens.jsonl`` (repositório sandeco/pix-golpe), classifica cada
uma com o checkpoint fine-tunado em ``models/laya_pix_golpe_finetuned/`` via
WebSocket e exibe o veredito em 3 faixas do protocolo original, com relatório
de acurácia (matriz de confusão, precisão e recall) ao final ou ao parar.

Uso::

    .venv/bin/python src/servidor_web.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import uvicorn
from colorama import init as colorama_init
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from laya_comum import QUESTIONS, AgenteSystemOne, carregar_agente, configurar_logging, veredito

RAIZ = Path(__file__).resolve().parent.parent
VEL_MIN, VEL_MAX = 0.0, 10.0


@dataclass(frozen=True)
class Mensagem:
    """Mensagem do dataset Pix Race com a verdade de referência."""

    id: int
    remetente: str
    texto: str
    golpe_real: bool


@dataclass
class Placar:
    """Placar acumulado da corrida no protocolo de veredito em 3 faixas."""

    processadas: int = 0
    golpes_detectados: int = 0
    erros: int = 0
    latencias: list[float] = field(default_factory=list)
    matriz: dict[str, dict[str, int]] = field(
        default_factory=lambda: {real: {pred: 0 for pred in ("golpe", "ok", "revisar")} for real in ("golpe", "normal")}
    )

    def registrar(self: "Placar", golpe_real: bool, veredito_pred: str, latencia_ms: float) -> None:
        """Acumula uma decisão no placar.

        Args:
            golpe_real: Verdade de referência da mensagem.
            veredito_pred: Veredito do modelo (``golpe``/``ok``/``revisar``).
            latencia_ms: Latência da decisão em milissegundos.
        """
        self.processadas += 1
        self.latencias.append(latencia_ms)
        real = "golpe" if golpe_real else "normal"
        self.matriz[real][veredito_pred] += 1
        if veredito_pred == "golpe":
            self.golpes_detectados += 1
        if (veredito_pred == "golpe") != golpe_real:
            self.erros += 1

    def relatorio(self: "Placar") -> dict[str, Any]:
        """Consolida o relatório final seguindo o protocolo do paper.

        Na classe binária só ``golpe`` conta como positivo; ``revisar`` conta
        como negativo (golpe perdido quando a mensagem era golpe).

        Returns:
            Dicionário com matriz, acurácia, precisão, recall e latências.
        """
        tp = self.matriz["golpe"]["golpe"]
        fn = self.matriz["golpe"]["ok"] + self.matriz["golpe"]["revisar"]
        fp = self.matriz["normal"]["golpe"]
        tn = self.matriz["normal"]["ok"] + self.matriz["normal"]["revisar"]
        total = tp + fn + fp + tn
        return {
            "total": total,
            "matriz": self.matriz,
            "golpes_reais": sum(self.matriz["golpe"].values()),
            "golpes_detectados": self.golpes_detectados,
            "acuracia": (tp + tn) / total if total else 0.0,
            "precisao": tp / (tp + fp) if (tp + fp) else 0.0,
            "recall": tp / (tp + fn) if (tp + fn) else 0.0,
            "latencia_media_ms": sum(self.latencias) / len(self.latencias) if self.latencias else 0.0,
            "latencia_p95_ms": sorted(self.latencias)[int(0.95 * (len(self.latencias) - 1))] if self.latencias else 0.0,
        }


def carregar_mensagens(caminho: Path, logger: logging.Logger) -> list[Mensagem]:
    """Lê as mensagens do dataset Pix Race.

    Args:
        caminho: Caminho do ``mensagens.jsonl``.
        logger: Logger central.

    Returns:
        Lista de mensagens tipadas.

    Raises:
        FileNotFoundError: Se o dataset não existir.
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
                    remetente=str(bruto["remetente"]),
                    texto=str(bruto["texto"]),
                    golpe_real=bool(bruto["golpe_real"]),
                )
            )
    logger.info("📥 %d mensagens carregadas (%d golpes)", len(mensagens), sum(1 for m in mensagens if m.golpe_real))
    return mensagens


def classificar(agente: AgenteSystemOne, mensagem: Mensagem) -> dict[str, Any]:
    """Executa as 4 perguntas tipadas em uma única passada do Laya.

    Args:
        agente: Agente Laya carregado.
        mensagem: Mensagem a classificar.

    Returns:
        Evento de decisão pronto para serializar no WebSocket.
    """
    state = {"remetente": mensagem.remetente, "mensagem": mensagem.texto}
    inicio = time.perf_counter()
    resultado = agente.predict(state, QUESTIONS)
    latencia_ms = (time.perf_counter() - inicio) * 1000
    answers: dict[str, Any] = resultado["answers"]
    prob_golpe = float(answers["golpe"]["noul"])
    return {
        "type": "decisao",
        "id": mensagem.id,
        "remetente": mensagem.remetente,
        "texto": mensagem.texto,
        "golpe_real": mensagem.golpe_real,
        "prob_golpe": round(prob_golpe, 3),
        "veredito": veredito(prob_golpe),
        "tipo": str(answers["tipo"]["choice"]),
        "tipo_confianca": round(float(answers["tipo"]["confidence"]), 3),
        "urgencia": round(float(answers["urgencia"]["score"]), 2),
        "pede_pix": round(float(answers["pede_pix"]["noul"]), 3),
        "latencia_ms": round(latencia_ms, 1),
    }


async def loop_corrida(
    ws: WebSocket,
    agente: AgenteSystemOne,
    mensagens: list[Mensagem],
    intervalo: float,
    limite: int,
    parar: asyncio.Event,
    logger: logging.Logger,
) -> None:
    """Percorre as mensagens, classifica e transmite cada decisão.

    Args:
        ws: Conexão WebSocket ativa.
        agente: Agente Laya carregado.
        mensagens: Mensagens do dataset em ordem.
        intervalo: Pausa em segundos entre mensagens (ritmo narrado/normal).
        limite: Máximo de mensagens (0 = todas).
        parar: Evento de parada acionado pelo botão Parar.
        logger: Logger central.
    """
    placar = Placar()
    selecionadas = mensagens[:limite] if limite > 0 else mensagens
    await ws.send_json({"type": "iniciado", "total": len(selecionadas)})
    logger.info("🏁 Corrida iniciada: %d mensagens | pausa %.1fs", len(selecionadas), intervalo)

    interrompida = False
    for i, mensagem in enumerate(selecionadas, start=1):
        if parar.is_set():
            interrompida = True
            break
        decisao = await asyncio.to_thread(classificar, agente, mensagem)
        placar.registrar(mensagem.golpe_real, str(decisao["veredito"]), float(decisao["latencia_ms"]))
        decisao["i"] = i
        decisao["placar"] = {
            "processadas": placar.processadas,
            "golpes_detectados": placar.golpes_detectados,
            "erros": placar.erros,
            "latencia_media_ms": round(placar.relatorio()["latencia_media_ms"], 1),
        }
        await ws.send_json(decisao)
        if intervalo > 0:
            try:
                await asyncio.wait_for(parar.wait(), timeout=intervalo)
                interrompida = True
                break
            except TimeoutError:
                pass

    motivo = "parada pelo usuário" if interrompida else "concluída"
    await ws.send_json({"type": "fim", "motivo": motivo, "relatorio": placar.relatorio()})
    logger.info(
        "🏁 Corrida %s: %d mensagens | %d golpes detectados | acurácia %.3f",
        motivo,
        placar.processadas,
        placar.golpes_detectados,
        placar.relatorio()["acuracia"],
    )


async def atender_ws(ws: WebSocket, app_state: dict[str, Any], logger: logging.Logger) -> None:
    """Gerencia uma conexão WebSocket: comandos e corrida concorrentes.

    Args:
        ws: Conexão aceita.
        app_state: Estado compartilhado (agente e mensagens).
        logger: Logger central.
    """
    await ws.accept()
    parar = asyncio.Event()
    corrida: asyncio.Task[None] | None = None
    logger.info("🔌 Cliente conectado")
    try:
        while True:
            comando = json.loads(await ws.receive_text())
            cmd = str(comando.get("cmd", ""))
            if cmd == "iniciar" and corrida is None:
                parar.clear()
                intervalo = min(VEL_MAX, max(VEL_MIN, float(comando.get("intervalo", 0.0))))
                limite = int(comando.get("limite", 0))
                corrida = asyncio.create_task(
                    loop_corrida(ws, app_state["agente"], app_state["mensagens"], intervalo, limite, parar, logger)
                )
                corrida.add_done_callback(lambda _t: None)
            elif cmd == "parar":
                parar.set()
            else:
                await ws.send_json({"type": "aviso", "detalhe": f"comando ignorado: {cmd!r}"})
                if corrida is not None and corrida.done():
                    corrida = None
    except WebSocketDisconnect:
        logger.info("🔌 Cliente desconectado")
    finally:
        parar.set()
        if corrida is not None:
            corrida.cancel()


def criar_app(app_state: dict[str, Any], logger: logging.Logger) -> FastAPI:
    """Constrói a aplicação FastAPI com a página e o WebSocket.

    Args:
        app_state: Estado compartilhado (agente e mensagens).
        logger: Logger central.

    Returns:
        Aplicação configurada.
    """
    app = FastAPI(title="gmr_pix_golpe — demo Laya fine-tuned")
    pagina = (RAIZ / "static" / "index.html").read_text(encoding="utf-8")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        """Serve a página da demo."""
        return HTMLResponse(pagina)

    @app.get("/api/saude")
    async def saude() -> dict[str, Any]:
        """Health check simples com o estado do modelo e do dataset."""
        return {"status": "ok", "mensagens": len(app_state["mensagens"])}

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        """Endpoint WebSocket da corrida de classificação."""
        await atender_ws(ws, app_state, logger)

    return app


def main() -> int:
    """Ponto de entrada: carrega insumos e sobe o servidor HTTP.

    Returns:
        Código de saída do processo (0 = sucesso).
    """
    colorama_init(autoreset=True)
    logger = configurar_logging(RAIZ / "logs", "servidor_web")
    try:
        config: dict[str, Any] = json.loads((RAIZ / "config" / "laya_config.json").read_text(encoding="utf-8"))
        mensagens = carregar_mensagens(RAIZ / config["paths"]["mensagens_raw"], logger)
        agente = carregar_agente(RAIZ / config["paths"]["diretorio_modelo_finetunado"], logger)
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
        logger.error("❌ %s", exc)
        return 1

    app = criar_app({"agente": agente, "mensagens": mensagens}, logger)
    logger.info("🌐 Demo disponível em http://localhost:8080")
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
