"""Componentes compartilhados do projeto gmr_pix_golpe.

Centraliza o fuso horário oficial de Brasília (UTC-3), o formatter de logging
colorido e o schema das 4 perguntas tipadas do domínio ``pix_golpe`` —
réplica exata de ``questions()`` em ``pix-golpe/src/jev.rs`` do repositório
sandeco/pix-golpe.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias
from zoneinfo import ZoneInfo

from colorama import Fore, Style

if TYPE_CHECKING:
    import laya
    from laya.onnx_agent import ONNXAgent

AgenteSystemOne: TypeAlias = "laya.Agent | ONNXAgent"
"""Agente de decisão tipada: PyTorch (``laya.Agent``) ou ONNX INT8 (``ONNXAgent``)."""

FUSO_BRASILIA: ZoneInfo = ZoneInfo("America/Sao_Paulo")
"""Horário oficial de Brasília (UTC-3) para todo o logging do projeto."""

QUESTIONS: dict[str, dict[str, Any]] = {
    "golpe": {
        "type": "noul",
        "instructions": "This message is a scam or fraud attempt (impersonation, fake bill, "
        "fake bank alert, fake prize, pressure to pay)",
    },
    "tipo": {
        "type": "choice",
        "instructions": "What kind of message is this",
        "criteria": {
            "troca_numero": "Someone claims to be a relative with a new phone number asking for money",
            "boleto_falso": "Fake bill or invoice to be paid",
            "falsa_central": "Fake bank or support center alert about the account",
            "premio": "Fake prize, lottery or giveaway",
            "pix_errado": "Claims a Pix was sent by mistake and asks for a refund",
            "cobranca_legitima": "Legitimate charge, bill or order confirmation",
            "pessoal": "Ordinary personal message between people who know each other",
            "outro": "Something else",
        },
    },
    "urgencia": {
        "type": "score",
        "instructions": "How much time pressure or threat the message applies",
        "criteria": ["No time pressure", "Some urgency", "Extreme urgency, threats or deadlines"],
    },
    "pede_pix": {
        "type": "noul",
        "instructions": "The message asks the reader to make a Pix transfer or payment",
    },
}
"""Perguntas tipadas idênticas a ``pix-golpe/src/jev.rs::questions()``."""

TIPOS_CRITERIA: tuple[str, ...] = tuple(QUESTIONS["tipo"]["criteria"].keys())
NIVEIS_URGENCIA: int = len(QUESTIONS["urgencia"]["criteria"])
LIMIAR_NOUL: float = 0.5
LIMIAR_GOLPE: float = 0.6
LIMIAR_OK: float = 0.4


def veredito(prob_golpe: float) -> str:
    """Aplica o veredito em 3 faixas do experimento Pix Race.

    Args:
        prob_golpe: Probabilidade de golpe retornada pela pergunta ``noul``.

    Returns:
        ``"golpe"`` (p >= 0.6), ``"ok"`` (p <= 0.4) ou ``"revisar"``.
    """
    if prob_golpe >= LIMIAR_GOLPE:
        return "golpe"
    if prob_golpe <= LIMIAR_OK:
        return "ok"
    return "revisar"


def carregar_agente(diretorio_modelo: Path, logger: logging.Logger) -> AgenteSystemOne:
    """Carrega o checkpoint fine-tunado local, preferindo a exportação ONNX (fp32).

    A variante ``laya.onnx`` (gerada pela célula 7 do notebook Colab) roda com
    ``onnxruntime``, sem alocar os pesos em PyTorch — ideal para CPUs sem GPU.
    A exportação é propositalmente **fp32**: a quantização dinâmica INT8 achata
    os logits da head fine-tunada neste checkpoint e degrada as decisões para
    p≈0,5 (medido na validação local de 29/09/2026). Sem ONNX, cai no
    ``laya.Agent`` PyTorch (fp16, mais pesado em RAM).

    Args:
        diretorio_modelo: Pasta com ``model.safetensors``, ``encoder/``,
            ``tokenizer/``, ``rl_agent_config.json`` e, idealmente, ``laya.onnx``.
        logger: Logger central.

    Returns:
        Agente Laya (ONNX ou PyTorch) pronto para ``predict``.

    Raises:
        FileNotFoundError: Se o checkpoint estiver incompleto ou ausente.
    """
    esperados = ["model.safetensors", "rl_agent_config.json", "encoder", "tokenizer"]
    if not diretorio_modelo.is_dir() or not all((diretorio_modelo / e).exists() for e in esperados):
        raise FileNotFoundError(
            f"Checkpoint fine-tunado incompleto ou ausente em {diretorio_modelo}. "
            "Baixe o .zip da célula 8 do notebook no Colab e descompacte nesta pasta."
        )

    onnx_path = diretorio_modelo / "laya.onnx"
    if onnx_path.is_file():
        from laya.onnx_agent import ONNXAgent

        logger.info("🧊 Carregando checkpoint ONNX fp32 de %s (onnxruntime, CPU)...", onnx_path)
        agente = ONNXAgent(str(diretorio_modelo), onnx_path=str(onnx_path))
        logger.info("✅ Modelo ONNX carregado")
        return agente

    import laya

    logger.warning(
        "⚠️ laya.onnx não encontrado — caindo no PyTorch fp16 (exige ~2 GB de RAM livres). "
        "Recomendado: rodar a célula 7 do notebook Colab para exportar o ONNX fp32."
    )
    logger.info("🧠 Carregando checkpoint PyTorch de %s (CPU)...", diretorio_modelo)
    agente = laya.Agent(str(diretorio_modelo), device="cpu")
    logger.info("✅ Modelo PyTorch carregado")
    return agente


class FormatadorBrasilia(logging.Formatter):
    """Formatter com timestamp no fuso de Brasília e cores via colorama."""

    CORES: dict[int, str] = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA,
    }

    def __init__(self: "FormatadorBrasilia", *, colorido: bool) -> None:
        """Inicializa o formatter indicando se a saída recebe cores ANSI.

        Args:
            colorido: ``True`` para colorir pelo nível do log (console);
                ``False`` para saída pura (arquivo de log).
        """
        super().__init__("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        self.colorido: bool = colorido

    def formatTime(self: "FormatadorBrasilia", record: logging.LogRecord, datefmt: str | None = None) -> str:
        """Formata o instante do registro no fuso oficial de Brasília (UTC-3).

        Args:
            record: Registro de log original.
            datefmt: Formato ``strftime`` opcional.

        Returns:
            Timestamp formatado em ``America/Sao_Paulo``.
        """
        instante = datetime.fromtimestamp(record.created, tz=FUSO_BRASILIA)
        if datefmt:
            return instante.strftime(datefmt)
        return instante.isoformat()

    def format(self: "FormatadorBrasilia", record: logging.LogRecord) -> str:
        """Aplica cor conforme o nível quando o handler for o console.

        Args:
            record: Registro de log original.

        Returns:
            Linha de log formatada, com ou sem ANSI.
        """
        texto = super().format(record)
        if self.colorido:
            cor = self.CORES.get(record.levelno, "")
            return f"{cor}{texto}{Style.RESET_ALL}"
        return texto


def configurar_logging(diretorio_logs: Path, nome_app: str) -> logging.Logger:
    """Configura o logger central: console colorido + arquivo em disco.

    Args:
        diretorio_logs: Pasta onde o arquivo ``.log`` será criado.
        nome_app: Nome do aplicativo, usado no nome do arquivo de log.

    Returns:
        Logger raiz do projeto configurado.
    """
    diretorio_logs.mkdir(parents=True, exist_ok=True)
    agora = datetime.now(tz=FUSO_BRASILIA).strftime("%Y%m%d_%H%M%S")
    arquivo_log = diretorio_logs / f"{nome_app}_{agora}.log"

    logger = logging.getLogger("gmr_pix_golpe")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(FormatadorBrasilia(colorido=True))

    arquivo = logging.FileHandler(arquivo_log, encoding="utf-8")
    arquivo.setLevel(logging.DEBUG)
    arquivo.setFormatter(FormatadorBrasilia(colorido=False))

    logger.addHandler(console)
    logger.addHandler(arquivo)
    logger.info("📝 Log centralizado ativo | arquivo: %s", arquivo_log)
    return logger
