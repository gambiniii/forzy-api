"""
Cria 7 motores ESTÁTICOS de demonstração, um por condição da tabela
"Resultado verificado" de docs/resposta-pendencias-ml-fase2.md (6 modos de
falha injetados + controle saudável) — prontos pra abrir direto na
apresentação, sem depender de nenhum processo ao vivo (sem poller, sem modo
demo, sem scheduler rodando pra "pegar o estado certo").

NÃO TOCA nos motores existentes (componente_id 1, 2, 3). Aloca os novos na
planta FIAP (id=2, já existente). NÃO cria leitura_sensor — só Máquina +
Componente + Especificação + um Diagnóstico já resolvido, com o
overall_status/threshold_status/breached_metrics corretos pra cada condição
já aparecerem prontos (Hero, KPIs, destaque 3D quando a condição tiver
segmento associado).

breached_metrics usa as mesmas 3 chaves de motorSegmentMap.ts (frontend):
velocidade/temperatura/aceleracao — cobre exatamente os 3 modos cuja
atribuição é determinística (Desbalanceamento, Sobreaquecimento, Rolamento).
Os 3 modos de instrumentação/parada (Sensor travado, Deriva, Perda de
acionamento) não tinham segmento algum na bancada de validação também
("nenhuma" na tabela) — Perda de acionamento é a única exceção documentada
(acende caixa de ligação + tampa traseira via atribuição de ML), mas isso
depende do endpoint /atribuicao rodando sobre leitura_sensor real, que este
script deliberadamente não cria — por isso fica sem destaque aqui também.

Uso: python scripts/seed_motores_demo_condicoes.py
Idempotente — não duplica se a máquina já existir (por nome).
"""
import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.db.postgres import SessionLocal, create_tables
from src.models.componente import Componente, EspecificacaoMotor
from src.models.diagnostico import Diagnostico
from src.models.enums import StatusEnum
from src.models.maquina import Maquina
from src.models.planta import Planta

PLANTA_FIAP_NOME = "FIAP"

ESPECIFICACAO_W22 = dict(
    potencia_kw=2.2, tensao_nominal=220, corrente_nominal=12,
    rpm_nominal=3525, frequencia_hz=60, numero_polos=2, rendimento=82,
)

# Cada condição = (nome da máquina, kwargs do Diagnostico)
CONDICOES: list[tuple[str, dict]] = [
    ("Motor Demo — Desbalanceamento do Rotor", dict(
        overall_status="critical", is_anomaly=True, lstm_severity="critical", risk_level="high",
        health_score=0.15, health_index=15.0, rul_hours=60.0, maintenance_window_days=1.0,
        recommendation="Desbalanceamento do rotor detectado — velocidade de vibração muito acima do "
                        "normal, concentrada em 1× a rotação, com a aceleração subindo bem menos que a "
                        "velocidade. Agendar balanceamento imediato.",
        confidence=95.0, threshold_status="critico",
        threshold_message="Velocidade em 14,70 mm/s, acima do limite crítico de 7,69 mm/s.",
        breached_metrics="velocidade",
    )),
    ("Motor Demo — Sobreaquecimento", dict(
        overall_status="warning", is_anomaly=True, lstm_severity="medium", risk_level="medium",
        health_score=0.55, health_index=55.0, rul_hours=300.0, maintenance_window_days=15.0,
        recommendation="Temperatura sustentada acima do normal com a vibração praticamente inalterada — "
                        "padrão típico de ventilação obstruída ou sobrecarga elétrica, não de falha "
                        "mecânica. Verificar entradas/saídas de ar e a corrente do motor.",
        confidence=88.0, threshold_status="atencao",
        threshold_message="Temperatura em 56,0°C, acima do limite de atenção de 45,0°C.",
        breached_metrics="temperatura",
    )),
    ("Motor Demo — Degradação de Rolamento", dict(
        overall_status="critical", is_anomaly=True, lstm_severity="critical", risk_level="high",
        health_score=0.20, health_index=20.0, rul_hours=90.0, maintenance_window_days=3.0,
        recommendation="Fator de crest elevado indica defeito de pista em rolamento — impactos curtos de "
                        "alta energia precedendo o aumento do valor eficaz (indicador precoce). "
                        "Inspecionar mancais, ruído e condição do lubrificante nas duas tampas.",
        confidence=91.0, threshold_status="critico",
        threshold_message="Aceleração em 0,98 g, acima do limite crítico de 0,794 g.",
        breached_metrics="aceleracao",
    )),
    ("Motor Demo — Sensor Travado (Falha de Instrumentação)", dict(
        overall_status="retido", is_anomaly=False, lstm_severity=None, risk_level=None,
        health_score=None, health_index=None, rul_hours=None, maintenance_window_days=None,
        recommendation="Diagnóstico retido — sinal do sensor aparenta estar congelado em valores "
                        "nominais (sem variação natural entre leituras). É falha de instrumentação, não "
                        "do motor. Verificar conexão e alimentação do sensor antes de confiar na leitura.",
        confidence=None, threshold_status="retido",
        threshold_message="Sensor com leitura repetida — instrumentação suspeita.",
        breached_metrics=None,
    )),
    ("Motor Demo — Deriva de Calibração do Sensor", dict(
        overall_status="retido", is_anomaly=False, lstm_severity=None, risk_level=None,
        health_score=None, health_index=None, rul_hours=None, maintenance_window_days=None,
        recommendation="Diagnóstico retido — o ganho do sensor aparenta ter derivado (valores plausíveis, "
                        "porém deslocados de forma consistente). É falha de calibração do instrumento, "
                        "não do motor. Recalibrar o sensor antes de confiar na leitura.",
        confidence=None, threshold_status="retido",
        threshold_message="Padrão de deriva de calibração detectado no sensor.",
        breached_metrics=None,
    )),
    ("Motor Demo — Perda de Acionamento", dict(
        overall_status="motor_desligado", is_anomaly=False, lstm_severity=None, risk_level=None,
        health_score=None, health_index=None, rul_hours=None, maintenance_window_days=None,
        recommendation="Perda de acionamento detectada — os níveis caíram a valores de motor desligado "
                        "sem um comando de parada correspondente. Verificar acoplamento, contator e "
                        "alimentação antes de assumir que é uma parada programada.",
        confidence=None, threshold_status=None, threshold_message=None,
        breached_metrics=None,
    )),
    ("Motor Demo — Controle Saudável", dict(
        overall_status="healthy", is_anomaly=False, lstm_severity="normal", risk_level="low",
        health_score=0.96, health_index=96.0, rul_hours=4500.0, maintenance_window_days=200.0,
        recommendation="Motor operando dentro dos parâmetros esperados — nenhuma anomalia detectada.",
        confidence=97.0, threshold_status="nominal",
        threshold_message="Todos os parâmetros dentro dos limites contratuais.",
        breached_metrics=None,
    )),
]


def _planta_fiap(db) -> Planta:
    planta = db.query(Planta).filter(Planta.nome == PLANTA_FIAP_NOME).first()
    if not planta:
        raise RuntimeError(f"Planta '{PLANTA_FIAP_NOME}' não encontrada — crie-a antes de rodar este script.")
    return planta


def seed_condicao(db, nome_maquina: str, diagnostico_kwargs: dict) -> None:
    if db.query(Maquina).filter(Maquina.nome == nome_maquina).first():
        print(f"  '{nome_maquina}' já existe — pulando.")
        return

    planta = _planta_fiap(db)

    maquina = Maquina(
        nome=nome_maquina, tipo="Motor Elétrico (demonstração estática)", fabricante="WEG",
        ano_instalacao=2024, status=StatusEnum.active, planta_id=planta.id,
    )
    db.add(maquina)
    db.flush()

    componente = Componente(
        maquina_id=maquina.id, nome=f"{nome_maquina} - Conjunto", tipo="Motor Elétrico",
        fabricante="WEG", data_instalacao=date(2024, 1, 10), status=StatusEnum.active,
    )
    db.add(componente)
    db.flush()

    db.add(EspecificacaoMotor(componente_id=componente.id, **ESPECIFICACAO_W22))

    db.add(Diagnostico(
        componente_id=componente.id,
        timestamp=datetime.now(timezone.utc),
        **diagnostico_kwargs,
    ))

    db.commit()
    print(f"  Criado: {nome_maquina} (maquina_id={maquina.id}, componente_id={componente.id})")


def run() -> None:
    create_tables()
    db = SessionLocal()
    try:
        print(f"Semeando {len(CONDICOES)} motores de demonstração na planta '{PLANTA_FIAP_NOME}'...")
        for nome_maquina, diagnostico_kwargs in CONDICOES:
            seed_condicao(db, nome_maquina, diagnostico_kwargs)
    finally:
        db.close()
    print("Concluído.")


if __name__ == "__main__":
    run()
