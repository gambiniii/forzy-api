"""
Aplica a migration 005 e recalibra os limites do Metric Contract.

POR QUE ISSO É NECESSÁRIO
=========================
Os limites gravados hoje no banco deixam os dois motores permanentemente em
estado crítico, o que torna o destaque no modelo 3D inútil: se tudo está sempre
vermelho, o vermelho não informa nada.

    componente 2 (S1): vib_atencao=1,8   vib_critico=4,5
    componente 3 (S2): vib_atencao=0,01  vib_critico=0,02

O componente 3 está calibrado em escala de ACELERAÇÃO (g) mas é comparado contra
a VELOCIDADE de vibração (mm/s), porque `leitura_sensor.rpm` guarda velocidade e
não rotação. Com o motor PARADO a velocidade já é 0,049 mm/s, que é maior que o
limite crítico de 0,02 — então o S2 é "crítico" 100% do tempo, inclusive desligado.

O componente 2 usa os defaults da ISO 10816-1 Classe I (1,8 e 4,5 mm/s), que são
corretos pela norma para um motor de 2,2 kW. O problema é empírico: a baseline
SAUDÁVEL medida em 3h57 de telemetria real é 6,68 ± 0,25 mm/s em regime. Ou seja,
o motor cruza o limite crítico no instante em que liga.

DE ONDE VÊM OS VALORES NOVOS
============================
Da baseline medida do próprio motor, não de norma absoluta. Isso é deliberado: o
manual WEG (Tabela 7.7, ISO 20816-3, saída ≤ 300 kW) dá crítico de 5,6 mm/s em
base rígida e 8,9 em base flexível — e a nossa baseline saudável de 6,68 já fica
acima do primeiro. Ou a montagem amplifica, ou o sensor não reporta velocidade
limitada à banda ISO. Em qualquer dos casos, o critério que funciona é DESVIO
RELATIVO À BASELINE DO PRÓPRIO MOTOR.

    vib_atencao  = baseline + 2 desvios  = 6,68 + 2(0,25) ≈ 7,2 mm/s
    vib_critico  = baseline + 4 desvios  = 6,68 + 4(0,25) ≈ 7,7 mm/s

Como o S2 vibra mais que o S1 (6,88 contra 6,68, medido, p < 1e-10), cada motor
recebe o seu. Para dar folga operacional sem perder sensibilidade, arredondamos
para cima e cruzamos com a referência da norma: o crítico fica abaixo do limite de
base flexível da WEG (8,9 mm/s), que é o teto que a norma admite.

    acel_atencao = 0,63 g   (baseline 0,50 g + 2 desvios de 0,074)
    acel_critico = 0,78 g   (baseline + 4 desvios)

Temperatura fica em 70 e 90 °C: a máxima medida foi 47 °C no pico do heat soak,
então esses limites nunca disparam por engano e continuam conservadores frente ao
teto de 155 °C da classe de isolamento F.

ESTES SÃO PRIORS DECLARADOS, NÃO AJUSTE ESTATÍSTICO. Com cerca de 19 minutos de
regime estacionário por motor, uma única condição de carga e zero exemplos de
falha no histórico, não existe como calibrar de outra forma. Isso está dito aqui
de propósito, para ninguém apresentar estes números como se fossem derivados.

USO
===
    python scripts/corrigir_limites.py --ver       # só mostra o estado atual
    python scripts/corrigir_limites.py --migrar    # SÓ o schema (ADD COLUMN)
    python scripts/corrigir_limites.py --aplicar   # schema + recalibra os valores

POR QUE --migrar EXISTE SEPARADO
================================
As colunas `acel_atencao` e `acel_critico` foram acrescentadas ao modelo ORM
`ComponenteLimite`. O SQLAlchemy seleciona TODAS as colunas mapeadas em qualquer
consulta, então enquanto a coluna não existir no banco o endpoint
`GET /componentes/{id}/limites` devolve erro 500 — o `getattr` defensivo no
`_classify_threshold` protege o acesso ao atributo, mas não a query.

Ou seja: `--migrar` conserta uma quebra. É aditivo, idempotente
(`ADD COLUMN IF NOT EXISTS`), tem default e não altera nenhum dado existente.

`--aplicar` vai além e RECALIBRA os valores, o que é uma mudança de
comportamento do sistema de alarme. Essa parte merece decisão explícita.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import text  # noqa: E402

from src.db.postgres import SessionLocal  # noqa: E402

# baseline medida (média, desvio) em regime de operação, por componente
BASELINE = {
    2: {"v": (6.684, 0.251), "a": (0.498, 0.074)},   # S1 = MOTOR-01
    3: {"v": (6.882, 0.381), "a": (0.509, 0.074)},   # S2 = MOTOR-02
}
TETO_NORMA = 8.9   # ISO 20816-3, base flexível, saída ≤ 300 kW (Tabela 7.7 WEG)


def _alvos(componente_id: int) -> dict:
    b = BASELINE.get(componente_id)
    if not b:
        return {}
    vm, vs = b["v"]
    am, asd = b["a"]
    return {
        "vib_atencao": round(vm + 2 * vs, 2),
        "vib_critico": round(min(vm + 4 * vs, TETO_NORMA), 2),
        "acel_atencao": round(am + 2 * asd, 3),
        "acel_critico": round(am + 4 * asd, 3),
    }


def mostrar(db) -> None:
    cols = [r[0] for r in db.execute(text(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name='componente_limite' ORDER BY ordinal_position"))]
    print("colunas de componente_limite:", cols)
    tem_acel = "acel_atencao" in cols
    print("migration 005 aplicada:", "sim" if tem_acel else "NÃO")
    print()
    print("estado atual:")
    for r in db.execute(text("SELECT * FROM componente_limite ORDER BY componente_id")):
        print("  ", dict(r._mapping))
    print()
    print("valores propostos (baseline do próprio motor + 2 e + 4 desvios):")
    for cid in sorted(BASELINE):
        print(f"   componente {cid}: {_alvos(cid)}")


def migrar(db) -> None:
    """Só o schema: ADD COLUMN IF NOT EXISTS. Aditivo e idempotente.

    O parse remove os comentários LINHA POR LINHA antes de separar os comandos.
    Filtrar o comando inteiro por começar com "--" não funciona: o bloco de
    comentários do topo do arquivo fica colado no primeiro ALTER, então esse
    primeiro comando era descartado em silêncio e só a segunda coluna era criada.
    """
    bruto = (PROJECT_ROOT / "migrations" / "005_componente_limite_aceleracao.sql").read_text(
        encoding="utf-8")
    sem_comentario = "\n".join(
        linha for linha in bruto.splitlines() if not linha.strip().startswith("--")
    )
    comandos = [c.strip() for c in sem_comentario.split(";") if c.strip()]
    for stmt in comandos:
        db.execute(text(stmt))
    db.commit()
    print(f"migration 005 aplicada (schema): {len(comandos)} comando(s).")


def aplicar(db) -> None:
    migrar(db)

    for cid, alvo in ((c, _alvos(c)) for c in sorted(BASELINE)):
        db.execute(text("""
            UPDATE componente_limite
               SET vib_atencao = :va, vib_critico = :vc,
                   acel_atencao = :aa, acel_critico = :ac
             WHERE componente_id = :cid
        """), {"cid": cid, "va": alvo["vib_atencao"], "vc": alvo["vib_critico"],
               "aa": alvo["acel_atencao"], "ac": alvo["acel_critico"]})
    db.commit()
    print("limites recalibrados.\n")
    mostrar(db)


if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else "--ver"
    db = SessionLocal()
    try:
        if modo == "--aplicar":
            aplicar(db)
        elif modo == "--migrar":
            migrar(db)
            print()
            mostrar(db)
            print("\nOs VALORES não foram alterados. Para recalibrar: "
                  "python scripts/corrigir_limites.py --aplicar")
        else:
            mostrar(db)
            print("\nNada foi alterado.")
            print("  --migrar   aplica só o schema; conserta o erro 500 do endpoint de limites")
            print("  --aplicar  aplica o schema E recalibra os valores")
    finally:
        db.close()
