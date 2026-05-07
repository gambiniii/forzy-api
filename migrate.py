"""
Roda todas as migrations pendentes em ordem.
Uso: python migrate.py
"""
from src.db.postgres import SessionLocal
from src.models.atributo import Atributo, ComponenteAtributoValor
from src.models.componente import Componente, EspecificacaoMotor
from src.models.maquina import Maquina
from src.models.enums import StatusEnum


def migration_002_motor_weg_w22(db):
    """Dados completos do Motor WEG W22 3cv — placa técnica + catálogo #13887610."""

    # -- Atributos novos --
    novos_atributos = [
        ("Carcaça",                       None,   "string"),
        ("Potência",                       "kW",   "float"),
        ("Frequência",                     "Hz",   "float"),
        ("Tensão nominal",                 "V",    "string"),
        ("Corrente nominal",               "A",    "string"),
        ("Número de polos",                None,   "int"),
        ("Ip/In",                          None,   "float"),
        ("Rotação nominal",                "rpm",  "float"),
        ("Conjugado nominal",              "kgfm", "float"),
        ("Elevação de temperatura",        "°C",   "int"),
        ("Regime de serviço",              None,   "string"),
        ("Temperatura ambiente",           None,   "string"),
        ("Altitude máxima",                "m",    "int"),
        ("Método de refrigeração",         None,   "string"),
        ("Forma construtiva",              None,   "string"),
        ("Sentido de rotação",             None,   "string"),
        ("Método de partida",              None,   "string"),
        ("Rendimento 50%",                 "%",    "float"),
        ("Rendimento 75%",                 "%",    "float"),
        ("Rendimento 100%",                "%",    "float"),
        ("Cos φ 50%",                      None,   "float"),
        ("Cos φ 75%",                      None,   "float"),
        ("Cos φ 100%",                     None,   "float"),
        ("Tração máxima",                  "kgf",  "int"),
        ("Compressão máxima",              "kgf",  "int"),
        ("Rolamento dianteiro",            None,   "string"),
        ("Rolamento traseiro",             None,   "string"),
        ("Vedação dianteira",              None,   "string"),
        ("Vedação traseira",               None,   "string"),
        ("Tipo de lubrificante",           None,   "string"),
        ("Tempo rotor bloqueado frio",     "s",    "int"),
        ("Tempo rotor bloqueado quente",   "s",    "int"),
        ("Código do produto",              None,   "string"),
    ]

    for nome, unidade, tipo_dado in novos_atributos:
        existe = db.query(Atributo).filter(Atributo.nome == nome).first()
        if not existe:
            db.add(Atributo(nome=nome, unidade=unidade, tipo_dado=tipo_dado))

    db.flush()

    # -- Atualiza especificação do motor (componente_id = 1) --
    esp = db.query(EspecificacaoMotor).filter(EspecificacaoMotor.componente_id == 1).first()
    if esp:
        esp.potencia_kw      = 2.2
        esp.tensao_nominal   = 220
        esp.corrente_nominal = 12.5
        esp.rpm_nominal      = 3525
        esp.frequencia_hz    = 60
        esp.numero_polos     = 2
        esp.rendimento       = 81.8

    # -- Valores dos atributos (componente_id = 1) --
    # formato: (nome_atributo, valor_string, valor_float, valor_int)
    valores = [
        ("Carcaça",                     "100L",                 None,   None),
        ("Potência",                     None,                  2.2,    None),
        ("Frequência",                   None,                  60.0,   None),
        ("Tensão nominal",               "110-127/220-254",     None,   None),
        ("Corrente nominal",             "25.0-21.7/12.5-10.8", None,  None),
        ("Número de polos",              None,                  None,   2),
        ("Ip/In",                        None,                  8.7,    None),
        ("Rotação nominal",              None,                  3525.0, None),
        ("Conjugado nominal",            None,                  0.608,  None),
        ("Classe de Isolamento",         "F",                   None,   None),
        ("Fator de Serviço",             None,                  1.15,   None),
        ("Elevação de temperatura",      None,                  None,   105),
        ("Regime de serviço",            "S1",                  None,   None),
        ("Temperatura ambiente",         "-20°C a +40°C",       None,   None),
        ("Altitude máxima",              None,                  None,   1000),
        ("Grau de Proteção",             "IP55",                None,   None),
        ("Método de refrigeração",       "IC411 - TFVE",        None,   None),
        ("Forma construtiva",            "B3D",                 None,   None),
        ("Sentido de rotação",           "Ambos",               None,   None),
        ("Ruído",                        None,                  72.0,   None),
        ("Método de partida",            "Partida direta",      None,   None),
        ("Massa",                        None,                  38.6,   None),
        ("Rendimento 50%",               None,                  72.7,   None),
        ("Rendimento 75%",               None,                  79.2,   None),
        ("Rendimento 100%",              None,                  81.8,   None),
        ("Cos φ 50%",                    None,                  0.92,   None),
        ("Cos φ 75%",                    None,                  0.95,   None),
        ("Cos φ 100%",                   None,                  0.98,   None),
        ("Tração máxima",                None,                  None,   26),
        ("Compressão máxima",            None,                  None,   64),
        ("Rolamento dianteiro",          "6206 ZZ",             None,   None),
        ("Rolamento traseiro",           "6206 ZZ",             None,   None),
        ("Vedação dianteira",            "V'Ring",              None,   None),
        ("Vedação traseira",             "V'Ring",              None,   None),
        ("Tipo de lubrificante",         "00088",               None,   None),
        ("Tempo rotor bloqueado frio",   None,                  None,   16),
        ("Tempo rotor bloqueado quente", None,                  None,   9),
        ("Código do produto",            "13887610",            None,   None),
    ]

    for nome_atrib, val_str, val_float, val_int in valores:
        atrib = db.query(Atributo).filter(Atributo.nome == nome_atrib).first()
        if not atrib:
            print(f"  [AVISO] Atributo '{nome_atrib}' não encontrado, pulando.")
            continue

        ja_existe = db.query(ComponenteAtributoValor).filter(
            ComponenteAtributoValor.componente_id == 1,
            ComponenteAtributoValor.atributo_id == atrib.id,
        ).first()

        if not ja_existe:
            db.add(ComponenteAtributoValor(
                componente_id=1,
                atributo_id=atrib.id,
                valor_string=val_str,
                valor_float=val_float,
                valor_int=val_int,
            ))


def _upsert_atributo(db, nome, unidade, tipo_dado):
    a = db.query(Atributo).filter(Atributo.nome == nome).first()
    if not a:
        a = Atributo(nome=nome, unidade=unidade, tipo_dado=tipo_dado)
        db.add(a)
        db.flush()
    return a


def _add_valor(db, componente_id, atrib, val_str, val_float, val_int):
    existe = db.query(ComponenteAtributoValor).filter(
        ComponenteAtributoValor.componente_id == componente_id,
        ComponenteAtributoValor.atributo_id == atrib.id,
    ).first()
    if not existe:
        db.add(ComponenteAtributoValor(
            componente_id=componente_id,
            atributo_id=atrib.id,
            valor_string=val_str,
            valor_float=val_float,
            valor_int=val_int,
        ))


def migration_003_motor_weg_w22_monofasico(db):
    """Nova máquina: Motor Monofásico WEG W22 3cv — placa técnica + catálogo #13887610."""

    # -- Máquina --
    maquina = db.query(Maquina).filter(Maquina.nome == "Motor Monofásico de Indução WEG W22").first()
    if not maquina:
        maquina = Maquina(
            nome="Motor Monofásico de Indução WEG W22",
            tipo="Motor Elétrico Monofásico",
            fabricante="WEG",
            ano_instalacao=2024,
            status=StatusEnum.active,
            localizacao="Setor B",
        )
        db.add(maquina)
        db.flush()

    # -- Componente --
    comp = db.query(Componente).filter(
        Componente.maquina_id == maquina.id,
        Componente.nome == "Motor W22 Monofásico 3cv",
    ).first()
    if not comp:
        from datetime import date
        comp = Componente(
            maquina_id=maquina.id,
            nome="Motor W22 Monofásico 3cv",
            tipo="Motor",
            fabricante="WEG",
            data_instalacao=date(2024, 1, 10),
            status=StatusEnum.active,
        )
        db.add(comp)
        db.flush()

    # -- Especificação motor --
    esp = db.query(EspecificacaoMotor).filter(EspecificacaoMotor.componente_id == comp.id).first()
    if not esp:
        db.add(EspecificacaoMotor(
            componente_id=comp.id,
            potencia_kw=2.2,
            tensao_nominal=220,
            corrente_nominal=12,
            rpm_nominal=3525,
            frequencia_hz=60,
            numero_polos=2,
            rendimento=82,
        ))

    # -- Atributos e valores --
    dados = [
        ("Linha do produto",             "string", None,  "W22 Monofásico",        None,   None),
        ("Código do produto",            "string", None,  "13887610",              None,   None),
        ("Carcaça",                      "string", None,  "100L",                  None,   None),
        ("Potência",                     "float",  "kW",   None,                   2.2,    None),
        ("Frequência",                   "float",  "Hz",   None,                   60.0,   None),
        ("Tensão nominal",               "string", "V",   "110-127/220-254",       None,   None),
        ("Corrente nominal",             "string", "A",   "25.0-21.7/12.5-10.8",  None,   None),
        ("Número de polos",              "int",    None,   None,                   None,   2),
        ("Ip/In",                        "float",  None,   None,                   8.7,    None),
        ("Rotação nominal",              "float",  "rpm",  None,                   3525.0, None),
        ("Conjugado nominal",            "float",  "kgfm", None,                   0.608,  None),
        ("Classe de Isolamento",         "string", None,  "F",                     None,   None),
        ("Fator de Serviço",             "float",  None,   None,                   1.15,   None),
        ("Elevação de temperatura",      "int",    "°C",   None,                   None,   105),
        ("Regime de serviço",            "string", None,  "S1",                    None,   None),
        ("Temperatura ambiente",         "string", None,  "-20°C a +40°C",         None,   None),
        ("Altitude máxima",              "int",    "m",    None,                   None,   1000),
        ("Grau de Proteção",             "string", None,  "IP55",                  None,   None),
        ("Método de refrigeração",       "string", None,  "IC411 - TFVE",          None,   None),
        ("Forma construtiva",            "string", None,  "B3D",                   None,   None),
        ("Sentido de rotação",           "string", None,  "Ambos",                 None,   None),
        ("Ruído",                        "float",  "dB",   None,                   72.0,   None),
        ("Método de partida",            "string", None,  "Partida direta",        None,   None),
        ("Massa",                        "float",  "kg",   None,                   38.6,   None),
        ("Rendimento 50%",               "float",  "%",    None,                   72.7,   None),
        ("Rendimento 75%",               "float",  "%",    None,                   79.2,   None),
        ("Rendimento 100%",              "float",  "%",    None,                   81.8,   None),
        ("Cos φ 50%",                    "float",  None,   None,                   0.92,   None),
        ("Cos φ 75%",                    "float",  None,   None,                   0.95,   None),
        ("Cos φ 100%",                   "float",  None,   None,                   0.98,   None),
        ("Tração máxima",                "int",    "kgf",  None,                   None,   26),
        ("Compressão máxima",            "int",    "kgf",  None,                   None,   64),
        ("Rolamento dianteiro",          "string", None,  "6206 ZZ",               None,   None),
        ("Rolamento traseiro",           "string", None,  "6206 ZZ",               None,   None),
        ("Vedação dianteira",            "string", None,  "V'Ring",                None,   None),
        ("Vedação traseira",             "string", None,  "V'Ring",                None,   None),
        ("Tipo de lubrificante",         "string", None,  "00088",                 None,   None),
        ("Tempo rotor bloqueado frio",   "int",    "s",    None,                   None,   16),
        ("Tempo rotor bloqueado quente", "int",    "s",    None,                   None,   9),
    ]

    for nome, tipo_dado, unidade, val_str, val_float, val_int in dados:
        atrib = _upsert_atributo(db, nome, unidade, tipo_dado)
        _add_valor(db, comp.id, atrib, val_str, val_float, val_int)


MIGRATIONS = [
    ("002_motor_weg_w22",                 migration_002_motor_weg_w22),
    ("003_motor_weg_w22_monofasico",      migration_003_motor_weg_w22_monofasico),
]


def run():
    db = SessionLocal()
    try:
        for nome, fn in MIGRATIONS:
            print(f"Rodando migration: {nome}...")
            fn(db)
            db.commit()
            print(f"  OK.")
    except Exception as e:
        db.rollback()
        print(f"  ERRO: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    run()
