-- =========================================
-- MIGRATION 002 — Dados completos Motor WEG W22 3cv
-- Fonte: placa técnica + catálogo WEG #13887610
-- =========================================

-- =========================================
-- NOVOS ATRIBUTOS
-- =========================================
INSERT INTO atributo (nome, unidade, tipo_dado) VALUES
    ('Carcaça',                   NULL,   'string'),
    ('Potência',                  'kW',   'float'),
    ('Frequência',                'Hz',   'float'),
    ('Tensão nominal',            'V',    'string'),
    ('Corrente nominal',          'A',    'string'),
    ('Número de polos',           NULL,   'int'),
    ('Ip/In',                     NULL,   'float'),
    ('Rotação nominal',           'rpm',  'float'),
    ('Conjugado nominal',         'kgfm', 'float'),
    ('Classe de isolamento',      NULL,   'string'),
    ('Fator de serviço',          NULL,   'float'),
    ('Elevação de temperatura',   '°C',   'int'),
    ('Regime de serviço',         NULL,   'string'),
    ('Temperatura ambiente',      NULL,   'string'),
    ('Altitude máxima',           'm',    'int'),
    ('Grau de proteção',          NULL,   'string'),
    ('Método de refrigeração',    NULL,   'string'),
    ('Forma construtiva',         NULL,   'string'),
    ('Sentido de rotação',        NULL,   'string'),
    ('Nível de ruído',            'dB',   'float'),
    ('Método de partida',         NULL,   'string'),
    ('Massa aproximada',          'kg',   'float'),
    ('Rendimento 50%',            '%',    'float'),
    ('Rendimento 75%',            '%',    'float'),
    ('Rendimento 100%',           '%',    'float'),
    ('Cos φ 50%',                 NULL,   'float'),
    ('Cos φ 75%',                 NULL,   'float'),
    ('Cos φ 100%',                NULL,   'float'),
    ('Tração máxima',             'kgf',  'int'),
    ('Compressão máxima',         'kgf',  'int'),
    ('Rolamento dianteiro',       NULL,   'string'),
    ('Rolamento traseiro',        NULL,   'string'),
    ('Vedação dianteira',         NULL,   'string'),
    ('Vedação traseira',          NULL,   'string'),
    ('Tipo de lubrificante',      NULL,   'string'),
    ('Tempo rotor bloqueado frio','s',    'int'),
    ('Tempo rotor bloqueado quente','s',  'int'),
    ('Código do produto',         NULL,   'string')
ON CONFLICT (nome) DO NOTHING;

-- =========================================
-- ESPECIFICAÇÃO MOTOR — atualiza com dados completos
-- =========================================
UPDATE especificacao_motor SET
    potencia_kw      = 2.2,
    tensao_nominal   = 220,
    corrente_nominal = 12.5,
    rpm_nominal      = 3525,
    frequencia_hz    = 60,
    numero_polos     = 2,
    rendimento       = 81.8
WHERE componente_id = 1;

-- =========================================
-- VALORES DOS NOVOS ATRIBUTOS (componente_id = 1)
-- =========================================

-- helper: busca id do atributo pelo nome
WITH a AS (SELECT id, nome FROM atributo)
INSERT INTO componente_atributo_valor (componente_id, atributo_id, valor_string, valor_float, valor_int)
SELECT 1, a.id, v.valor_string, v.valor_float, v.valor_int
FROM (VALUES
    ('Carcaça',                     '100L',             NULL,   NULL),
    ('Potência',                    NULL,               2.2,    NULL),
    ('Frequência',                  NULL,               60.0,   NULL),
    ('Tensão nominal',              '110-127/220-254',  NULL,   NULL),
    ('Corrente nominal',            '25.0-21.7/12.5-10.8', NULL, NULL),
    ('Número de polos',             NULL,               NULL,   2),
    ('Ip/In',                       NULL,               8.7,    NULL),
    ('Rotação nominal',             NULL,               3525.0, NULL),
    ('Conjugado nominal',           NULL,               0.608,  NULL),
    ('Classe de isolamento',        'F',                NULL,   NULL),
    ('Fator de serviço',            NULL,               1.15,   NULL),
    ('Elevação de temperatura',     NULL,               NULL,   105),
    ('Regime de serviço',           'S1',               NULL,   NULL),
    ('Temperatura ambiente',        '-20°C a +40°C',    NULL,   NULL),
    ('Altitude máxima',             NULL,               NULL,   1000),
    ('Grau de proteção',            'IP55',             NULL,   NULL),
    ('Método de refrigeração',      'IC411 - TFVE',     NULL,   NULL),
    ('Forma construtiva',           'B3D',              NULL,   NULL),
    ('Sentido de rotação',          'Ambos',            NULL,   NULL),
    ('Nível de ruído',              NULL,               72.0,   NULL),
    ('Método de partida',           'Partida direta',   NULL,   NULL),
    ('Massa aproximada',            NULL,               38.6,   NULL),
    ('Rendimento 50%',              NULL,               72.7,   NULL),
    ('Rendimento 75%',              NULL,               79.2,   NULL),
    ('Rendimento 100%',             NULL,               81.8,   NULL),
    ('Cos φ 50%',                   NULL,               0.92,   NULL),
    ('Cos φ 75%',                   NULL,               0.95,   NULL),
    ('Cos φ 100%',                  NULL,               0.98,   NULL),
    ('Tração máxima',               NULL,               NULL,   26),
    ('Compressão máxima',           NULL,               NULL,   64),
    ('Rolamento dianteiro',         '6206 ZZ',          NULL,   NULL),
    ('Rolamento traseiro',          '6206 ZZ',          NULL,   NULL),
    ('Vedação dianteira',           'V''Ring',          NULL,   NULL),
    ('Vedação traseira',            'V''Ring',          NULL,   NULL),
    ('Tipo de lubrificante',        '00088',            NULL,   NULL),
    ('Tempo rotor bloqueado frio',  NULL,               NULL,   16),
    ('Tempo rotor bloqueado quente',NULL,               NULL,   9),
    ('Código do produto',           '13887610',         NULL,   NULL)
) AS v(nome, valor_string, valor_float, valor_int)
JOIN a ON a.nome = v.nome;
