-- 005_componente_limite_aceleracao.sql
-- Acrescenta os limites de ACELERAÇÃO ao Metric Contract.
--
-- Motivo: `_classify_threshold` só sabia emitir "velocidade" e "temperatura" em
-- `breached_metrics`, porque `componente_limite` só tinha esses dois pares de
-- limites. Consequência prática: a métrica "aceleracao" nunca entrava no
-- diagnóstico e, no modelo 3D, os rolamentos nunca podiam ser destacados —
-- justamente o componente que a aceleração melhor denuncia.
--
-- Semântica dos campos (atenção aos nomes das colunas de leitura_sensor, que
-- enganam): `leitura_sensor.vibracao` guarda ACELERAÇÃO em g, e
-- `leitura_sensor.rpm` guarda a VELOCIDADE de vibração em mm/s.
-- Estes limites são comparados contra `leitura_sensor.vibracao`.
--
-- Defaults derivados da telemetria real (3h57 dos motores S1 e S2, 19/05/2026):
-- em regime de operação a aceleração saudável medida é 0,484 ± 0,074 g.
--   atenção = baseline + 2 desvios ≈ 0,63 g
--   crítico = baseline + 4 desvios ≈ 0,78 g
-- São PRIORS declarados, não ajuste estatístico: com ~19 min de regime e zero
-- exemplos de falha no histórico, não há como calibrar de outra forma.
-- NÃO use os 2 g / 4 g do documento de governança: aqueles valores se referem à
-- banda de alta frequência (cláusula prospectiva), não à aceleração agregada que
-- o sensor entrega hoje — com eles o alarme nunca dispararia.

ALTER TABLE componente_limite
    ADD COLUMN IF NOT EXISTS acel_atencao DOUBLE PRECISION NOT NULL DEFAULT 0.63;

ALTER TABLE componente_limite
    ADD COLUMN IF NOT EXISTS acel_critico DOUBLE PRECISION NOT NULL DEFAULT 0.78;
