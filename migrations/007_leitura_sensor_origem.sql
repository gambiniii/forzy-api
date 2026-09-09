-- 007_leitura_sensor_origem.sql
-- Marca a origem de cada leitura ("real" = hardware Forzy, "demo" = replay do
-- histórico controlado por src/services/sensor_demo.py). Sem isso, dado de
-- demonstração fica indistinguível de dado real no banco para sempre.

ALTER TABLE leitura_sensor
    ADD COLUMN IF NOT EXISTS origem VARCHAR(10) NOT NULL DEFAULT 'real';
