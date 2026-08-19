-- 003_forzy_sensor_readings.sql
-- Armazena leituras coletadas da API Forzy (motor WEG W22)
-- Janela de coleta: Segunda, Terça, Quarta — 12h-14h BRT

CREATE TABLE IF NOT EXISTS forzy_sensor_readings (
    id           BIGSERIAL PRIMARY KEY,
    collected_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    motor_id     INTEGER NOT NULL,
    sensor_port  SMALLINT NOT NULL CHECK (sensor_port IN (1, 2)),
    velocidade   FLOAT NOT NULL,
    aceleracao   FLOAT NOT NULL,
    temperatura  FLOAT NOT NULL,
    api_url      TEXT,
    session_id   TEXT
);

CREATE INDEX IF NOT EXISTS idx_fsr_collected_at
    ON forzy_sensor_readings(collected_at DESC);

CREATE INDEX IF NOT EXISTS idx_fsr_motor_sensor
    ON forzy_sensor_readings(motor_id, sensor_port, collected_at DESC);
