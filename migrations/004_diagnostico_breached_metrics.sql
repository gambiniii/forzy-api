-- 004_diagnostico_breached_metrics.sql
-- Expõe quais métricas do Metric Contract (threshold determinístico) estouraram
-- limite num diagnóstico, de forma estruturada (hoje só existe em texto livre
-- concatenado em threshold_message). Usado pelo frontend para destacar o
-- segmento físico correspondente no modelo 3D do equipamento.

ALTER TABLE diagnostico
    ADD COLUMN IF NOT EXISTS breached_metrics VARCHAR(200);
