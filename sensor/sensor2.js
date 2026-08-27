const fs = require("fs");
const path = require("path");
const WebSocket = require("ws");

const CSV_FILE    = path.join(__dirname, "History_32026-05-19T11-46-10-920.csv");
const WS_URL      = "ws://localhost:8000/leituras/ws/3";  // S2 → componente_id=3 (Unidade S2)
const INTERVAL_MS = 1000;
const PORT        = "port2";

function loadRows() {
  const lines = fs.readFileSync(CSV_FILE, "utf-8").split("\n").filter(Boolean);
  return lines.slice(3).map(line => {
    const cols = line.split(";");
    return {
      rpm:         parseFloat(cols[6]),  // 2.1. Velocidade
      vibracao:    parseFloat(cols[7]),  // 2.2. Aceleração
      temperatura: parseFloat(cols[8]),  // 2.3. Temperatura
    };
  }).filter(r => !isNaN(r.temperatura));
}

let index = 0;

function connect(rows) {
  console.log(`[${PORT}] Conectando em ${WS_URL}...`);
  const ws = new WebSocket(WS_URL);
  let timer = null;

  ws.on("open", () => {
    console.log(`[${PORT}] Conectado. Iniciando envio de leituras...`);
    timer = setInterval(() => {
      const row = rows[index % rows.length];
      index++;
      const payload = { type: "sensor", ...row };
      ws.send(JSON.stringify(payload));
      console.log(`[${PORT}] #${index} → temp=${row.temperatura}°C  vib=${row.vibracao}  rpm=${row.rpm}`);
    }, INTERVAL_MS);
  });

  ws.on("message", (raw) => {
    try {
      const msg = JSON.parse(raw);
      if (msg.type === "status") console.log(`[${PORT}] status: online=${msg.online}`);
    } catch (_) {}
  });

  ws.on("close", () => {
    console.log(`[${PORT}] Conexão encerrada. Reconectando em 3s...`);
    if (timer) clearInterval(timer);
    setTimeout(() => connect(rows), 3000);
  });

  ws.on("error", (err) => console.error(`[${PORT}] Erro:`, err.message));
}

const rows = loadRows();
console.log(`[${PORT}] Dataset carregado: ${rows.length} leituras`);
connect(rows);
