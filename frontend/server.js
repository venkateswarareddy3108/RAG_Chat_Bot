/**
 * Corpus frontend server — static UI + authenticated API proxy to FastAPI.
 */

const path = require("path");
const express = require("express");
const { createProxyMiddleware } = require("http-proxy-middleware");

const PORT = Number(process.env.PORT || 3000);
const BACKEND_URL = process.env.BACKEND_URL || "http://127.0.0.1:8000";

const app = express();

app.use(
  "/api",
  createProxyMiddleware({
    target: BACKEND_URL,
    changeOrigin: true,
    // Express strips the /api mount path; restore it for FastAPI routes.
    pathRewrite: (proxyPath) =>
      proxyPath.startsWith("/api") ? proxyPath : `/api${proxyPath}`,
    proxyTimeout: 0,
    timeout: 0,
    on: {
      error(err, _req, res) {
        console.error("[corpus-proxy]", err.message);
        if (!res.headersSent) {
          res.writeHead(502, { "Content-Type": "application/json" });
        }
        res.end(
          JSON.stringify({
            error: `Backend unreachable at ${BACKEND_URL}. Start FastAPI on port 8000.`,
          })
        );
      },
    },
  })
);

app.use(express.static(path.join(__dirname, "public"), {
  etag: true,
  maxAge: process.env.NODE_ENV === "production" ? "1h" : 0,
}));

app.get("*", (_req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

app.listen(PORT, () => {
  console.log(`Corpus UI  → http://localhost:${PORT}`);
  console.log(`API proxy  → ${BACKEND_URL}/api/*`);
});
