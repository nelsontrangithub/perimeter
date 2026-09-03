import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The console is served by Perimeter itself at /admin/ (FastAPI StaticFiles).
// In development, API calls are proxied to a locally running `perimeter serve`.
export default defineConfig({
  base: "/admin/",
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/admin/api": "http://127.0.0.1:8000",
      "/health": "http://127.0.0.1:8000",
    },
  },
  build: { outDir: "dist", sourcemap: false },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test-setup.ts"],
  },
});
