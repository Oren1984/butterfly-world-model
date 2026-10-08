import { defineConfig } from "vitest/config";

// In development the browser talks to Vite, which forwards /api to the local backend.
// In Docker, nginx does the same job, so the frontend only ever uses relative URLs.
const target = "http://localhost:8000";

export default defineConfig({
  server: { proxy: { "/api": target, "/health": target } },
  test: { include: ["src/**/*.test.ts"] },
});
