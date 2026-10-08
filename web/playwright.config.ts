import { defineConfig } from "@playwright/test";

// Runs against the already running application (Docker Compose by default).
export default defineConfig({
  testDir: "e2e",
  timeout: 120_000,
  workers: 1,
  reporter: "list",
  use: {
    baseURL: process.env.BASE_URL ?? "http://localhost:8080",
    viewport: { width: 1440, height: 900 },
  },
});
