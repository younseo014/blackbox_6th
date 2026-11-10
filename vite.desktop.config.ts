import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/postcss";
import { defineConfig } from "vite";
import { resolve } from "node:path";

export default defineConfig({
  root: resolve(__dirname, "desktop"),
  base: "./",
  publicDir: resolve(__dirname, "public"),
  plugins: [react()],
  optimizeDeps: { include: ["onnxruntime-web/wasm"] },
  server: {
    fs: { allow: ["desktop", "app", "models", "node_modules"].map(directory => resolve(__dirname, directory)) },
  },
  css: {
    postcss: {
      plugins: [tailwindcss()],
    },
  },
  build: {
    outDir: resolve(__dirname, "desktop", "dist-desktop"),
    emptyOutDir: true,
  },
});
