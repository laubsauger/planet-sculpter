import { defineConfig } from 'vite';

// Pages serves under /<repo>/ — set base for build only so dev stays at root.
export default defineConfig(({ command }) => ({
  base: command === 'build' ? '/planet-sculpter/' : '/',
  server: { port: 5173, open: false },
  build: { target: 'esnext' },
}));
