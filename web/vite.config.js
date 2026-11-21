import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// base './' so the built site (dist/) works from any folder or static host
export default defineConfig({
  base: './',
  plugins: [react()],
  server: { port: 4501 },
  preview: { port: 4502 },
});
