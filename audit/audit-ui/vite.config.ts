import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { '@': decodeURIComponent(new URL('./src', import.meta.url).pathname).replace(/^\/([A-Za-z]:)/, '$1') } },
  server: {
    proxy: { '/api': 'http://127.0.0.1:8765' },
  },
})
