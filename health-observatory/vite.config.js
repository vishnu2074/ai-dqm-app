import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3030,
    proxy: {
      '/api': {
        target: 'https://ai-dqm-app.onrender.com',
        changeOrigin: true,
        secure: true,
      }
    }
  }
})
