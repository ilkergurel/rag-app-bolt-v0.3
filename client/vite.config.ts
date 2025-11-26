import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  optimizeDeps: {
    exclude: ['lucide-react'],
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:5000', // Your Node.js backend
        changeOrigin: true,
        secure: false,
        // This is vital for streaming:
        configure: (proxy, _options) => {
          proxy.on('proxyReq', (proxyReq, req, _res) => {
            // Force keep-alive on the outgoing request
            proxyReq.setHeader('Connection', 'keep-alive');
          });
        },
      },
    },
  },
});
