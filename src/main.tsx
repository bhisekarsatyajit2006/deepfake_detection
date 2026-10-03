// Ensure window.fetch has a setter in environments where it might be getter-only
try {
  if (typeof window !== 'undefined') {
    const origFetch = window.fetch;
    let currentFetch = origFetch;
    const desc = {
      get() {
        return currentFetch;
      },
      set(fn: typeof fetch) {
        currentFetch = fn;
      },
      configurable: true,
      enumerable: true,
    };
    try {
      Object.defineProperty(window, 'fetch', desc);
    } catch {}
    if (window.Window && window.Window.prototype) {
      try {
        Object.defineProperty(window.Window.prototype, 'fetch', desc);
      } catch {}
    }
  }
} catch {}

import {StrictMode} from 'react';
import {createRoot} from 'react-dom/client';
import App from './App.tsx';
import './index.css';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
