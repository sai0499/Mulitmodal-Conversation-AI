import axios from 'axios';

// Run this before any other axios usage:
axios.interceptors.request.use(config => {
  if (typeof config.url === 'string') {
    // Redirect express calls
    if (config.url.startsWith('http://localhost:4000')) {
      config.url = config.url.replace('http://localhost:4000', '/express');
    }
    // Redirect flask calls
    if (config.url.startsWith('http://localhost:5000')) {
      config.url = config.url.replace('http://localhost:5000', '/flask');
    }
  }
  return config;
}, error => Promise.reject(error));