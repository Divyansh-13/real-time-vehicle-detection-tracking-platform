import axios from 'axios';

const API_BASE = '/api/v1';

const api = axios.create({
  baseURL: API_BASE,
  headers: { 'Content-Type': 'application/json' },
});

// Add JWT token to requests
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Handle 401 responses
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('access_token');
      // Don't redirect to login for public endpoints
    }
    return Promise.reject(error);
  }
);

// ── Auth ──
export const authAPI = {
  register: (data) => api.post('/auth/register', data),
  login: (data) => api.post('/auth/login', data),
  getMe: () => api.get('/auth/me'),
};

// ── Predictions ──
export const predictAPI = {
  image: (formData, config = {}) =>
    api.post('/predict/image', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      ...config,
    }),
  video: (formData, config = {}) =>
    api.post('/predict/video', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      ...config,
    }),
  getById: (id) => api.get(`/predict/${id}`),
  list: (params = {}) => api.get('/predict/', { params }),
};

// ── Analytics ──
export const analyticsAPI = {
  summary: () => api.get('/analytics/summary'),
  timeline: (days = 30) => api.get('/analytics/timeline', { params: { days } }),
  classes: () => api.get('/analytics/classes'),
  recent: (limit = 10) => api.get('/analytics/recent', { params: { limit } }),
};

// ── Models ──
export const modelsAPI = {
  list: () => api.get('/models/'),
  getActive: () => api.get('/models/active'),
  activate: (id) => api.post(`/models/activate/${id}`),
  info: () => api.get('/models/info'),
};

// ── Health ──
export const healthAPI = {
  check: () => api.get('/health'),
  ready: () => api.get('/health/ready'),
};

export default api;
