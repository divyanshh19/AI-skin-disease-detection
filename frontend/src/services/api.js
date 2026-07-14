import axios from 'axios';

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 30000,
});

export async function predictImage(file) {
  const formData = new FormData();
  formData.append('file', file);
  const response = await api.post('/predict', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
}

export async function checkHealth() {
  const response = await api.get('/health');
  return response.data;
}

export default api;
