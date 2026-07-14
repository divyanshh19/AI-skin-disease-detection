import React, { useState } from 'react';
import ImageUploader from './components/ImageUploader.jsx';
import PredictionResult from './components/PredictionResult.jsx';
import { predictImage } from './services/api.js';

export default function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await predictImage(file);
      setResult(data);
    } catch (err) {
      const detail = err.response?.data?.detail || err.message || 'Something went wrong.';
      setError(detail);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="header">
        <h1>SkinScan AI</h1>
        <p>CNN + Vision Transformer ensemble for skin lesion classification</p>
      </div>

      <div className="disclaimer-banner">
        AI-powered skin disease detection system developed for research, learning, and demonstration purposes.
      </div>

      <div className="main-grid">
        <div className="panel">
          <h3 style={{ marginTop: 0 }}>Upload Image</h3>
          <ImageUploader onFileSelected={setFile} disabled={loading} />
          <button className="btn" onClick={handleAnalyze} disabled={!file || loading}>
            {loading ? 'Analyzing...' : 'Analyze Image'}
          </button>
          {loading && <div className="spinner" style={{ marginTop: '16px' }} />}
          {error && <div className="error-box">{error}</div>}
        </div>

        <div className="panel">
          <h3 style={{ marginTop: 0 }}>Results</h3>
          <PredictionResult result={result} />
        </div>
      </div>
    </div>
  );
}
