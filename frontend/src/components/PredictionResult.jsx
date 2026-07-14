import React from 'react';

function ProbabilityBar({ label, probability }) {
  return (
    <div className="prob-bar-row">
      <div className="prob-bar-label">{label}</div>
      <div className="prob-bar-track">
        <div className="prob-bar-fill" style={{ width: `${(probability * 100).toFixed(1)}%` }} />
      </div>
      <div className="prob-bar-value">{(probability * 100).toFixed(1)}%</div>
    </div>
  );
}

export default function PredictionResult({ result }) {
  if (!result) {
    return (
      <div className="empty-state">
        Upload an image and click "Analyze" to see results here.
      </div>
    );
  }

  const { predicted_display_name, confidence, class_probabilities,
          disease_info, gradcam_image_base64, warning, models_agree } = result;

  return (
    <div>
      <div className="result-header">
        <div className="predicted-label">{predicted_display_name}</div>
        <div className="confidence-badge">{(confidence * 100).toFixed(1)}% confidence</div>
      </div>

      <span className={disease_info.is_malignant_risk ? 'malignant-tag' : 'benign-tag'}>
        {disease_info.is_malignant_risk ? '⚠ Malignant risk' : '✓ Typically benign'}
      </span>

      <p style={{ color: 'var(--text-dim)', fontSize: '0.9rem', lineHeight: 1.5 }}>
        {disease_info.description}
      </p>
      <p style={{ fontSize: '0.9rem' }}>
        <strong>Recommendation:</strong> {disease_info.recommendation}
      </p>

      {!models_agree && warning && (
        <div className="warning-box">{warning}</div>
      )}

      <h4 style={{ marginTop: '20px', marginBottom: '8px' }}>Class Probabilities</h4>
      {class_probabilities.map((cp) => (
        <ProbabilityBar key={cp.label} label={cp.display_name} probability={cp.probability} />
      ))}

      {gradcam_image_base64 && (
        <div className="gradcam-section">
          <h4>Grad-CAM Explainability</h4>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-dim)' }}>
            Highlighted regions show what the CNN focused on to make this prediction.
          </p>
          <img
            src={`data:image/png;base64,${gradcam_image_base64}`}
            alt="Grad-CAM overlay"
            className="gradcam-image"
          />
        </div>
      )}

      <div className="disclaimer-banner" style={{ marginTop: '20px' }}>
        AI-powered skin disease detection system developed for research, learning, and demonstration purposes.
      </div>
    </div>
  );
}
