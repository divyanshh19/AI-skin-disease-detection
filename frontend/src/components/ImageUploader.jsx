import React, { useState, useRef } from 'react';

export default function ImageUploader({ onFileSelected, disabled }) {
  const [preview, setPreview] = useState(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  const handleFile = (file) => {
    if (!file) return;
    if (!['image/jpeg', 'image/png', 'image/jpg'].includes(file.type)) {
      alert('Please upload a JPEG or PNG image.');
      return;
    }
    setPreview(URL.createObjectURL(file));
    onFileSelected(file);
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  };

  return (
    <div>
      <div
        className={`upload-zone ${dragging ? 'dragging' : ''}`}
        onClick={() => !disabled && inputRef.current.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <p>📷 Click or drag a skin lesion image here</p>
        <p style={{ fontSize: '0.8rem', color: 'var(--text-dim)' }}>
          JPEG or PNG, up to 10MB
        </p>
        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png"
          style={{ display: 'none' }}
          disabled={disabled}
          onChange={(e) => handleFile(e.target.files[0])}
        />
      </div>
      {preview && <img src={preview} alt="Preview" className="preview-image" />}
    </div>
  );
}
