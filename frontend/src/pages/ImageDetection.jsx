import React, { useState, useRef, useCallback } from 'react';
import { predictAPI } from '../services/api.js';

export default function ImageDetection() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef(null);

  const handleFileSelect = useCallback((selectedFile) => {
    if (!selectedFile) return;
    setFile(selectedFile);
    setResult(null);
    setError(null);
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target.result);
    reader.readAsDataURL(selectedFile);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type.startsWith('image/')) {
      handleFileSelect(droppedFile);
    }
  }, [handleFileSelect]);

  const handleDetect = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);
      const response = await predictAPI.image(formData);
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Detection failed. Is the backend running?');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Image Detection</h1>
        <p className="page-subtitle">Upload an image to detect and classify vehicles</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-lg)' }}>
        {/* Upload Area */}
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Upload Image</h3>
            {file && (
              <button className="btn btn-sm btn-secondary" onClick={() => {
                setFile(null); setPreview(null); setResult(null); setError(null);
              }}>
                Clear
              </button>
            )}
          </div>

          {!preview ? (
            <div
              className={`upload-zone ${dragOver ? 'dragover' : ''}`}
              onClick={() => fileInputRef.current?.click()}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
            >
              <div className="upload-zone-icon">📷</div>
              <div className="upload-zone-text">Drop an image here or click to browse</div>
              <div className="upload-zone-hint">Supports: JPEG, PNG, BMP, WebP (max 100MB)</div>
            </div>
          ) : (
            <div>
              <div className="detection-viewer">
                <img src={preview} alt="Upload preview" />
              </div>
              <div style={{ marginTop: 'var(--space-md)', display: 'flex', gap: 'var(--space-sm)' }}>
                <button
                  className="btn btn-primary btn-lg"
                  onClick={handleDetect}
                  disabled={loading}
                  style={{ flex: 1 }}
                >
                  {loading ? (
                    <><div className="spinner" style={{ width: 20, height: 20, borderWidth: 2 }} /> Detecting...</>
                  ) : (
                    <>🔍 Run Detection</>
                  )}
                </button>
              </div>
            </div>
          )}

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            style={{ display: 'none' }}
            onChange={(e) => handleFileSelect(e.target.files[0])}
          />

          {error && (
            <div style={{
              marginTop: 'var(--space-md)', padding: 'var(--space-md)',
              background: 'rgba(245, 87, 108, 0.1)',
              border: '1px solid rgba(245, 87, 108, 0.3)',
              borderRadius: 'var(--radius-sm)', color: 'var(--color-error)',
              fontSize: '0.9rem',
            }}>
              ⚠️ {error}
            </div>
          )}
        </div>

        {/* Results Area */}
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Detection Results</h3>
            {result && (
              <span className="badge badge--success">{result.count} objects found</span>
            )}
          </div>

          {!result && !loading ? (
            <div style={{
              textAlign: 'center', padding: 'var(--space-2xl)',
              color: 'var(--text-muted)',
            }}>
              <p style={{ fontSize: '3rem', marginBottom: 'var(--space-md)' }}>🔍</p>
              <p>Upload an image and click "Run Detection"</p>
              <p style={{ fontSize: '0.85rem', marginTop: 'var(--space-sm)' }}>
                Results will appear here
              </p>
            </div>
          ) : loading ? (
            <div className="loading-overlay">
              <div className="spinner" />
              <span>Running YOLOv8 detection...</span>
            </div>
          ) : result ? (
            <div className="animate-fade-in">
              {/* Annotated Image */}
              {result.result_url && (
                <div className="detection-viewer" style={{ marginBottom: 'var(--space-md)' }}>
                  <img src={result.result_url} alt="Detection result" />
                </div>
              )}

              {/* Stats */}
              <div className="detection-results">
                <div style={{
                  padding: 'var(--space-md)', background: 'var(--bg-glass)',
                  borderRadius: 'var(--radius-sm)',
                }}>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Inference Time</div>
                  <div style={{ fontSize: '1.3rem', fontWeight: 700 }}>
                    {result.inference_time_ms?.toFixed(1)}ms
                  </div>
                </div>
                <div style={{
                  padding: 'var(--space-md)', background: 'var(--bg-glass)',
                  borderRadius: 'var(--radius-sm)',
                }}>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>FPS</div>
                  <div style={{ fontSize: '1.3rem', fontWeight: 700 }}>{result.fps}</div>
                </div>
              </div>

              {/* Class Counts */}
              {result.class_counts && Object.keys(result.class_counts).length > 0 && (
                <div style={{ marginTop: 'var(--space-md)' }}>
                  <h4 style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: 'var(--space-sm)' }}>
                    Detected Classes
                  </h4>
                  <div style={{ display: 'flex', gap: 'var(--space-sm)', flexWrap: 'wrap' }}>
                    {Object.entries(result.class_counts).map(([cls, count]) => (
                      <span key={cls} className="badge badge--info" style={{ fontSize: '0.85rem', padding: '6px 14px' }}>
                        {cls}: {count}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Detections Table */}
              {result.detections?.length > 0 && (
                <div style={{ marginTop: 'var(--space-md)' }}>
                  <h4 style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: 'var(--space-sm)' }}>
                    Detection Details
                  </h4>
                  <div className="table-container" style={{ maxHeight: 300, overflowY: 'auto' }}>
                    <table>
                      <thead>
                        <tr>
                          <th>#</th>
                          <th>Class</th>
                          <th>Confidence</th>
                          <th>Size</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.detections.map((det, idx) => (
                          <tr key={idx}>
                            <td>{idx + 1}</td>
                            <td><span className="badge badge--info">{det.class_name}</span></td>
                            <td>{(det.confidence * 100).toFixed(1)}%</td>
                            <td>{det.width?.toFixed(0)}×{det.height?.toFixed(0)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
