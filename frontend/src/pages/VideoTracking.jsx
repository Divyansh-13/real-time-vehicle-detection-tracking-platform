import React, { useState, useRef, useCallback } from 'react';
import { predictAPI } from '../services/api.js';

export default function VideoTracking() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [progress, setProgress] = useState(0);
  const fileInputRef = useRef(null);

  const handleFileSelect = useCallback((selectedFile) => {
    if (!selectedFile) return;
    setFile(selectedFile);
    setResult(null);
    setError(null);
    setProgress(0);
  }, []);

  const handleTrack = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);
      const response = await predictAPI.video(formData, {
        onUploadProgress: (e) => {
          if (e.total) setProgress(Math.round((e.loaded * 100) / e.total));
        },
      });
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Tracking failed. Is the backend running?');
    } finally {
      setLoading(false);
      setProgress(0);
    }
  };

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Video Tracking</h1>
        <p className="page-subtitle">Upload a video to track vehicles with ByteTrack</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-lg)' }}>
        {/* Upload */}
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Upload Video</h3>
          </div>

          <div
            className="upload-zone"
            onClick={() => fileInputRef.current?.click()}
          >
            <div className="upload-zone-icon">🎬</div>
            <div className="upload-zone-text">
              {file ? file.name : 'Drop a video here or click to browse'}
            </div>
            <div className="upload-zone-hint">Supports: MP4, AVI, MOV, WebM (max 100MB)</div>
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            style={{ display: 'none' }}
            onChange={(e) => handleFileSelect(e.target.files[0])}
          />

          {file && (
            <div style={{ marginTop: 'var(--space-md)', display: 'flex', gap: 'var(--space-sm)' }}>
              <button
                className="btn btn-primary btn-lg"
                onClick={handleTrack}
                disabled={loading}
                style={{ flex: 1 }}
              >
                {loading ? (
                  <>
                    <div className="spinner" style={{ width: 20, height: 20, borderWidth: 2 }} />
                    {progress > 0 ? `Uploading ${progress}%...` : 'Processing...'}
                  </>
                ) : (
                  <>🎯 Run Tracking</>
                )}
              </button>
              <button className="btn btn-secondary" onClick={() => {
                setFile(null); setResult(null);
              }}>Clear</button>
            </div>
          )}

          {error && (
            <div style={{
              marginTop: 'var(--space-md)', padding: 'var(--space-md)',
              background: 'rgba(245, 87, 108, 0.1)',
              border: '1px solid rgba(245, 87, 108, 0.3)',
              borderRadius: 'var(--radius-sm)', color: 'var(--color-error)',
            }}>
              ⚠️ {error}
            </div>
          )}
        </div>

        {/* Results */}
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Tracking Results</h3>
            {result && (
              <span className="badge badge--success">{result.unique_vehicles} vehicles tracked</span>
            )}
          </div>

          {!result && !loading ? (
            <div style={{
              textAlign: 'center', padding: 'var(--space-2xl)',
              color: 'var(--text-muted)',
            }}>
              <p style={{ fontSize: '3rem', marginBottom: 'var(--space-md)' }}>🎬</p>
              <p>Upload a video to start tracking</p>
            </div>
          ) : loading ? (
            <div className="loading-overlay">
              <div className="spinner" />
              <span>Processing video with ByteTrack...</span>
              <span style={{ fontSize: '0.85rem' }}>This may take a few minutes</span>
            </div>
          ) : result ? (
            <div className="animate-fade-in">
              {/* Summary Stats */}
              <div className="stats-grid" style={{ marginBottom: 'var(--space-md)' }}>
                <div style={{
                  padding: 'var(--space-md)', background: 'var(--bg-glass)',
                  borderRadius: 'var(--radius-sm)', textAlign: 'center',
                }}>
                  <div style={{ fontSize: '1.5rem', fontWeight: 800 }}>{result.total_frames}</div>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Frames</div>
                </div>
                <div style={{
                  padding: 'var(--space-md)', background: 'var(--bg-glass)',
                  borderRadius: 'var(--radius-sm)', textAlign: 'center',
                }}>
                  <div style={{ fontSize: '1.5rem', fontWeight: 800 }}>{result.unique_vehicles}</div>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Unique Vehicles</div>
                </div>
                <div style={{
                  padding: 'var(--space-md)', background: 'var(--bg-glass)',
                  borderRadius: 'var(--radius-sm)', textAlign: 'center',
                }}>
                  <div style={{ fontSize: '1.5rem', fontWeight: 800 }}>{result.avg_fps}</div>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Avg FPS</div>
                </div>
              </div>

              {/* Result Video */}
              {result.result_url && (
                <div style={{ marginBottom: 'var(--space-md)' }}>
                  <video
                    controls
                    style={{
                      width: '100%', borderRadius: 'var(--radius-md)',
                      border: '1px solid var(--border-color)',
                    }}
                    src={result.result_url}
                  />
                </div>
              )}

              {/* Tracks Table */}
              {result.tracks?.length > 0 && (
                <div>
                  <h4 style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: 'var(--space-sm)' }}>
                    Vehicle Tracks
                  </h4>
                  <div className="table-container">
                    <table>
                      <thead>
                        <tr>
                          <th>Track ID</th>
                          <th>Class</th>
                          <th>Frames</th>
                          <th>Confidence</th>
                          <th>Distance</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.tracks.map((track) => (
                          <tr key={track.track_id}>
                            <td><span style={{ fontWeight: 700 }}>#{track.track_id}</span></td>
                            <td><span className="badge badge--info">{track.class_name}</span></td>
                            <td>{track.total_frames}</td>
                            <td>{(track.avg_confidence * 100).toFixed(1)}%</td>
                            <td>{track.travel_distance?.toFixed(0)}px</td>
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
