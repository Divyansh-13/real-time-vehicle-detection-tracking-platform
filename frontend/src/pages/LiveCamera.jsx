import React, { useState, useRef, useCallback, useEffect } from 'react';

const WS_URL = 'ws://localhost:8000/api/v1/ws/detect';

const CLASS_COLORS = {
  car: '#00d2ff',
  minivan: '#7928ca',
};

export default function LiveCamera() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const animFrameRef = useRef(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [stats, setStats] = useState({ fps: 0, count: 0, inference_time_ms: 0, class_counts: {} });
  const [error, setError] = useState(null);
  const [confidence, setConfidence] = useState(0.25);
  const [cameraReady, setCameraReady] = useState(false);

  // Start camera
  const startCamera = useCallback(async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720, facingMode: 'environment' },
        audio: false,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current.play();
          setCameraReady(true);
        };
      }
    } catch (err) {
      setError('Camera access denied. Please allow camera permissions.');
      console.error('Camera error:', err);
    }
  }, []);

  // Stop camera
  const stopCamera = useCallback(() => {
    if (videoRef.current?.srcObject) {
      videoRef.current.srcObject.getTracks().forEach((t) => t.stop());
      videoRef.current.srcObject = null;
    }
    setCameraReady(false);
  }, []);

  // Connect WebSocket and start detection loop
  const startDetection = useCallback(() => {
    if (!cameraReady) return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setIsStreaming(true);
      setError(null);
      sendFrame();
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.error) {
        setError(data.error);
        return;
      }
      setStats(data);
      drawDetections(data.detections);
      // Send next frame after receiving response (throttle to detection speed)
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        requestAnimationFrame(() => sendFrame());
      }
    };

    ws.onerror = () => setError('WebSocket connection failed. Is the backend running?');
    ws.onclose = () => {
      setIsStreaming(false);
      clearCanvas();
    };
  }, [cameraReady, confidence]);

  // Stop detection
  const stopDetection = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
    }
    setIsStreaming(false);
    clearCanvas();
  }, []);

  // Capture frame and send via WebSocket
  const sendFrame = useCallback(() => {
    if (!videoRef.current || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

    const video = videoRef.current;
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = video.videoWidth || 640;
    tempCanvas.height = video.videoHeight || 480;
    const ctx = tempCanvas.getContext('2d');
    ctx.drawImage(video, 0, 0);

    const jpeg = tempCanvas.toDataURL('image/jpeg', 0.7);
    wsRef.current.send(JSON.stringify({ image: jpeg, conf: confidence }));
  }, [confidence]);

  // Draw bounding boxes on overlay canvas
  const drawDetections = useCallback((detections) => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (const det of detections) {
      const color = CLASS_COLORS[det.class_name] || '#00ff80';
      const w = det.x2 - det.x1;
      const h = det.y2 - det.y1;

      // Box
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(det.x1, det.y1, w, h);

      // Label background
      const label = `${det.class_name} ${(det.confidence * 100).toFixed(0)}%`;
      ctx.font = 'bold 14px Inter, sans-serif';
      const textWidth = ctx.measureText(label).width;
      ctx.fillStyle = color;
      ctx.fillRect(det.x1, det.y1 - 22, textWidth + 10, 22);

      // Label text
      ctx.fillStyle = '#fff';
      ctx.fillText(label, det.x1 + 5, det.y1 - 6);
    }
  }, []);

  const clearCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  }, []);

  // Auto-start camera on mount
  useEffect(() => {
    startCamera();
    return () => {
      stopDetection();
      stopCamera();
    };
  }, []);

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Live Camera Detection</h1>
        <p className="page-subtitle">Real-time vehicle detection using your webcam</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 'var(--space-lg)' }}>
        {/* Camera Feed */}
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">
              {isStreaming ? (
                <><span style={{ color: 'var(--color-error)', animation: 'pulse 1.5s infinite' }}>●</span> Live Detection</>
              ) : (
                '📷 Camera Feed'
              )}
            </h3>
            <div style={{ display: 'flex', gap: 'var(--space-sm)' }}>
              {!isStreaming ? (
                <button className="btn btn-primary" onClick={startDetection} disabled={!cameraReady}>
                  🎯 Start Detection
                </button>
              ) : (
                <button className="btn btn-secondary" onClick={stopDetection}>
                  ⏹ Stop
                </button>
              )}
            </div>
          </div>

          <div style={{ position: 'relative', width: '100%', borderRadius: 'var(--radius-md)', overflow: 'hidden', background: '#000' }}>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              style={{ width: '100%', display: 'block' }}
            />
            <canvas
              ref={canvasRef}
              style={{
                position: 'absolute', top: 0, left: 0,
                width: '100%', height: '100%',
                pointerEvents: 'none',
              }}
            />
            {!cameraReady && (
              <div style={{
                position: 'absolute', inset: 0,
                display: 'flex', flexDirection: 'column',
                alignItems: 'center', justifyContent: 'center',
                color: 'var(--text-muted)',
              }}>
                <div className="spinner" style={{ marginBottom: 'var(--space-md)' }} />
                <span>Starting camera...</span>
              </div>
            )}
          </div>

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

        {/* Controls & Stats */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
          {/* Controls */}
          <div className="card">
            <div className="card-header">
              <h3 className="card-title">Controls</h3>
            </div>
            <div style={{ marginBottom: 'var(--space-md)' }}>
              <label style={{ fontSize: '0.85rem', color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>
                Confidence: {(confidence * 100).toFixed(0)}%
              </label>
              <input
                type="range"
                min="0.1"
                max="0.9"
                step="0.05"
                value={confidence}
                onChange={(e) => setConfidence(parseFloat(e.target.value))}
                style={{ width: '100%', accentColor: 'var(--color-primary)' }}
              />
            </div>
            <div style={{ display: 'flex', gap: 'var(--space-sm)' }}>
              <button className="btn btn-secondary btn-sm" onClick={() => { stopCamera(); startCamera(); }} style={{ flex: 1 }}>
                🔄 Reset Camera
              </button>
            </div>
          </div>

          {/* Live Stats */}
          <div className="card">
            <div className="card-header">
              <h3 className="card-title">Live Stats</h3>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-sm)' }}>
              <div style={{
                display: 'flex', justifyContent: 'space-between', padding: 'var(--space-sm) var(--space-md)',
                background: 'var(--bg-glass)', borderRadius: 'var(--radius-sm)',
              }}>
                <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>FPS</span>
                <span style={{ fontWeight: 700, color: 'var(--text-accent)' }}>{stats.fps}</span>
              </div>
              <div style={{
                display: 'flex', justifyContent: 'space-between', padding: 'var(--space-sm) var(--space-md)',
                background: 'var(--bg-glass)', borderRadius: 'var(--radius-sm)',
              }}>
                <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>Inference</span>
                <span style={{ fontWeight: 700 }}>{stats.inference_time_ms}ms</span>
              </div>
              <div style={{
                display: 'flex', justifyContent: 'space-between', padding: 'var(--space-sm) var(--space-md)',
                background: 'var(--bg-glass)', borderRadius: 'var(--radius-sm)',
              }}>
                <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>Objects</span>
                <span style={{ fontWeight: 700, color: '#38ef7d' }}>{stats.count}</span>
              </div>
              <div style={{
                display: 'flex', justifyContent: 'space-between', padding: 'var(--space-sm) var(--space-md)',
                background: 'var(--bg-glass)', borderRadius: 'var(--radius-sm)',
              }}>
                <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>Frame</span>
                <span style={{ fontWeight: 700 }}>#{stats.frame || 0}</span>
              </div>
            </div>
          </div>

          {/* Class Breakdown */}
          <div className="card">
            <div className="card-header">
              <h3 className="card-title">Detected</h3>
            </div>
            {stats.class_counts && Object.keys(stats.class_counts).length > 0 ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-sm)' }}>
                {Object.entries(stats.class_counts).map(([cls, count]) => (
                  <div key={cls} style={{
                    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                    padding: 'var(--space-sm) var(--space-md)',
                    background: 'var(--bg-glass)', borderRadius: 'var(--radius-sm)',
                  }}>
                    <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span style={{
                        width: 10, height: 10, borderRadius: '50%',
                        background: CLASS_COLORS[cls] || '#00ff80',
                      }} />
                      {cls}
                    </span>
                    <span style={{ fontWeight: 800, fontSize: '1.1rem' }}>{count}</span>
                  </div>
                ))}
              </div>
            ) : (
              <div style={{ textAlign: 'center', padding: 'var(--space-md)', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                {isStreaming ? 'Waiting for detections...' : 'Start detection to see results'}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
