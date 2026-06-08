import React, { useEffect, useState } from 'react';
import { analyticsAPI } from '../services/api.js';

function StatCard({ icon, value, label, variant = 'blue' }) {
  return (
    <div className={`stat-card stat-card--${variant}`}>
      <div className="stat-card-icon">{icon}</div>
      <div className="stat-card-value">{value}</div>
      <div className="stat-card-label">{label}</div>
    </div>
  );
}

export default function Dashboard() {
  const [summary, setSummary] = useState(null);
  const [recent, setRecent] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchData() {
      try {
        const [sumRes, recentRes] = await Promise.all([
          analyticsAPI.summary(),
          analyticsAPI.recent(5),
        ]);
        setSummary(sumRes.data);
        setRecent(recentRes.data);
      } catch (err) {
        console.log('API not available yet — showing placeholder data');
        setSummary({
          total_images_processed: 0,
          total_videos_processed: 0,
          total_detections: 0,
          total_unique_vehicles: 0,
          car_count: 0,
          minivan_count: 0,
          avg_confidence: 0,
          avg_inference_time_ms: 0,
        });
        setRecent([]);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="loading-overlay">
        <div className="spinner" />
        <span>Loading dashboard...</span>
      </div>
    );
  }

  const s = summary || {};

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Dashboard</h1>
        <p className="page-subtitle">
          Real-time overview of vehicle detection & tracking system
        </p>
      </div>

      <div className="stats-grid stagger-children">
        <StatCard
          icon="🖼️"
          value={s.total_images_processed}
          label="Images Processed"
          variant="blue"
        />
        <StatCard
          icon="🎬"
          value={s.total_videos_processed}
          label="Videos Processed"
          variant="purple"
        />
        <StatCard
          icon="🚗"
          value={s.total_detections}
          label="Total Detections"
          variant="green"
        />
        <StatCard
          icon="🎯"
          value={s.total_unique_vehicles}
          label="Unique Vehicles"
          variant="pink"
        />
        <StatCard
          icon="📊"
          value={`${(s.avg_confidence * 100).toFixed(1)}%`}
          label="Avg Confidence"
          variant="warm"
        />
        <StatCard
          icon="⚡"
          value={`${s.avg_inference_time_ms.toFixed(0)}ms`}
          label="Avg Inference Time"
          variant="blue"
        />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-lg)' }}>
        {/* Class Distribution */}
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Class Distribution</h3>
          </div>
          <div style={{ display: 'flex', gap: 'var(--space-lg)', alignItems: 'center' }}>
            <div style={{ flex: 1 }}>
              <div style={{ marginBottom: 'var(--space-md)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                  <span style={{ fontSize: '0.9rem' }}>🚗 Cars</span>
                  <span style={{ fontWeight: 700 }}>{s.car_count}</span>
                </div>
                <div style={{
                  height: 8, background: 'var(--bg-glass)',
                  borderRadius: 'var(--radius-full)', overflow: 'hidden'
                }}>
                  <div style={{
                    height: '100%',
                    width: `${s.car_count / Math.max(s.car_count + s.minivan_count, 1) * 100}%`,
                    background: 'var(--gradient-primary)',
                    borderRadius: 'var(--radius-full)',
                    transition: 'width 0.8s ease',
                  }} />
                </div>
              </div>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                  <span style={{ fontSize: '0.9rem' }}>🚐 Minivans</span>
                  <span style={{ fontWeight: 700 }}>{s.minivan_count}</span>
                </div>
                <div style={{
                  height: 8, background: 'var(--bg-glass)',
                  borderRadius: 'var(--radius-full)', overflow: 'hidden'
                }}>
                  <div style={{
                    height: '100%',
                    width: `${s.minivan_count / Math.max(s.car_count + s.minivan_count, 1) * 100}%`,
                    background: 'var(--gradient-secondary)',
                    borderRadius: 'var(--radius-full)',
                    transition: 'width 0.8s ease',
                  }} />
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Recent Activity */}
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Recent Detections</h3>
          </div>
          {recent.length === 0 ? (
            <div style={{
              textAlign: 'center', padding: 'var(--space-xl)',
              color: 'var(--text-muted)'
            }}>
              <p style={{ fontSize: '2rem', marginBottom: 'var(--space-sm)' }}>🔍</p>
              <p>No detections yet.</p>
              <p style={{ fontSize: '0.85rem' }}>Upload an image to get started!</p>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-sm)' }}>
              {recent.map((item, idx) => (
                <div key={idx} style={{
                  display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                  padding: 'var(--space-sm) var(--space-md)',
                  background: 'var(--bg-glass)',
                  borderRadius: 'var(--radius-sm)',
                }}>
                  <div>
                    <span style={{ fontSize: '0.85rem' }}>
                      {item.source_type === 'image' ? '🖼️' : '🎬'}{' '}
                      {item.source_filename?.slice(0, 30) || 'Unknown'}
                    </span>
                  </div>
                  <div style={{ display: 'flex', gap: 'var(--space-sm)', alignItems: 'center' }}>
                    <span className="badge badge--info">{item.total_objects} objects</span>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                      {item.inference_time_ms?.toFixed(0)}ms
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
