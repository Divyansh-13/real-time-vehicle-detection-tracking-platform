import React, { useEffect, useState } from 'react';
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts';
import { analyticsAPI } from '../services/api.js';

const CHART_COLORS = ['#00d2ff', '#7928ca', '#38ef7d', '#f5576c', '#f5af19'];

export default function Analytics() {
  const [summary, setSummary] = useState(null);
  const [timeline, setTimeline] = useState([]);
  const [classData, setClassData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchData() {
      try {
        const [sumRes, timeRes, classRes] = await Promise.all([
          analyticsAPI.summary(),
          analyticsAPI.timeline(30),
          analyticsAPI.classes(),
        ]);
        setSummary(sumRes.data);
        setTimeline(timeRes.data.entries || []);
        setClassData(classRes.data);
      } catch (err) {
        console.log('Analytics API not available');
        setSummary({
          total_images_processed: 0, total_videos_processed: 0,
          total_detections: 0, avg_confidence: 0,
        });
        setTimeline([]);
        setClassData({ classes: [{ name: 'car', count: 0 }, { name: 'minivan', count: 0 }] });
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
        <span>Loading analytics...</span>
      </div>
    );
  }

  const pieData = classData?.classes?.map((c) => ({
    name: c.name,
    value: c.count,
    percentage: c.percentage,
  })) || [];

  const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null;
    return (
      <div style={{
        background: 'var(--bg-secondary)', border: '1px solid var(--border-color)',
        borderRadius: 'var(--radius-sm)', padding: 'var(--space-sm) var(--space-md)',
        boxShadow: 'var(--shadow-lg)',
      }}>
        <p style={{ fontWeight: 600, marginBottom: 4 }}>{label}</p>
        {payload.map((entry, idx) => (
          <p key={idx} style={{ color: entry.color, fontSize: '0.85rem' }}>
            {entry.name}: {entry.value}
          </p>
        ))}
      </div>
    );
  };

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Analytics</h1>
        <p className="page-subtitle">Detection trends, class distribution, and system performance</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 'var(--space-lg)', marginBottom: 'var(--space-lg)' }}>
        {/* Timeline Chart */}
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Detection Timeline (Last 30 Days)</h3>
          </div>
          {timeline.length === 0 ? (
            <div style={{ textAlign: 'center', padding: 'var(--space-2xl)', color: 'var(--text-muted)' }}>
              <p style={{ fontSize: '2rem' }}>📊</p>
              <p>No data yet. Start detecting to see trends!</p>
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={timeline}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip content={<CustomTooltip />} />
                <Legend />
                <Bar dataKey="cars" name="Cars" fill="#00d2ff" radius={[4, 4, 0, 0]} />
                <Bar dataKey="minivans" name="Minivans" fill="#7928ca" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>

        {/* Pie Chart */}
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Class Distribution</h3>
          </div>
          {pieData.some(d => d.value > 0) ? (
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={5}
                  dataKey="value"
                  label={({ name, percentage }) => `${name} (${percentage}%)`}
                >
                  {pieData.map((entry, idx) => (
                    <Cell key={idx} fill={CHART_COLORS[idx % CHART_COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip content={<CustomTooltip />} />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <div style={{ textAlign: 'center', padding: 'var(--space-2xl)', color: 'var(--text-muted)' }}>
              <p style={{ fontSize: '2rem' }}>🍩</p>
              <p>No class data yet</p>
            </div>
          )}
        </div>
      </div>

      {/* Performance Metrics */}
      <div className="card">
        <div className="card-header">
          <h3 className="card-title">System Performance</h3>
        </div>
        <div className="stats-grid stagger-children" style={{ marginTop: 0 }}>
          <div style={{
            padding: 'var(--space-lg)', background: 'var(--bg-glass)',
            borderRadius: 'var(--radius-md)', textAlign: 'center',
          }}>
            <div style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--text-accent)' }}>
              {summary?.total_detections || 0}
            </div>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Total Detections</div>
          </div>
          <div style={{
            padding: 'var(--space-lg)', background: 'var(--bg-glass)',
            borderRadius: 'var(--radius-md)', textAlign: 'center',
          }}>
            <div style={{ fontSize: '2rem', fontWeight: 800, color: '#38ef7d' }}>
              {((summary?.avg_confidence || 0) * 100).toFixed(1)}%
            </div>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Avg Confidence</div>
          </div>
          <div style={{
            padding: 'var(--space-lg)', background: 'var(--bg-glass)',
            borderRadius: 'var(--radius-md)', textAlign: 'center',
          }}>
            <div style={{ fontSize: '2rem', fontWeight: 800, color: '#7928ca' }}>
              {summary?.total_images_processed || 0}
            </div>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Images Processed</div>
          </div>
          <div style={{
            padding: 'var(--space-lg)', background: 'var(--bg-glass)',
            borderRadius: 'var(--radius-md)', textAlign: 'center',
          }}>
            <div style={{ fontSize: '2rem', fontWeight: 800, color: '#f5af19' }}>
              {summary?.total_videos_processed || 0}
            </div>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Videos Processed</div>
          </div>
        </div>
      </div>
    </div>
  );
}
