import React from 'react';
import { BrowserRouter, Routes, Route, NavLink, useLocation } from 'react-router-dom';
import Dashboard from './pages/Dashboard.jsx';
import ImageDetection from './pages/ImageDetection.jsx';
import VideoTracking from './pages/VideoTracking.jsx';
import Analytics from './pages/Analytics.jsx';
import LiveCamera from './pages/LiveCamera.jsx';

const navItems = [
  { path: '/', label: 'Dashboard', icon: '📊' },
  { path: '/detect', label: 'Image Detection', icon: '🖼️' },
  { path: '/track', label: 'Video Tracking', icon: '🎬' },
  { path: '/live', label: 'Live Camera', icon: '📹' },
  { path: '/analytics', label: 'Analytics', icon: '📈' },
];

function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="sidebar-logo">
          <div className="sidebar-logo-icon">🚗</div>
          <span className="sidebar-logo-text">VehicleAI</span>
        </div>
      </div>
      <nav className="sidebar-nav">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            end={item.path === '/'}
            className={({ isActive }) =>
              `nav-link ${isActive ? 'active' : ''}`
            }
          >
            <span className="nav-link-icon">{item.icon}</span>
            <span>{item.label}</span>
          </NavLink>
        ))}
      </nav>
      <div style={{ padding: 'var(--space-md)', borderTop: '1px solid var(--border-color)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{
            width: 8, height: 8,
            borderRadius: '50%',
            background: 'var(--color-success)',
            boxShadow: '0 0 8px var(--color-success)',
          }} />
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>System Online</span>
        </div>
      </div>
    </aside>
  );
}

function PageTransition({ children }) {
  const location = useLocation();
  return (
    <div key={location.pathname} className="animate-fade-in">
      {children}
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-layout">
        <Sidebar />
        <main className="main-content">
          <PageTransition>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/detect" element={<ImageDetection />} />
              <Route path="/track" element={<VideoTracking />} />
              <Route path="/live" element={<LiveCamera />} />
              <Route path="/analytics" element={<Analytics />} />
            </Routes>
          </PageTransition>
        </main>
      </div>
    </BrowserRouter>
  );
}
