"use client";

import { useState, useEffect } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar
} from "recharts";
import { Activity, AlertCircle, CheckCircle, FileText, Database } from "lucide-react";

const COLORS = ["#2f81f7", "#39d353", "#d29922", "#f85149", "#8b949e"];

export default function Dashboard({ apiBase }) {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      const res = await fetch(`${apiBase}/stats`);
      const data = await res.json();
      setStats(data);
    } catch (e) {
      console.error("Failed to fetch stats", e);
    } finally {
      setLoading(false);
    }
  };

  if (loading) return <div className="loading-container">Chargement des statistiques...</div>;
  if (!stats || stats.error) return <div className="error-container">Erreur lors de la récupération des statistiques.</div>;

  const pieData = stats.status_distribution.map(s => ({
    name: s.interpretation_status,
    value: s.count
  }));

  const barData = stats.top_markers.map(m => ({
    name: m.analyte_norm.substring(0, 15),
    count: m.count
  }));

  const radarData = stats.top_sections.map(s => ({
    subject: s.section_norm.substring(0, 12),
    A: s.count,
    fullMark: Math.max(...stats.top_sections.map(x => x.count))
  }));

  return (
    <div className="dashboard-container">
      <h2 className="dashboard-title">Tableau de Bord Clinique</h2>
      
      {/* Overview Cards */}
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-icon" style={{background: 'rgba(47, 129, 247, 0.1)', color: 'var(--accent-primary)'}}>
            <FileText size={20} />
          </div>
          <div className="stat-info">
            <span className="stat-label">Documents</span>
            <span className="stat-value">{stats.overview.total_docs}</span>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon" style={{background: 'rgba(57, 211, 83, 0.1)', color: 'var(--accent-success)'}}>
            <Activity size={20} />
          </div>
          <div className="stat-info">
            <span className="stat-label">Analyses</span>
            <span className="stat-value">{stats.overview.total_results}</span>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon" style={{background: 'rgba(210, 153, 34, 0.1)', color: 'var(--accent-warning)'}}>
            <AlertCircle size={20} />
          </div>
          <div className="stat-info">
            <span className="stat-label">Anomalies</span>
            <span className="stat-value">
              {stats.status_distribution.find(s => s.interpretation_status?.toLowerCase().includes('abnormal'))?.count || 0}
            </span>
          </div>
        </div>
      </div>

      <div className="charts-main-grid">
        {/* Status Distribution (Pie) */}
        <div className="chart-wrapper">
          <h3>Répartition des États</h3>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie
                data={pieData}
                innerRadius={60}
                outerRadius={80}
                paddingAngle={5}
                dataKey="value"
              >
                {pieData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip 
                contentStyle={{background: '#161b22', border: '1px solid #30363d'}}
                itemStyle={{color: '#fff'}}
              />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Top Markers (Bar) */}
        <div className="chart-wrapper">
          <h3>Top Marqueurs Bio</h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={barData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#30363d" vertical={false} />
              <XAxis dataKey="name" fontSize={10} stroke="#8b949e" />
              <YAxis fontSize={10} stroke="#8b949e" />
              <Tooltip 
                cursor={{fill: 'rgba(255,255,255,0.05)'}}
                contentStyle={{background: '#161b22', border: '1px solid #30363d'}}
              />
              <Bar dataKey="count" fill="var(--accent-primary)" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Section Distribution (Radar) */}
        <div className="chart-wrapper">
          <h3>Spécialités Dominantes</h3>
          <ResponsiveContainer width="100%" height={250}>
            <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarData}>
              <PolarGrid stroke="#30363d" />
              <PolarAngleAxis dataKey="subject" fontSize={10} stroke="#8b949e" />
              <PolarRadiusAxis fontSize={8} />
              <Radar
                name="Volume"
                dataKey="A"
                stroke="var(--accent-success)"
                fill="var(--accent-success)"
                fillOpacity={0.5}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <style jsx>{`
        .dashboard-container {
          padding: 2rem;
          height: 100%;
          overflow-y: auto;
          animation: fadeIn 0.5s ease;
        }

        .dashboard-title {
          font-family: var(--font-outfit);
          font-size: 1.5rem;
          margin-bottom: 2rem;
          background: linear-gradient(to right, #fff, var(--accent-primary));
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
        }

        .stats-grid {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 1.5rem;
          margin-bottom: 2rem;
        }

        .stat-card {
          background: var(--bg-secondary);
          border: 1px solid var(--border-color);
          border-radius: 16px;
          padding: 1.5rem;
          display: flex;
          align-items: center;
          gap: 1.25rem;
        }

        .stat-icon {
          width: 48px;
          height: 48px;
          border-radius: 12px;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .stat-info {
          display: flex;
          flex-direction: column;
        }

        .stat-label {
          font-size: 0.8rem;
          color: var(--text-secondary);
        }

        .stat-value {
          font-size: 1.5rem;
          font-weight: 700;
          font-family: var(--font-outfit);
        }

        .charts-main-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
          gap: 1.5rem;
        }

        .chart-wrapper {
          background: var(--bg-secondary);
          border: 1px solid var(--border-color);
          border-radius: 16px;
          padding: 1.5rem;
        }

        .chart-wrapper h3 {
          font-size: 0.9rem;
          font-weight: 600;
          color: var(--text-secondary);
          margin-bottom: 1.5rem;
          text-transform: uppercase;
          letter-spacing: 0.05em;
        }

        .loading-container, .error-container {
          display: flex;
          height: 100%;
          align-items: center;
          justify-content: center;
          color: var(--text-secondary);
        }

        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
      `}</style>
    </div>
  );
}
