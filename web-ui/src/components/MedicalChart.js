"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend
} from 'recharts';

export default function MedicalChart({ data, label }) {
  if (!data || data.length === 0) return null;

  return (
    <div style={{ 
      width: '100%', 
      height: 300, 
      marginTop: '1.5rem', 
      padding: '1rem', 
      background: 'rgba(255,255,255,0.03)', 
      borderRadius: '8px',
      border: '1px solid rgba(255,255,255,0.1)'
    }}>
      <h4 style={{ marginBottom: '1rem', fontSize: '0.9rem', color: '#39d353' }}>
        📊 Évolution : {label}
      </h4>
      <ResponsiveContainer width="100%" height="80%">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
          <XAxis 
            dataKey="date" 
            stroke="#8b949e" 
            fontSize={10} 
            tickFormatter={(str) => str.split('-').slice(1).join('/')}
          />
          <YAxis stroke="#8b949e" fontSize={10} />
          <Tooltip 
            contentStyle={{ backgroundColor: '#161b22', border: '1px solid #30363d', fontSize: '0.8rem' }}
            itemStyle={{ color: '#39d353' }}
          />
          <Legend wrapperStyle={{ fontSize: '0.8rem' }} />
          <Line 
            type="monotone" 
            dataKey="value" 
            name={label} 
            stroke="#2f81f7" 
            strokeWidth={2}
            dot={{ r: 4, fill: '#39d353' }}
            activeDot={{ r: 6 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
