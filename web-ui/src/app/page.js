"use client";

import { useState, useEffect, useRef } from "react";
import { v4 as uuidv4 } from "uuid";
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import MedicalChart from "@/components/MedicalChart";
import Dashboard from "@/components/Dashboard";
import { 
  Plus, 
  Send, 
  History, 
  Activity, 
  FileText, 
  User, 
  Bot, 
  Link as LinkIcon, 
  ChevronRight,
  Database,
  BarChart2,
  MessageSquare
} from "lucide-react";

export default function Home() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [conversations, setConversations] = useState([]);
  const [activeConvId, setActiveConvId] = useState(null);
  const [view, setView] = useState("chat"); // "chat" or "dashboard"
  const messagesEndRef = useRef(null);

  const API_BASE = "http://localhost:8000";

  useEffect(() => {
    fetchConversations();
    const savedId = localStorage.getItem("current_conv_id");
    if (savedId) {
      loadConversation(savedId);
    } else {
      startNewChat();
    }
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, view]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const fetchConversations = async () => {
    try {
      const res = await fetch(`${API_BASE}/conversations`);
      const data = await res.json();
      setConversations(data);
    } catch (e) {
      console.error("Failed to fetch history", e);
    }
  };

  const startNewChat = () => {
    const newId = uuidv4();
    setActiveConvId(newId);
    setMessages([]);
    setView("chat");
    localStorage.setItem("current_conv_id", newId);
  };

  const loadConversation = async (id) => {
    setLoading(true);
    setActiveConvId(id);
    setView("chat");
    localStorage.setItem("current_conv_id", id);
    try {
      const res = await fetch(`${API_BASE}/conversations/${id}`);
      const data = await res.json();
      setMessages(data);
    } catch (e) {
      console.error("Failed to load conversation", e);
    } finally {
      setLoading(false);
    }
  };

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userQuery = input;
    setInput("");
    
    const userMsg = { role: "user", content: userQuery };
    setMessages(prev => [...prev, userMsg]);
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: userQuery,
          conversation_id: activeConvId
        })
      });

      if (!res.ok) throw new Error("API Error");

      const data = await res.json();
      const assistantMsg = { 
        role: "assistant", 
        content: data.answer,
        metadata: data.sources 
      };
      
      setMessages(prev => [...prev, assistantMsg]);
      fetchConversations();
    } catch (e) {
      setMessages(prev => [...prev, { role: "assistant", content: "⚠️ Erreur : " + e.message }]);
    } finally {
      setLoading(false);
    }
  };

  const parseChartData = (content) => {
    const chartRegex = /\[CHART_DATA:\s*(\{.*?\})\]/s;
    const match = content.match(chartRegex);
    if (match) {
      try {
        const chartData = JSON.parse(match[1]);
        const cleanContent = content.replace(chartRegex, '').trim();
        return { chartData, cleanContent };
      } catch (e) {
        console.error("Failed to parse chart data", e);
      }
    }
    return { chartData: null, cleanContent: content };
  };

  return (
    <div className="app-container">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="logo-container">
          <div className="logo-icon">
            <Activity size={20} color="white" />
          </div>
          <span className="logo-text">Medical RAG</span>
        </div>

        <button className="new-chat-btn" onClick={startNewChat}>
          <Plus size={18} /> Nouveau Chat
        </button>

        <div className="nav-group">
          <div 
            className={`nav-item ${view === "chat" ? "active" : ""}`}
            onClick={() => setView("chat")}
          >
            <MessageSquare size={18} /> Chat Assistant
          </div>
          <div 
            className={`nav-item ${view === "dashboard" ? "active" : ""}`}
            onClick={() => setView("dashboard")}
          >
            <BarChart2 size={18} /> Tableau de Bord
          </div>
        </div>

        <div className="history-label" style={{marginTop: '2rem'}}>
          <History size={14} style={{display: 'inline', marginRight: '5px'}} /> Historique
        </div>
        
        <div className="history-list">
          {conversations.map(conv => (
            <div 
              key={conv.id} 
              className={`history-item ${activeConvId === conv.id && view === "chat" ? 'active' : ''}`}
              onClick={() => loadConversation(conv.id)}
            >
              <FileText size={16} />
              <span>{conv.title}</span>
            </div>
          ))}
        </div>

        <div style={{marginTop: 'auto', padding: '0.5rem', fontSize: '0.7rem', color: '#484f58', borderTop: '1px solid var(--border-color)'}}>
          <Database size={12} style={{display: 'inline', marginRight: '5px'}} />
          Instance Locale : Ollama
        </div>
      </aside>

      {/* Main Area */}
      <main className="chat-main">
        <header className="chat-header">
          <div className="header-info">
            <div className="header-title">
              {view === "chat" ? "Assistant Clinique Intelligent" : "Analyse Statistique Globale"}
            </div>
            <div className="header-status">
              <span className="pulse-dot"></span> Système Opérationnel
            </div>
          </div>
          <div className="badge">Provenance Stricte</div>
        </header>

        {view === "chat" ? (
          <>
            <div className="messages-container">
              {messages.length === 0 && !loading && (
                <div className="hero-state">
                  <div className="hero-icon">
                    <Bot size={48} color="var(--accent-primary)" />
                  </div>
                  <h2 style={{fontFamily: 'var(--font-outfit)', fontSize: '2rem', marginBottom: '1rem'}}>
                    Analyse de Rapports Médicaux
                  </h2>
                  <p style={{color: 'var(--text-secondary)', maxWidth: '500px', margin: '0 auto 2rem'}}>
                    Posez des questions sur vos analyses biologiques, suivez l'évolution de vos constantes ou identifiez les anomalies détectées dans vos documents.
                  </p>
                  <div className="suggestion-grid">
                    <div className="suggestion-card" onClick={() => setInput("Quelle est l'évolution de mon taux de cholestérol ?")}>
                      "Évolution du cholestérol"
                    </div>
                    <div className="suggestion-card" onClick={() => setInput("Analyse mes derniers résultats de glycémie.")}>
                      "Analyse de glycémie"
                    </div>
                    <div className="suggestion-card" onClick={() => setInput("Y a-t-il des anomalies dans mon dernier rapport ?")}>
                      "Détection d'anomalies"
                    </div>
                  </div>
                </div>
              )}

              {messages.map((msg, i) => {
                const { chartData, cleanContent } = msg.role === 'assistant' ? parseChartData(msg.content) : { chartData: null, cleanContent: msg.content };
                
                return (
                  <div key={i} className={`message-row ${msg.role}`}>
                    <div className="avatar">
                      {msg.role === 'user' ? <User size={20} /> : <Bot size={20} />}
                    </div>
                    <div className="message-bubble">
                      <div className="content">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {cleanContent}
                        </ReactMarkdown>
                      </div>
                      
                      {chartData && (
                        <MedicalChart data={chartData.data} label={chartData.label} />
                      )}

                      {msg.metadata && msg.metadata.length > 0 && (
                        <div className="sources-section">
                          <div className="source-title">
                            <LinkIcon size={12} /> Sources Documentaires
                          </div>
                          <div className="source-chips">
                            {msg.metadata.map((s, idx) => {
                              const pdfName = s.doc_id.includes('.pdf') ? s.doc_id : `${s.doc_id}.pdf`;
                              const safeUrl = `${API_BASE}/pdf/${encodeURIComponent(pdfName)}`;
                              return (
                                <a key={idx} href={safeUrl} target="_blank" rel="noopener noreferrer" className="source-chip">
                                  {s.doc_id} {s.page_number ? `• Page ${s.page_number}` : ''}
                                  <ChevronRight size={10} />
                                </a>
                              );
                            })}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
              
              {loading && (
                <div className="message-row assistant">
                  <div className="avatar"><Bot size={20} /></div>
                  <div className="message-bubble">
                    <div className="typing-indicator">
                      <span></span><span></span><span></span>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            <div className="input-area">
              <div className="input-wrapper">
                <textarea 
                  className="input-field"
                  rows="1"
                  placeholder="Décrivez votre besoin médical (ex: évolution du fer...)"
                  value={input}
                  onChange={(e) => {
                    setInput(e.target.value);
                    e.target.style.height = 'auto';
                    e.target.style.height = e.target.scrollHeight + 'px';
                  }}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleSend();
                    }
                  }}
                />
                <button 
                  className="send-button" 
                  onClick={handleSend}
                  disabled={loading || !input.trim()}
                >
                  <Send size={18} />
                </button>
              </div>
              <div style={{textAlign: 'center', marginTop: '0.75rem', fontSize: '0.65rem', color: 'var(--text-secondary)'}}>
                Assistant IA spécialisé. Vérifiez toujours auprès d'un professionnel de santé.
              </div>
            </div>
          </>
        ) : (
          <Dashboard apiBase={API_BASE} />
        )}
      </main>

      <style jsx>{`
        .pulse-dot {
          width: 8px;
          height: 8px;
          background: var(--accent-success);
          border-radius: 50%;
          display: inline-block;
          box-shadow: 0 0 0 rgba(57, 211, 83, 0.4);
          animation: pulse 2s infinite;
        }

        @keyframes pulse {
          0% { box-shadow: 0 0 0 0 rgba(57, 211, 83, 0.7); }
          70% { box-shadow: 0 0 0 10px rgba(57, 211, 83, 0); }
          100% { box-shadow: 0 0 0 0 rgba(57, 211, 83, 0); }
        }

        .hero-state {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          text-align: center;
          margin-top: 5vh;
        }

        .hero-icon {
          width: 80px;
          height: 80px;
          background: rgba(47, 129, 247, 0.1);
          border-radius: 20px;
          display: flex;
          align-items: center;
          justify-content: center;
          margin-bottom: 2rem;
        }

        .suggestion-grid {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 1rem;
          max-width: 600px;
        }

        .suggestion-card {
          padding: 1rem;
          background: var(--bg-secondary);
          border: 1px solid var(--border-color);
          border-radius: 12px;
          font-size: 0.85rem;
          cursor: pointer;
          transition: all 0.2s;
        }

        .suggestion-card:hover {
          background: var(--bg-tertiary);
          border-color: var(--accent-primary);
          transform: translateY(-2px);
        }

        .badge {
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid var(--border-color);
          padding: 4px 12px;
          border-radius: 20px;
          font-size: 0.7rem;
          color: var(--text-secondary);
        }

        .nav-group {
          display: flex;
          flex-direction: column;
          gap: 0.25rem;
        }

        .nav-item {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          padding: 0.8rem 1rem;
          border-radius: 10px;
          cursor: pointer;
          font-size: 0.9rem;
          color: var(--text-secondary);
          transition: all 0.2s;
        }

        .nav-item:hover {
          background: var(--bg-tertiary);
          color: var(--text-primary);
        }

        .nav-item.active {
          background: rgba(47, 129, 247, 0.1);
          color: var(--accent-primary);
        }
      `}</style>
    </div>
  );
}
