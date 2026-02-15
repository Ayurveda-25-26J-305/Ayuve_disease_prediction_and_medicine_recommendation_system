"use client";

import { useState, useEffect, useRef } from "react";

const API_BASE_URL = "https://ayurvedic-qa.loca.lt";

interface Message {
  type: "question" | "answer";
  content: string;
  citations?: Citation[];
}

interface Citation {
  book: string;
  chapter: string;
  paragraph: string;
  formatted: string;
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState({
    docCount: "Loading...",
    modelName: "Loading...",
  });
  const [showWelcome, setShowWelcome] = useState(true);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    loadStats();
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const loadStats = async () => {
    try {
      console.log("Fetching stats from:", `${API_BASE_URL}/api/stats`);
      const response = await fetch(`${API_BASE_URL}/api/stats`, {
        mode: "cors",
        credentials: "include",
        headers: {
          "Bypass-Tunnel-Reminder": "true",
          "ngrok-skip-browser-warning": "true",
        },
      });
      console.log("Stats response status:", response.status);
      const data = await response.json();
      console.log("Stats data received:", data);
      
      if (data.success) {
        setStats({
          docCount: data.stats.total_documents.toLocaleString(),
          modelName: data.stats.model.split("/").pop(),
        });
        console.log("Stats updated successfully");
      } else {
        console.error("Stats API returned success=false:", data);
        setStats({ docCount: "Error", modelName: "Error" });
      }
    } catch (error) {
      console.error("Failed to load stats:", error);
      setStats({ docCount: "N/A", modelName: "Offline" });
    }
  };

  const askQuestion = async (question: string) => {
    if (!question.trim()) return;

    setShowWelcome(false);
    setMessages((prev) => [...prev, { type: "question", content: question }]);
    setInput("");
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/ask`, {
        method: "POST",
        mode: "cors",
        credentials: "include",
        headers: {
          "Content-Type": "application/json",
          "Bypass-Tunnel-Reminder": "true",
          "ngrok-skip-browser-warning": "true",
        },
        body: JSON.stringify({ question }),
      });

      const data = await response.json();
      console.log("Backend response:", data);

      if (data.success) {
        console.log("Setting answer:", data.answer);
        setMessages((prev) => [
          ...prev,
          {
            type: "answer",
            content: data.answer,
            citations: data.citations,
          },
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          {
            type: "answer",
            content: `❌ Error: ${data.error || "Unknown error"}`,
          },
        ]);
      }
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          type: "answer",
          content:
            "❌ Failed to connect to server. Make sure the backend is running.",
        },
      ]);
    }

    setLoading(false);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    askQuestion(input);
  };

  const askExample = (question: string) => {
    setInput(question);
    setTimeout(() => askQuestion(question), 100);
  };

  return (
    <div className="app">
      <div className="forest-bg"></div>

      <div className="container">
        {/* Header */}
        <header className="header">
          <div className="logo">
            <span className="leaf-icon">🌿</span>
            <h1>Ayurvedic Knowledge Assistant</h1>
          </div>
          <div className="subtitle">Ancient Wisdom, Modern Insights</div>
        </header>

        {/* Stats Bar */}
        <div className="stats-bar">
          <div className="stat-item">
            <span className="stat-icon">📚</span>
            <span className="stat-value">{stats.docCount}</span>
            <span className="stat-label">Documents</span>
          </div>
          <div className="stat-item">
            <span className="stat-icon">⚡</span>
            <span className="stat-value">{stats.modelName}</span>
            <span className="stat-label">Model</span>
          </div>
        </div>

        {/* Chat Container */}
        <div className="chat-container">
          {showWelcome && (
            <div className="welcome-message">
              <div className="welcome-icon">🙏</div>
              <h2>Welcome to Ayurvedic Knowledge Assistant</h2>
              <p>
                Ask me anything about Ayurvedic medicine, doshas, treatments,
                and ancient wisdom.
              </p>
              <div className="example-questions">
                <p className="example-label">Try asking:</p>
                <button
                  className="example-btn"
                  onClick={() => askExample("What causes Vata imbalance?")}
                >
                  What causes Vata imbalance?
                </button>
                <button
                  className="example-btn"
                  onClick={() => askExample("How to treat Pitta disorders?")}
                >
                  How to treat Pitta disorders?
                </button>
                <button
                  className="example-btn"
                  onClick={() =>
                    askExample("What are the properties of Kapha dosha?")
                  }
                >
                  What are the properties of Kapha dosha?
                </button>
              </div>
            </div>
          )}

          {messages.map((msg, idx) => (
            <MessageComponent key={idx} message={msg} />
          ))}

          {loading && (
            <div className="message message-answer">
              <div className="message-content">
                <div className="loading">
                  <div className="loading-dot"></div>
                  <div className="loading-dot"></div>
                  <div className="loading-dot"></div>
                </div>
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        {/* Input Area */}
        <div className="input-area">
          <form className="input-wrapper" onSubmit={handleSubmit}>
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask your Ayurvedic question..."
              disabled={loading}
            />
            <button type="submit" disabled={loading}>
              <span className="send-icon">{loading ? "⏳" : "🌿"}</span>
              <span>{loading ? "Thinking..." : "Ask"}</span>
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

function MessageComponent({ message }: { message: Message }) {
  return (
    <div className={`message message-${message.type}`}>
      <div className="message-content">
        <div>{message.content}</div>

        {message.citations && message.citations.length > 0 && (
          <div className="citations">
            <div className="citations-header">📚 Sources:</div>
            {message.citations.map((citation, idx) => (
              <div key={idx} className="citation-item">
                <span className="citation-book">{citation.book}</span> – Chapter{" "}
                {citation.chapter} – Paragraph {citation.paragraph}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
