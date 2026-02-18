"use client";

import { useState, useEffect, useRef } from "react";

const API_BASE_URL = "https://earlier-pvc-least-tagged.trycloudflare.com";

interface Message {
  type: "question" | "answer";
  content: string;
  citations?: Citation[];
  validation?: {
    confidence: number;
    confidence_level: string;
  } | null;
}

interface Citation {
  source: string;
  type?: string;
  chapter?: string;
  paragraph?: string;
  qa_id?: string;
  related_question?: string;
  formatted?: string;
  similarity_percentage?: number;
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
        headers: {
          "Content-Type": "application/json",
          "Bypass-Tunnel-Reminder": "true",
          "ngrok-skip-browser-warning": "true",
        },
        body: JSON.stringify({ question }),
      });

      console.log("Response status:", response.status);
      console.log("Response OK:", response.ok);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log("Backend response:", data);
      console.log("Data type:", typeof data);
      console.log("Success field:", data.success);
      console.log("Answer field:", data.answer);
      console.log("Citations field:", data.citations);

      if (data.success) {
        console.log("Setting answer:", data.answer);
        console.log("Citations:", data.citations);
        console.log("Validation:", data.validation);
        setMessages((prev) => [
          ...prev,
          {
            type: "answer",
            content: data.answer || "No answer received",
            citations: data.citations || [],
            validation: data.validation || null,
          },
        ]);
      } else {
        console.error("Backend returned success=false:", data);
        setMessages((prev) => [
          ...prev,
          {
            type: "answer",
            content: `❌ Error: ${data.error || "Unknown error"}`,
          },
        ]);
      }
    } catch (error) {
      console.error("Error in askQuestion:", error);
      setMessages((prev) => [
        ...prev,
        {
          type: "answer",
          content:
            "❌ Failed to connect to server. Make sure the backend is running.",
        },
      ]);
    }

    console.log("Setting loading to false");
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
  // Parse answer into bullet points for better readability
  const formatAnswer = (content: string) => {
    // Split by common delimiters and create bullet points
    const sentences = content
      .split(/[.;]\s+/)
      .map((s) => s.trim())
      .filter((s) => s.length > 20); // Filter out very short fragments

    return sentences;
  };

  const answerPoints =
    message.type === "answer" ? formatAnswer(message.content) : [];

  return (
    <div className={`message message-${message.type}`}>
      <div className="message-content">
        {message.type === "question" ? (
          <div>{message.content}</div>
        ) : (
          <>
            {/* Answer Section with Bullet Points */}
            <div style={{ marginBottom: "16px" }}>
              <div
                style={{
                  fontSize: "0.9em",
                  color: "#059669",
                  fontWeight: "600",
                  marginBottom: "8px",
                  textTransform: "uppercase",
                  letterSpacing: "0.5px",
                }}
              >
                📋 Answer Summary
              </div>
              <div
                style={{
                  backgroundColor: "#ffffff",
                  padding: "12px",
                  borderRadius: "8px",
                  border: "1px solid #d1fae5",
                }}
              >
                {answerPoints.length > 0 ? (
                  <ul
                    style={{
                      margin: 0,
                      paddingLeft: "24px",
                      lineHeight: "1.8",
                    }}
                  >
                    {answerPoints.map((point, idx) => (
                      <li
                        key={idx}
                        style={{
                          marginBottom: "8px",
                          color: "#1f2937",
                        }}
                      >
                        {point}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <div>{message.content}</div>
                )}
              </div>
            </div>

            {/* Confidence Score */}
            {message.validation && (
              <div
                style={{
                  marginBottom: "16px",
                  padding: "12px",
                  backgroundColor:
                    message.validation.confidence >= 75
                      ? "#d1fae5"
                      : message.validation.confidence >= 50
                        ? "#fef3c7"
                        : "#fee2e2",
                  borderRadius: "8px",
                  fontSize: "0.95em",
                  borderLeft: `4px solid ${
                    message.validation.confidence >= 75
                      ? "#059669"
                      : message.validation.confidence >= 50
                        ? "#f59e0b"
                        : "#ef4444"
                  }`,
                }}
              >
                <strong style={{ fontSize: "1.05em" }}>
                  🎯 Confidence Score: {message.validation.confidence}%
                </strong>
                <span
                  style={{
                    marginLeft: "8px",
                    padding: "2px 8px",
                    backgroundColor: "rgba(255,255,255,0.7)",
                    borderRadius: "12px",
                    fontSize: "0.9em",
                    textTransform: "capitalize",
                  }}
                >
                  {message.validation.confidence_level}
                </span>
              </div>
            )}

            {/* Sources Section */}
            {message.citations && message.citations.length > 0 && (
              <div
                style={{
                  backgroundColor: "#f9fafb",
                  padding: "12px",
                  borderRadius: "8px",
                  border: "1px solid #e5e7eb",
                }}
              >
                <div
                  style={{
                    fontSize: "0.9em",
                    color: "#059669",
                    fontWeight: "600",
                    marginBottom: "10px",
                    textTransform: "uppercase",
                    letterSpacing: "0.5px",
                  }}
                >
                  📚 Reference Sources
                </div>
                {message.citations.map((citation, idx) => (
                  <div
                    key={idx}
                    style={{
                      backgroundColor: "#ffffff",
                      padding: "10px 12px",
                      marginBottom: "8px",
                      borderRadius: "6px",
                      border: "1px solid #e5e7eb",
                      fontSize: "0.9em",
                    }}
                  >
                    <div
                      style={{
                        fontWeight: "600",
                        color: "#059669",
                        marginBottom: "4px",
                      }}
                    >
                      {citation.source}
                    </div>
                    {citation.type === "book" ? (
                      <div
                        style={{
                          color: "#6b7280",
                          fontSize: "0.9em",
                        }}
                      >
                        {citation.chapter && citation.chapter !== "N/A" ? (
                          <>
                            Chapter {citation.chapter}
                            {citation.paragraph &&
                              citation.paragraph !== "N/A" && (
                                <> • Verse/Paragraph {citation.paragraph}</>
                              )}
                          </>
                        ) : citation.paragraph &&
                          citation.paragraph !== "N/A" ? (
                          <>Section {citation.paragraph}</>
                        ) : (
                          <>Book Reference</>
                        )}
                        {citation.similarity_percentage && (
                          <span
                            style={{
                              marginLeft: "12px",
                              color: "#059669",
                              fontWeight: "600",
                              backgroundColor: "#d1fae5",
                              padding: "2px 8px",
                              borderRadius: "12px",
                            }}
                          >
                            {citation.similarity_percentage}% match
                          </span>
                        )}
                      </div>
                    ) : (
                      <div
                        style={{
                          color: "#6b7280",
                          fontSize: "0.9em",
                          fontStyle: "italic",
                        }}
                      >
                        Q&A Reference
                        {citation.related_question && (
                          <span
                            style={{
                              display: "block",
                              marginTop: "4px",
                              fontSize: "0.85em",
                            }}
                          >
                            Related:{" "}
                            {citation.related_question.substring(0, 80)}
                            {citation.related_question.length > 80 ? "..." : ""}
                          </span>
                        )}
                        {citation.similarity_percentage && (
                          <span
                            style={{
                              marginLeft: "8px",
                              color: "#059669",
                              fontWeight: "600",
                              backgroundColor: "#d1fae5",
                              padding: "2px 8px",
                              borderRadius: "12px",
                            }}
                          >
                            {citation.similarity_percentage}% match
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
