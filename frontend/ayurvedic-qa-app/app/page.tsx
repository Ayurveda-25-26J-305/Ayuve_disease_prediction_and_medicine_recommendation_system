"use client";

import { useState, useEffect, useRef } from "react";

// Use local proxy routes to avoid Localtunnel browser auth
const API_BASE_URL = "";

interface Message {
  id: number;
  type: "question" | "answer";
  content: string;
  citations?: Citation[];
  validation?: {
    confidence: number;
    confidence_level: string;
  } | null;
  personalized?: boolean;
  userInfo?: { dominant_dosha: string; current_season: string } | null;
  detectedLanguage?: string;
  detectedDosha?: string;
  personalizedTips?: string;
}

const PRAKRITI_QUESTIONS = [
  {
    id: "q1",
    question: "What is your body frame and build?",
    options: {
      A: "Thin, light frame, hard to gain weight",
      B: "Medium build, muscular, athletic",
      C: "Heavy, sturdy frame, easy to gain weight",
    },
  },
  {
    id: "q2",
    question: "How is your digestion typically?",
    options: {
      A: "Irregular, often gas or bloating",
      B: "Strong, feel hungry often, can't skip meals",
      C: "Slow but steady, can skip meals easily",
    },
  },
  {
    id: "q3",
    question: "What is your skin type?",
    options: {
      A: "Dry, rough, thin, gets dry patches",
      B: "Warm, oily, prone to rashes or acne",
      C: "Thick, moist, smooth, oily",
    },
  },
  {
    id: "q4",
    question: "How do you handle stress?",
    options: {
      A: "Anxious, worried, mind races",
      B: "Irritable, angry, impatient",
      C: "Calm, withdrawn, avoid confrontation",
    },
  },
  {
    id: "q5",
    question: "What is your sleep pattern like?",
    options: {
      A: "Light sleeper, difficulty falling asleep",
      B: "Moderate sleep, wake refreshed",
      C: "Heavy sleeper, need lots of sleep",
    },
  },
  {
    id: "q6",
    question: "How is your energy level?",
    options: {
      A: "Comes in bursts, get tired easily",
      B: "Consistent and strong",
      C: "Steady and enduring, slow to start",
    },
  },
  {
    id: "q7",
    question: "What is your temperature preference?",
    options: {
      A: "Prefer warm weather, dislike cold",
      B: "Prefer cool weather, dislike heat",
      C: "Comfortable in most weather",
    },
  },
  {
    id: "q8",
    question: "How do you learn and remember?",
    options: {
      A: "Learn quickly, forget quickly, creative",
      B: "Sharp intellect, good memory, focused",
      C: "Learn slowly but retain well",
    },
  },
  {
    id: "q9",
    question: "What is your speaking style?",
    options: {
      A: "Fast talker, talkative, scattered",
      B: "Precise, articulate, argumentative",
      C: "Slow, melodious, measured",
    },
  },
  {
    id: "q10",
    question: "How do you approach new activities?",
    options: {
      A: "Enthusiastic but may not finish",
      B: "Focused and determined, competitive",
      C: "Resistant to change, prefer routine",
    },
  },
];

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
  const [userId, setUserId] = useState<string>("");
  const [userProfile, setUserProfile] = useState<{
    dominant_dosha: string;
    current_season: string;
  } | null>(null);
  const [showPrakritiQuiz, setShowPrakritiQuiz] = useState(false);
  const [prakritiAnswers, setPrakritiAnswers] = useState<
    Record<string, string>
  >({});
  const [prakritiSubmitting, setPrakritiSubmitting] = useState(false);
  const [copiedId, setCopiedId] = useState<number | null>(null);
  const [lastQuestion, setLastQuestion] = useState<string>("");
  const msgIdRef = useRef(0);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    loadStats();
    // Load or generate persistent user ID
    let id = localStorage.getItem("ayurveda_user_id");
    if (!id) {
      id = "user_" + Math.random().toString(36).substring(2, 11);
      localStorage.setItem("ayurveda_user_id", id);
    }
    setUserId(id);
    // Load saved profile
    const saved = localStorage.getItem("ayurveda_user_profile");
    if (saved) setUserProfile(JSON.parse(saved));
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const loadStats = async () => {
    try {
      console.log("Fetching stats from: /api/stats (proxy)");
      const response = await fetch(`/api/stats`);
      console.log("Stats response status:", response.status);
      const data = await response.json();
      console.log("Stats data received:", data);

      if (data.success) {
        setStats({
          docCount: data.stats.total_documents.toLocaleString(),
          modelName: data.stats.model.split("/").pop(),
        });
      } else {
        // Stats failure is non-critical — app still works for questions
        setStats({ docCount: "2,958", modelName: "Phi-3-mini" });
      }
    } catch (error) {
      // Stats failure is non-critical — show defaults
      setStats({ docCount: "2,958", modelName: "Phi-3-mini" });
    }
  };

  const nextId = () => ++msgIdRef.current;

  // --- Conversation memory: detect follow-up questions ---
  const FOLLOW_UP_RE =
    /^(what about|tell me more|more about|how does it|how does that|and what|explain more|why is that|is it good for|what else|any side effects|side effects of|dosage of|how much|when to take|how to use it|how to take it|what are its|what are the side|is it safe|can i|how often)/i;
  const PRONOUN_ONLY_RE = /^(it|that|this|those|they|them|its)\b/i;

  const buildQuestion = (q: string): string => {
    if (!lastQuestion) return q;
    const words = q.trim().split(/\s+/);
    const isFollowUp =
      FOLLOW_UP_RE.test(q) || (words.length <= 5 && PRONOUN_ONLY_RE.test(q));
    if (isFollowUp) return `${q} (regarding: ${lastQuestion})`;
    return q;
  };

  // --- Delete a Q+A pair by message id ---
  const deleteMessage = (id: number) => {
    setMessages((prev) => {
      const idx = prev.findIndex((m) => m.id === id);
      if (idx === -1) return prev;
      const next = [...prev];
      if (next[idx].type === "question") {
        // Remove question + its following answer
        if (next[idx + 1]?.type === "answer") next.splice(idx, 2);
        else next.splice(idx, 1);
      } else {
        // Clicked on answer — also remove its preceding question
        if (next[idx - 1]?.type === "question") next.splice(idx - 1, 2);
        else next.splice(idx, 1);
      }
      if (next.length === 0) setShowWelcome(true);
      return next;
    });
  };

  // --- Undo: remove last Q+A pair ---
  const undoLast = () => {
    setMessages((prev) => {
      if (prev.length === 0) return prev;
      const next = [...prev];
      // Remove last answer if it exists
      if (next[next.length - 1]?.type === "answer") next.pop();
      // Remove last question
      if (next[next.length - 1]?.type === "question") next.pop();
      if (next.length === 0) setShowWelcome(true);
      return next;
    });
  };

  // --- Copy to clipboard ---
  const copyToClipboard = (text: string, id: number) => {
    navigator.clipboard.writeText(text).then(() => {
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 2000);
    });
  };

  const askQuestion = async (question: string) => {
    if (!question.trim()) return;

    const finalQuestion = buildQuestion(question);
    setLastQuestion(question); // store original (not expanded) for display

    setShowWelcome(false);
    const qId = nextId();
    setMessages((prev) => [
      ...prev,
      { id: qId, type: "question", content: question },
    ]);
    setInput("");
    setLoading(true);

    try {
      // Add timeout of 5 minutes (300 seconds)
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 300000);

      const response = await fetch(`/api/ask`, {
        method: "POST",
        signal: controller.signal,
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: finalQuestion,
          user_id: userId || undefined,
        }),
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      if (data.success) {
        const aId = nextId();
        setMessages((prev) => [
          ...prev,
          {
            id: aId,
            type: "answer",
            content: data.answer || "No answer received",
            citations: data.citations || [],
            validation: data.validation || null,
            personalized: data.personalized || false,
            userInfo: data.user_info || null,
            detectedLanguage: data.detected_language || "en",
            detectedDosha: data.detected_dosha || "General",
            personalizedTips: data.personalized_tips || "",
          },
        ]);
      } else {
        console.error("Backend returned success=false:", data);
        setMessages((prev) => [
          ...prev,
          {
            id: nextId(),
            type: "answer",
            content: `❌ Error: ${data.error || "Unknown error"}`,
          },
        ]);
      }
    } catch (error) {
      console.error("Error in askQuestion:", error);
      const errorMessage =
        error instanceof Error && error.name === "AbortError"
          ? "⏱️ Request timed out. The server is taking too long to respond. Try a simpler question."
          : "❌ Failed to connect to server. Make sure the backend is running.";

      setMessages((prev) => [
        ...prev,
        {
          id: nextId(),
          type: "answer",
          content: errorMessage,
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

  const handlePrakritiAnswer = (questionId: string, choice: string) => {
    setPrakritiAnswers((prev) => ({ ...prev, [questionId]: choice }));
  };

  const submitPrakriti = async () => {
    if (Object.keys(prakritiAnswers).length < PRAKRITI_QUESTIONS.length) {
      alert("Please answer all questions before submitting.");
      return;
    }
    setPrakritiSubmitting(true);
    try {
      const res = await fetch("/api/prakriti/assess", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: userId, responses: prakritiAnswers }),
      });
      const data = await res.json();
      if (data.success) {
        const profile = {
          dominant_dosha: data.profile.dominant_dosha,
          current_season: data.profile.current_season,
        };
        setUserProfile(profile);
        localStorage.setItem("ayurveda_user_profile", JSON.stringify(profile));
        setShowPrakritiQuiz(false);
        setPrakritiAnswers({});
      } else {
        alert("Failed to save profile: " + (data.error || "Unknown error"));
      }
    } catch {
      alert("Could not connect to backend.");
    } finally {
      setPrakritiSubmitting(false);
    }
  };

  const clearProfile = () => {
    setUserProfile(null);
    localStorage.removeItem("ayurveda_user_profile");
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
          <div
            className="stat-item"
            style={{ cursor: "pointer" }}
            onClick={() => setShowPrakritiQuiz(!showPrakritiQuiz)}
          >
            <span className="stat-icon">{userProfile ? "🧬" : "👤"}</span>
            <span className="stat-value" style={{ fontSize: "0.95em" }}>
              {userProfile
                ? userProfile.dominant_dosha.charAt(0).toUpperCase() +
                  userProfile.dominant_dosha.slice(1)
                : "Set Profile"}
            </span>
            <span className="stat-label">
              {userProfile ? "My Dosha" : "Personalize"}
            </span>
          </div>
        </div>

        {/* Prakriti Quiz Panel */}
        {showPrakritiQuiz && (
          <div
            style={{
              background: "#fff",
              border: "1px solid #d1fae5",
              borderRadius: "12px",
              padding: "20px",
              marginBottom: "16px",
              maxHeight: "400px",
              overflowY: "auto",
            }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                marginBottom: "16px",
              }}
            >
              <h3 style={{ color: "#059669", margin: 0 }}>
                🧬 Prakriti Assessment
              </h3>
              {userProfile && (
                <button
                  onClick={clearProfile}
                  style={{
                    fontSize: "0.8em",
                    color: "#ef4444",
                    background: "none",
                    border: "1px solid #ef4444",
                    borderRadius: "6px",
                    padding: "4px 10px",
                    cursor: "pointer",
                  }}
                >
                  Clear Profile
                </button>
              )}
            </div>
            {userProfile ? (
              <div style={{ textAlign: "center", padding: "20px" }}>
                <div style={{ fontSize: "2em", marginBottom: "8px" }}>🧬</div>
                <p
                  style={{
                    color: "#059669",
                    fontWeight: "600",
                    fontSize: "1.1em",
                  }}
                >
                  Your Dominant Dosha:{" "}
                  {userProfile.dominant_dosha.charAt(0).toUpperCase() +
                    userProfile.dominant_dosha.slice(1)}
                </p>
                <p style={{ color: "#6b7280", fontSize: "0.9em" }}>
                  Season: {userProfile.current_season} • Your answers will now
                  be personalized
                </p>
                <button
                  onClick={() => {
                    setUserProfile(null);
                    localStorage.removeItem("ayurveda_user_profile");
                    setPrakritiAnswers({});
                  }}
                  style={{
                    marginTop: "12px",
                    color: "#6b7280",
                    background: "none",
                    border: "1px solid #d1d5db",
                    borderRadius: "6px",
                    padding: "6px 14px",
                    cursor: "pointer",
                    fontSize: "0.9em",
                  }}
                >
                  Retake Quiz
                </button>
              </div>
            ) : (
              <>
                {PRAKRITI_QUESTIONS.map((q, idx) => (
                  <div key={q.id} style={{ marginBottom: "16px" }}>
                    <p
                      style={{
                        fontWeight: "600",
                        color: "#1f2937",
                        marginBottom: "8px",
                      }}
                    >
                      {idx + 1}. {q.question}
                    </p>
                    {Object.entries(q.options).map(([choice, text]) => (
                      <label
                        key={choice}
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: "8px",
                          marginBottom: "6px",
                          cursor: "pointer",
                          color:
                            prakritiAnswers[q.id] === choice
                              ? "#059669"
                              : "#6b7280",
                          fontWeight:
                            prakritiAnswers[q.id] === choice ? "600" : "400",
                        }}
                      >
                        <input
                          type="radio"
                          name={q.id}
                          value={choice}
                          checked={prakritiAnswers[q.id] === choice}
                          onChange={() => handlePrakritiAnswer(q.id, choice)}
                          style={{ accentColor: "#059669" }}
                        />
                        <span>
                          {choice}: {text}
                        </span>
                      </label>
                    ))}
                  </div>
                ))}
                <button
                  onClick={submitPrakriti}
                  disabled={prakritiSubmitting}
                  style={{
                    width: "100%",
                    padding: "10px",
                    background: "#059669",
                    color: "#fff",
                    border: "none",
                    borderRadius: "8px",
                    fontWeight: "600",
                    cursor: prakritiSubmitting ? "not-allowed" : "pointer",
                    opacity: prakritiSubmitting ? 0.7 : 1,
                  }}
                >
                  {prakritiSubmitting ? "Saving..." : "Save My Profile"}
                </button>
              </>
            )}
          </div>
        )}

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
                  suppressHydrationWarning
                  className="example-btn"
                  onClick={() => askExample("What causes Vata imbalance?")}
                >
                  What causes Vata imbalance?
                </button>
                <button
                  suppressHydrationWarning
                  className="example-btn"
                  onClick={() => askExample("How to treat Pitta disorders?")}
                >
                  How to treat Pitta disorders?
                </button>
                <button
                  suppressHydrationWarning
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

          {messages.map((msg) => (
            <MessageComponent
              key={msg.id}
              message={msg}
              onDelete={() => deleteMessage(msg.id)}
              onCopy={(text) => copyToClipboard(text, msg.id)}
              onUndo={undoLast}
              isCopied={copiedId === msg.id}
            />
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
              suppressHydrationWarning
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask your Ayurvedic question..."
              disabled={loading}
            />
            <button suppressHydrationWarning type="submit" disabled={loading}>
              <span className="send-icon">{loading ? "⏳" : "🌿"}</span>
              <span>{loading ? "Thinking..." : "Ask"}</span>
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

function MessageComponent({
  message,
  onDelete,
  onCopy,
  onUndo,
  isCopied,
}: {
  message: Message;
  onDelete: () => void;
  onCopy: (text: string) => void;
  onUndo: () => void;
  isCopied: boolean;
}) {
  const [hovered, setHovered] = useState(false);
  const [deleteHovered, setDeleteHovered] = useState(false);
  const [undoHovered, setUndoHovered] = useState(false);

  // SVG icon components
  const CopyIcon = () => (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="9" y="9" width="13" height="13" rx="2"/>
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
    </svg>
  );
  const CheckIcon = () => (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="20 6 9 17 4 12"/>
    </svg>
  );
  const UndoIcon = () => (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 7v6h6"/>
      <path d="M3 13a9 9 0 1 0 2.83-6.36L3 9"/>
    </svg>
  );
  const TrashIcon = () => (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="3 6 5 6 21 6"/>
      <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/>
      <path d="M10 11v6M14 11v6"/>
      <path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>
    </svg>
  );

  const iconBtnBase: React.CSSProperties = {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    width: "30px",
    height: "30px",
    borderRadius: "8px",
    border: "1px solid #e5e7eb",
    background: "#fff",
    cursor: "pointer",
    boxShadow: "0 1px 4px rgba(0,0,0,0.08)",
    transition: "all 0.15s",
  };
  // Answer is prose (2-3 sentences). Split on sentence boundaries for display.
  // For Sinhala text, don't split — render as a single paragraph.
  const formatAnswer = (content: string, lang?: string): string[] => {
    if (!content || !content.trim()) return [];

    // Non-English scripts (Sinhala / Tamil): show as a single block, no splitting
    if (lang === "si" || lang === "ta") return [content.trim()];

    // Structured bullet output from backend (• prefix lines)
    const lines = content
      .split("\n")
      .map((l) => l.replace(/^[•\-\*]\s*/, "").trim())
      .filter((l) => l.length > 15);
    if (content.includes("•") && lines.length > 0) return lines;

    // Plain prose: split on sentence boundaries (. ! ? followed by capital)
    const sentences = content
      .split(/(?<=[.!?])\s+(?=[A-Z])/)
      .map((s) => s.trim())
      .filter((s) => s.length > 20);
    return sentences.length > 1 ? sentences : [content.trim()];
  };

  const answerPoints =
    message.type === "answer"
      ? formatAnswer(message.content, message.detectedLanguage)
      : [];

  // Plain-text version of answer for copying
  const answerPlainText =
    message.type === "answer"
      ? [
          message.content,
          message.personalizedTips
            ? `\nTips:\n${message.personalizedTips}`
            : "",
        ].join("")
      : message.content;

  return (
    <div
      className={`message message-${message.type}`}
      style={{ position: "relative" }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      {/* Action toolbar — shown on hover */}
      {hovered && (
        <div
          style={{
            position: "absolute",
            bottom: "-16px",
            right: "8px",
            display: "flex",
            gap: "4px",
            zIndex: 10,
          }}
        >
          {/* Copy button */}
          <button
            onClick={() => onCopy(answerPlainText)}
            title={isCopied ? "Copied!" : "Copy"}
            style={{
              ...iconBtnBase,
              color: isCopied ? "#059669" : "#6b7280",
              borderColor: isCopied ? "#6ee7b7" : "#e5e7eb",
            }}
          >
            {isCopied ? <CheckIcon /> : <CopyIcon />}
          </button>

          {/* Undo last button */}
          <button
            onClick={onUndo}
            title="Undo last Q&A"
            onMouseEnter={() => setUndoHovered(true)}
            onMouseLeave={() => setUndoHovered(false)}
            style={{
              ...iconBtnBase,
              color: undoHovered ? "#f59e0b" : "#6b7280",
              borderColor: undoHovered ? "#fcd34d" : "#e5e7eb",
            }}
          >
            <UndoIcon />
          </button>

          {/* Delete button */}
          <button
            onClick={onDelete}
            title="Delete this Q&A"
            onMouseEnter={() => setDeleteHovered(true)}
            onMouseLeave={() => setDeleteHovered(false)}
            style={{
              ...iconBtnBase,
              color: deleteHovered ? "#ef4444" : "#6b7280",
              borderColor: deleteHovered ? "#fca5a5" : "#e5e7eb",
            }}
          >
            <TrashIcon />
          </button>
        </div>
      )}
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
                      paddingLeft: "20px",
                      lineHeight: "1.8",
                    }}
                  >
                    {answerPoints.map((point, idx) => (
                      <li
                        key={idx}
                        style={{
                          marginBottom:
                            idx < answerPoints.length - 1 ? "8px" : 0,
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

            {/* Personalized Tips Section */}
            {message.personalizedTips &&
              message.personalizedTips.trim().length > 0 && (
                <div style={{ marginBottom: "16px" }}>
                  {/* Dosha Badge Header */}
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "10px",
                      marginBottom: "8px",
                    }}
                  >
                    <div
                      style={{
                        fontSize: "0.9em",
                        color: "#7c3aed",
                        fontWeight: "600",
                        textTransform: "uppercase",
                        letterSpacing: "0.5px",
                      }}
                    >
                      💡 Personalized Tips
                    </div>
                    {message.detectedDosha &&
                      message.detectedDosha !== "General" && (
                        <span
                          style={{
                            padding: "2px 10px",
                            borderRadius: "12px",
                            fontSize: "0.8em",
                            fontWeight: "600",
                            backgroundColor: message.detectedDosha.includes(
                              "Vata",
                            )
                              ? "#ede9fe"
                              : message.detectedDosha.includes("Pitta")
                                ? "#fef3c7"
                                : "#d1fae5",
                            color: message.detectedDosha.includes("Vata")
                              ? "#7c3aed"
                              : message.detectedDosha.includes("Pitta")
                                ? "#d97706"
                                : "#059669",
                            border: `1px solid ${message.detectedDosha.includes("Vata") ? "#c4b5fd" : message.detectedDosha.includes("Pitta") ? "#fcd34d" : "#6ee7b7"}`,
                          }}
                        >
                          🧬 {message.detectedDosha} Dosha
                        </span>
                      )}
                  </div>
                  <div
                    style={{
                      backgroundColor: "#faf5ff",
                      padding: "12px",
                      borderRadius: "8px",
                      border: "1px solid #e9d5ff",
                    }}
                  >
                    {message.personalizedTips
                      .split("\n")
                      .filter((l) => l.trim())
                      .map((line, idx) => (
                        <div
                          key={idx}
                          style={{
                            display: "flex",
                            gap: "8px",
                            marginBottom:
                              idx <
                              message
                                .personalizedTips!.split("\n")
                                .filter((l) => l.trim()).length -
                                1
                                ? "8px"
                                : 0,
                            color: "#4b5563",
                            lineHeight: "1.7",
                          }}
                        >
                          <span
                            style={{
                              color: "#7c3aed",
                              fontWeight: "700",
                              minWidth: "20px",
                            }}
                          >
                            {idx + 1}.
                          </span>
                          <span>{line.replace(/^\d+\.\s*/, "")}</span>
                        </div>
                      ))}
                  </div>
                </div>
              )}

            {/* Language Detection Badge */}
            {message.detectedLanguage && message.detectedLanguage !== "en" && (
              <div
                style={{
                  marginBottom: "12px",
                  padding: "8px 12px",
                  backgroundColor: "#eff6ff",
                  borderRadius: "8px",
                  border: "1px solid #bfdbfe",
                  fontSize: "0.9em",
                  display: "flex",
                  alignItems: "center",
                  gap: "8px",
                }}
              >
                <span>🌐</span>
                <span style={{ color: "#2563eb", fontWeight: "600" }}>
                  {message.detectedLanguage === "ta"
                    ? "Tamil detected"
                    : "Sinhala detected"}
                </span>
                <span style={{ color: "#6b7280", fontSize: "0.85em" }}>
                  • Answer translated to{" "}
                  {message.detectedLanguage === "ta" ? "Tamil" : "Sinhala"}
                </span>
              </div>
            )}

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
