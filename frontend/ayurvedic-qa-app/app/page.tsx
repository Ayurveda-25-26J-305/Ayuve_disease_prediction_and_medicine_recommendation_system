"use client";

import { useState, useEffect, useRef, useCallback } from "react";

const API_BASE_URL = "";

// --- Interfaces ---

interface Message {
  id: number;
  type: "question" | "answer";
  content: string;
  citations?: Citation[];
  validation?: { confidence: number; confidence_level: string } | null;
  personalized?: boolean;
  userInfo?: { dominant_dosha: string; current_season: string } | null;
  detectedLanguage?: string;
  detectedDosha?: string;
  personalizedTips?: string;
  followUps?: string[];
  bookmarked?: boolean;
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

interface ChatSession {
  id: string;
  title: string;
  createdAt: string;
  messages: Message[];
}

// --- Prakriti ---

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

// --- Follow-up question generator ---

function generateFollowUps(question: string): string[] {
  const q = question.toLowerCase();
  const herbPatterns: Array<[RegExp, string[]]> = [
    [
      /turmeric|kaha|manjal|haldi/,
      [
        "What are the side effects of Turmeric?",
        "What is the correct dosage of Turmeric?",
        "Can Turmeric be combined with black pepper?",
      ],
    ],
    [
      /cinnamon|kurudu/,
      [
        "What are the side effects of Cinnamon?",
        "How much Cinnamon should I take daily?",
        "Is Cinnamon good for blood sugar?",
      ],
    ],
    [
      /ginger|inguru/,
      [
        "What are the side effects of Ginger?",
        "Is Ginger good for digestion?",
        "Can I take Ginger with Turmeric?",
      ],
    ],
    [
      /neem|kohomba/,
      [
        "How is Neem used for skin?",
        "What is the dosage of Neem?",
        "What are Neem side effects?",
      ],
    ],
    [
      /aloe.?vera|welpenela/,
      [
        "How much Aloe Vera is safe to drink?",
        "Is Aloe Vera good for digestion?",
        "What are the skin benefits of Aloe Vera?",
      ],
    ],
    [
      /ashwagandha/,
      [
        "How does Ashwagandha help stress?",
        "What is the dosage of Ashwagandha?",
        "Can I take Ashwagandha daily?",
      ],
    ],
    [
      /nelli|gooseberry|amla/,
      [
        "What vitamins are in Gooseberry?",
        "How does Nelli help immunity?",
        "Can I eat Nelli daily?",
      ],
    ],
    [
      /coriander|kottamalli/,
      [
        "How is Coriander used in Ayurveda?",
        "What are the digestive benefits of Coriander?",
        "Can Coriander seeds treat UTI?",
      ],
    ],
  ];
  for (const [pattern, questions] of herbPatterns) {
    if (pattern.test(q)) return questions;
  }
  if (/benefit|guna|use|property|properties/.test(q))
    return [
      "Are there any side effects?",
      "What is the recommended dosage?",
      "Which dosha benefits most from this?",
    ];
  if (/side.?effect|harm|danger|safe/.test(q))
    return [
      "What is a safe dosage?",
      "What are the benefits?",
      "Can children use this?",
    ];
  if (/dosage|dose|how much|quantity/.test(q))
    return [
      "What are the benefits?",
      "Are there side effects?",
      "How long should I take it?",
    ];
  if (/vata|pitta|kapha|dosha/.test(q))
    return [
      "What foods balance this dosha?",
      "Which herbs help this dosha?",
      "What activities help balance?",
    ];
  if (/digest|stomach|gut|bowel/.test(q))
    return [
      "What herbs improve digestion?",
      "What foods are easy to digest?",
      "How does Ayurveda treat constipation?",
    ];
  if (/skin|rash|acne/.test(q))
    return [
      "What herbs are good for skin?",
      "How is Neem used for skin?",
      "What foods improve skin health?",
    ];
  return [
    "Tell me more about this topic",
    "What are the Ayurvedic tips for this?",
    "Which dosha is affected?",
  ];
}

// --- Main Component ---

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
  const [sharedId, setSharedId] = useState<number | null>(null);
  const [lastQuestion, setLastQuestion] = useState<string>("");
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string>("");
  const [showSidebar, setShowSidebar] = useState(false);
  const [showBookmarks, setShowBookmarks] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [sidebarSearch, setSidebarSearch] = useState("");

  const msgIdRef = useRef(0);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const recognitionRef = useRef<any>(null);

  useEffect(() => {
    loadStats();
    let id = localStorage.getItem("ayurveda_user_id");
    if (!id) {
      id = "user_" + Math.random().toString(36).substring(2, 11);
      localStorage.setItem("ayurveda_user_id", id);
    }
    setUserId(id);
    const saved = localStorage.getItem("ayurveda_user_profile");
    if (saved) setUserProfile(JSON.parse(saved));
    const rawSessions = localStorage.getItem("ayurveda_sessions");
    const loadedSessions: ChatSession[] = rawSessions
      ? JSON.parse(rawSessions)
      : [];
    setSessions(loadedSessions);
    const newSessionId = "sess_" + Date.now();
    setCurrentSessionId(newSessionId);
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (!currentSessionId || messages.length === 0) return;
    const firstQ = messages.find((m) => m.type === "question");
    const session: ChatSession = {
      id: currentSessionId,
      title: firstQ
        ? firstQ.content.substring(0, 60) +
          (firstQ.content.length > 60 ? "..." : "")
        : "New session",
      createdAt: new Date().toISOString(),
      messages,
    };
    setSessions((prev) => {
      const filtered = prev.filter((s) => s.id !== currentSessionId);
      const updated = [session, ...filtered].slice(0, 20);
      localStorage.setItem("ayurveda_sessions", JSON.stringify(updated));
      return updated;
    });
  }, [messages, currentSessionId]);

  const loadStats = async () => {
    try {
      const response = await fetch(`/api/stats`);
      const data = await response.json();
      if (data.success) {
        setStats({
          docCount: data.stats.total_documents.toLocaleString(),
          modelName: data.stats.model.split("/").pop(),
        });
      } else {
        setStats({ docCount: "2,958", modelName: "Phi-3-mini" });
      }
    } catch {
      setStats({ docCount: "2,958", modelName: "Phi-3-mini" });
    }
  };

  const nextId = () => ++msgIdRef.current;

  const FOLLOW_UP_RE =
    /^(what about|tell me more|more about|how does it|how does that|and what|explain more|why is that|is it good for|what else|any side effects|side effects of|dosage of|how much|when to take|how to use it|how to take it|what are its|is it safe|can i|how often)/i;
  const PRONOUN_ONLY_RE = /^(it|that|this|those|they|them|its)\b/i;

  const buildQuestion = (q: string): string => {
    if (!lastQuestion) return q;
    const words = q.trim().split(/\s+/);
    const isFollowUp =
      FOLLOW_UP_RE.test(q) || (words.length <= 5 && PRONOUN_ONLY_RE.test(q));
    if (isFollowUp) return `${q} (regarding: ${lastQuestion})`;
    return q;
  };

  const loadSession = (session: ChatSession) => {
    setMessages(session.messages);
    setCurrentSessionId(session.id);
    setShowWelcome(session.messages.length === 0);
    setShowSidebar(false);
    const maxId = Math.max(0, ...session.messages.map((m) => m.id));
    msgIdRef.current = maxId;
  };

  const newSession = () => {
    setMessages([]);
    setShowWelcome(true);
    setCurrentSessionId("sess_" + Date.now());
    setLastQuestion("");
    setShowSidebar(false);
  };

  const deleteSession = (id: string) => {
    setSessions((prev) => {
      const updated = prev.filter((s) => s.id !== id);
      localStorage.setItem("ayurveda_sessions", JSON.stringify(updated));
      return updated;
    });
    if (id === currentSessionId) newSession();
  };

  const toggleBookmark = (id: number) => {
    setMessages((prev) =>
      prev.map((m) => (m.id === id ? { ...m, bookmarked: !m.bookmarked } : m)),
    );
  };

  const bookmarkedMessages = messages.filter(
    (m) => m.type === "answer" && m.bookmarked,
  );

  const deleteMessage = (id: number) => {
    setMessages((prev) => {
      const idx = prev.findIndex((m) => m.id === id);
      if (idx === -1) return prev;
      const next = [...prev];
      if (next[idx].type === "question") {
        if (next[idx + 1]?.type === "answer") next.splice(idx, 2);
        else next.splice(idx, 1);
      } else {
        if (next[idx - 1]?.type === "question") next.splice(idx - 1, 2);
        else next.splice(idx, 1);
      }
      if (next.length === 0) setShowWelcome(true);
      return next;
    });
  };

  const undoLast = () => {
    setMessages((prev) => {
      if (prev.length === 0) return prev;
      const next = [...prev];
      if (next[next.length - 1]?.type === "answer") next.pop();
      if (next[next.length - 1]?.type === "question") next.pop();
      if (next.length === 0) setShowWelcome(true);
      return next;
    });
  };

  const copyToClipboard = (text: string, id: number) => {
    navigator.clipboard.writeText(text).then(() => {
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 2000);
    });
  };

  const shareAnswer = (msg: Message, id: number) => {
    const idx = messages.findIndex((m) => m.id === msg.id);
    const q =
      idx > 0 && messages[idx - 1].type === "question"
        ? messages[idx - 1]
        : null;
    const card = [
      "Ayurvedic Knowledge Assistant",
      "----------------------------",
      q ? `Q: ${q.content}` : "",
      "",
      msg.content,
      msg.personalizedTips ? `\nTips:\n${msg.personalizedTips}` : "",
      msg.validation ? `\nConfidence: ${msg.validation.confidence}%` : "",
      "\nPowered by Ayurvedic Knowledge Assistant",
    ]
      .filter(Boolean)
      .join("\n");
    navigator.clipboard.writeText(card).then(() => {
      setSharedId(id);
      setTimeout(() => setSharedId(null), 2500);
    });
  };

  const startVoice = useCallback(() => {
    const SpeechRecognitionCtor =
      (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition;
    if (!SpeechRecognitionCtor) {
      alert(
        "Voice input is not supported in your browser. Try Chrome or Edge.",
      );
      return;
    }
    if (isListening && recognitionRef.current) {
      recognitionRef.current.stop();
      return;
    }
    const rec = new SpeechRecognitionCtor();
    rec.lang = "en-US";
    rec.interimResults = false;
    rec.maxAlternatives = 1;
    rec.onstart = () => setIsListening(true);
    rec.onend = () => setIsListening(false);
    rec.onerror = () => setIsListening(false);
    rec.onresult = (e: any) => {
      const t = e.results[0][0].transcript;
      setInput(t);
    };
    recognitionRef.current = rec;
    rec.start();
  }, [isListening]);

  const exportPDF = () => {
    const win = window.open("", "_blank");
    if (!win) return;
    const rows = messages
      .map((m) => {
        if (m.type === "question")
          return `<div class="q"><span class="ql">Question</span><p>${m.content}</p></div>`;
        const tips = m.personalizedTips
          ? `<div class="tips"><b>Tips:</b>${m.personalizedTips
              .split("\n")
              .filter(Boolean)
              .map((l) => `<p>${l}</p>`)
              .join("")}</div>`
          : "";
        const conf = m.validation
          ? `<div class="conf">Confidence: ${m.validation.confidence}% (${m.validation.confidence_level})</div>`
          : "";
        return `<div class="a"><span class="al">Answer</span><p>${m.content}</p>${tips}${conf}</div>`;
      })
      .join("");
    win.document.write(
      `<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Ayurvedic Chat Export</title><style>body{font-family:Georgia,serif;max-width:800px;margin:40px auto;color:#1f2937}h1{color:#059669;border-bottom:2px solid #d1fae5;padding-bottom:10px}.q{background:#f0fdf4;border-left:4px solid #059669;padding:12px 16px;margin:16px 0;border-radius:6px}.a{background:#fff;border:1px solid #e5e7eb;padding:12px 16px;margin:8px 0 20px;border-radius:6px}.ql{background:#059669;color:#fff;font-size:.75em;font-weight:700;padding:2px 8px;border-radius:12px}.al{background:#7c3aed;color:#fff;font-size:.75em;font-weight:700;padding:2px 8px;border-radius:12px}.tips{background:#faf5ff;border-left:3px solid #7c3aed;padding:10px;margin-top:10px;border-radius:4px}.conf{color:#059669;font-size:.85em;margin-top:8px;font-weight:600}p{margin:6px 0;line-height:1.7}.footer{color:#9ca3af;font-size:.8em;margin-top:40px;text-align:center}</style></head><body><h1>Ayurvedic Knowledge Assistant - Chat Export</h1><p style="color:#6b7280;font-size:.9em">Exported on ${new Date().toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" })}</p>${rows}<div class="footer">Generated by Ayurvedic Knowledge Assistant</div></body></html>`,
    );
    win.document.close();
    win.focus();
    setTimeout(() => win.print(), 500);
  };

  const askQuestion = async (question: string) => {
    if (!question.trim()) return;
    const finalQuestion = buildQuestion(question);
    setLastQuestion(question);
    setShowWelcome(false);
    const qId = nextId();
    setMessages((prev) => [
      ...prev,
      { id: qId, type: "question", content: question },
    ]);
    setInput("");
    setLoading(true);
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 300000);
      const response = await fetch(`/api/ask`, {
        method: "POST",
        signal: controller.signal,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: finalQuestion,
          user_id: userId || undefined,
        }),
      });
      clearTimeout(timeoutId);
      if (!response.ok)
        throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      if (data.success) {
        const followUps = generateFollowUps(question);
        setMessages((prev) => [
          ...prev,
          {
            id: nextId(),
            type: "answer",
            content: data.answer || "No answer received",
            citations: data.citations || [],
            validation: data.validation || null,
            personalized: data.personalized || false,
            userInfo: data.user_info || null,
            detectedLanguage: data.detected_language || "en",
            detectedDosha: data.detected_dosha || "General",
            personalizedTips: data.personalized_tips || "",
            followUps,
            bookmarked: false,
          },
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          {
            id: nextId(),
            type: "answer",
            content: `Error: ${data.error || "Unknown error"}`,
          },
        ]);
      }
    } catch (error) {
      const msg =
        error instanceof Error && error.name === "AbortError"
          ? "Request timed out. Try a simpler question."
          : "Failed to connect to server. Make sure the backend is running.";
      setMessages((prev) => [
        ...prev,
        { id: nextId(), type: "answer", content: msg },
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

  const handlePrakritiAnswer = (questionId: string, choice: string) =>
    setPrakritiAnswers((prev) => ({ ...prev, [questionId]: choice }));

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

  const filteredSessions = sessions.filter((s) =>
    sidebarSearch
      ? s.title.toLowerCase().includes(sidebarSearch.toLowerCase()) ||
        s.messages.some((m) =>
          m.content.toLowerCase().includes(sidebarSearch.toLowerCase()),
        )
      : true,
  );

  return (
    <div className="app" style={{ display: "flex" }}>
      <div className="forest-bg"></div>

      {/* Sidebar */}
      {showSidebar && (
        <div
          style={{
            position: "fixed",
            left: 0,
            top: 0,
            bottom: 0,
            width: "300px",
            background: "#fff",
            borderRight: "1px solid #d1fae5",
            zIndex: 100,
            display: "flex",
            flexDirection: "column",
            boxShadow: "4px 0 20px rgba(26,77,46,0.12)",
          }}
        >
          <div
            style={{
              padding: "14px 16px",
              borderBottom: "1px solid #e8f5e9",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              background: "linear-gradient(135deg, #1a4d2e 0%, #2d6a4f 100%)",
              borderRadius: "0",
            }}
          >
            <span
              style={{
                fontWeight: "700",
                color: "#fff",
                fontSize: "0.95em",
                letterSpacing: "0.2px",
              }}
            >
              Chat History
            </span>
            <button
              onClick={() => setShowSidebar(false)}
              style={{
                background: "rgba(255,255,255,0.15)",
                border: "none",
                cursor: "pointer",
                color: "#fff",
                width: "26px",
                height: "26px",
                borderRadius: "6px",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                transition: "background 0.15s",
              }}
              onMouseEnter={(e) =>
                (e.currentTarget.style.background = "rgba(255,255,255,0.25)")
              }
              onMouseLeave={(e) =>
                (e.currentTarget.style.background = "rgba(255,255,255,0.15)")
              }
            >
              <svg
                width="12"
                height="12"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.5"
                strokeLinecap="round"
              >
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </div>
          <div
            style={{ padding: "10px 12px", borderBottom: "1px solid #e8f5e9" }}
          >
            <input
              type="text"
              placeholder="Search chats..."
              value={sidebarSearch}
              onChange={(e) => setSidebarSearch(e.target.value)}
              style={{
                width: "100%",
                padding: "7px 10px",
                borderRadius: "8px",
                border: "1.5px solid #c6f0d8",
                fontSize: "0.85em",
                outline: "none",
                boxSizing: "border-box",
                background: "#f9fffe",
              }}
            />
          </div>
          <div
            style={{ padding: "10px 12px", borderBottom: "1px solid #e8f5e9" }}
          >
            <button
              onClick={newSession}
              style={{
                width: "100%",
                padding: "8px",
                background: "linear-gradient(135deg, #2d6a4f 0%, #40916c 100%)",
                color: "#fff",
                border: "none",
                borderRadius: "8px",
                fontWeight: "600",
                cursor: "pointer",
                fontSize: "0.88em",
                letterSpacing: "0.2px",
              }}
            >
              + New Chat
            </button>
          </div>
          <div style={{ flex: 1, overflowY: "auto", padding: "8px 10px" }}>
            {filteredSessions.length === 0 ? (
              <p
                style={{
                  color: "#9ca3af",
                  fontSize: "0.85em",
                  textAlign: "center",
                  padding: "20px",
                }}
              >
                No sessions yet
              </p>
            ) : (
              filteredSessions.map((s) => (
                <div
                  key={s.id}
                  style={{
                    padding: "9px 12px",
                    borderRadius: "8px",
                    marginBottom: "4px",
                    background:
                      s.id === currentSessionId ? "#f0fdf4" : "#fafffe",
                    border:
                      s.id === currentSessionId
                        ? "1px solid #6ee7b7"
                        : "1px solid #e8f5e9",
                    cursor: "pointer",
                    transition: "all 0.15s",
                  }}
                  onClick={() => loadSession(s)}
                >
                  <div
                    style={{
                      fontWeight: "600",
                      fontSize: "0.85em",
                      color: "#1f2937",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                    }}
                  >
                    {s.title}
                  </div>
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      marginTop: "4px",
                    }}
                  >
                    <span style={{ fontSize: "0.75em", color: "#9ca3af" }}>
                      {new Date(s.createdAt).toLocaleDateString()} ·{" "}
                      {s.messages.filter((m) => m.type === "question").length} Q
                    </span>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        deleteSession(s.id);
                      }}
                      style={{
                        background: "none",
                        border: "1px solid #fca5a5",
                        borderRadius: "5px",
                        color: "#ef4444",
                        cursor: "pointer",
                        padding: "2px 5px",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                      }}
                    >
                      <svg
                        width="12"
                        height="12"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      >
                        <polyline points="3 6 5 6 21 6" />
                        <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
                        <path d="M10 11v6" />
                        <path d="M14 11v6" />
                        <path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" />
                      </svg>
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      )}

      {/* Bookmarks Panel */}
      {showBookmarks && (
        <div
          style={{
            position: "fixed",
            right: 0,
            top: 0,
            bottom: 0,
            width: "320px",
            background: "#fff",
            borderLeft: "1px solid #6ee7b7",
            zIndex: 100,
            display: "flex",
            flexDirection: "column",
            boxShadow: "-4px 0 20px rgba(5,150,105,0.15)",
          }}
        >
          <div
            style={{
              padding: "14px 16px",
              borderBottom: "1px solid #6ee7b7",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              background: "linear-gradient(135deg, #065f46 0%, #059669 100%)",
            }}
          >
            <span
              style={{
                fontWeight: "700",
                color: "#fff",
                fontSize: "0.95em",
                letterSpacing: "0.2px",
                display: "flex",
                alignItems: "center",
                gap: "7px",
              }}
            >
              <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="#fff"
                stroke="#fff"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z" />
              </svg>
              Bookmarks
            </span>
            <button
              onClick={() => setShowBookmarks(false)}
              style={{
                background: "rgba(255,255,255,0.15)",
                border: "none",
                cursor: "pointer",
                color: "#fff",
                width: "26px",
                height: "26px",
                borderRadius: "6px",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
              onMouseEnter={(e) =>
                (e.currentTarget.style.background = "rgba(255,255,255,0.25)")
              }
              onMouseLeave={(e) =>
                (e.currentTarget.style.background = "rgba(255,255,255,0.15)")
              }
            >
              <svg
                width="12"
                height="12"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.5"
                strokeLinecap="round"
              >
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </div>
          <div style={{ flex: 1, overflowY: "auto", padding: "12px" }}>
            {bookmarkedMessages.length === 0 ? (
              <p
                style={{
                  color: "#9ca3af",
                  fontSize: "0.85em",
                  textAlign: "center",
                  padding: "30px 16px",
                }}
              >
                No bookmarks yet. Click the bookmark icon on any answer to save
                it here.
              </p>
            ) : (
              bookmarkedMessages.map((m) => {
                const idx = messages.findIndex((x) => x.id === m.id);
                const q =
                  idx > 0 && messages[idx - 1].type === "question"
                    ? messages[idx - 1]
                    : null;
                return (
                  <div
                    key={m.id}
                    style={{
                      background: "#faf5ff",
                      border: "1px solid #e9d5ff",
                      borderRadius: "8px",
                      padding: "12px",
                      marginBottom: "10px",
                    }}
                  >
                    {q && (
                      <div
                        style={{
                          fontSize: "0.8em",
                          color: "#6b7280",
                          marginBottom: "6px",
                          fontStyle: "italic",
                        }}
                      >
                        Q: {q.content}
                      </div>
                    )}
                    <div
                      style={{
                        fontSize: "0.88em",
                        color: "#1f2937",
                        lineHeight: "1.6",
                      }}
                    >
                      {m.content.substring(0, 200)}
                      {m.content.length > 200 ? "..." : ""}
                    </div>
                    <button
                      onClick={() => toggleBookmark(m.id)}
                      style={{
                        marginTop: "8px",
                        fontSize: "0.75em",
                        color: "#7c3aed",
                        background: "none",
                        border: "1px solid #c4b5fd",
                        borderRadius: "6px",
                        padding: "3px 8px",
                        cursor: "pointer",
                      }}
                    >
                      Remove
                    </button>
                  </div>
                );
              })
            )}
          </div>
        </div>
      )}

      {/* Main */}
      <div
        className="container"
        style={{
          flex: 1,
          marginLeft: showSidebar ? "300px" : "auto",
          marginRight: showBookmarks ? "320px" : "auto",
          transition: "margin 0.2s ease",
        }}
      >
        <header className="header">
          <div style={{ display: "flex", alignItems: "center", gap: "12px", flex: 1 }}>
            <button
              onClick={() => setShowSidebar(!showSidebar)}
              title="Chat history"
              style={{
                background: "none",
                border: "1px solid #d1fae5",
                borderRadius: "8px",
                padding: "6px 10px",
                cursor: "pointer",
                color: "#ffffff",
                fontSize: "1.1em",
              }}
            >
              <svg
                width="18"
                height="18"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <line x1="3" y1="6" x2="21" y2="6" />
                <line x1="3" y1="12" x2="21" y2="12" />
                <line x1="3" y1="18" x2="21" y2="18" />
              </svg>
            </button>
            <div className="logo">
              <span className="leaf-icon">🌿</span>
              <h1>Ayurvedic Knowledge Assistant</h1>
            </div>
          </div>
          {/* Centre slot — "Ancient Wisdom, Modern Insights" */}
          <div style={{ flex: 1, textAlign: "center" }}>
            <div className="subtitle" style={{ margin: 0 }}>
              Ancient Wisdom, Modern Insights
            </div>
          </div>

          {/* Right slot — action buttons */}
          <div style={{ display: "flex", alignItems: "center", gap: "8px", flex: 1, justifyContent: "flex-end" }}>
            {messages.length > 0 && (
              <button
                onClick={exportPDF}
                title="Export as PDF"
                style={{
                  background: "none",
                  border: "1px solid #d1fae5",
                  borderRadius: "8px",
                  padding: "5px 10px",
                  cursor: "pointer",
                  color: "#ffffff",
                  fontSize: "0.82em",
                  fontWeight: "600",
                  display: "flex",
                  alignItems: "center",
                  gap: "4px",
                }}
              >
                <svg
                  width="13"
                  height="13"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="7 10 12 15 17 10" />
                  <line x1="12" y1="15" x2="12" y2="3" />
                </svg>
                Export PDF
              </button>
            )}
            <button
              onClick={() => setShowBookmarks(!showBookmarks)}
              title="Bookmarks"
              style={{
                background: showBookmarks ? "#065f46" : "none",
                border: "1px solid #d1fae5",
                borderRadius: "8px",
                padding: "5px 10px",
                cursor: "pointer",
                color: "#ffffff",
                fontSize: "0.82em",
                fontWeight: "600",
                display: "flex",
                alignItems: "center",
                gap: "4px",
              }}
            >
              <svg
                width="13"
                height="13"
                viewBox="0 0 24 24"
                fill={bookmarkedMessages.length > 0 ? "currentColor" : "none"}
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z" />
              </svg>
              {bookmarkedMessages.length > 0
                ? `${bookmarkedMessages.length} saved`
                : "Bookmarks"}
            </button>
          </div>
        </header>

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
                Prakriti Assessment
              </h3>
              {userProfile && (
                <button
                  onClick={() => {
                    setUserProfile(null);
                    localStorage.removeItem("ayurveda_user_profile");
                  }}
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
                  Season: {userProfile.current_season} - Your answers will now
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
              allMessages={messages}
              onDelete={() => deleteMessage(msg.id)}
              onCopy={(text) => copyToClipboard(text, msg.id)}
              onUndo={undoLast}
              onBookmark={() => toggleBookmark(msg.id)}
              onShare={() => shareAnswer(msg, msg.id)}
              isCopied={copiedId === msg.id}
              isShared={sharedId === msg.id}
              onFollowUp={(q) => askQuestion(q)}
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
            <button
              type="button"
              onClick={startVoice}
              title={isListening ? "Stop listening" : "Voice input"}
              className={`voice-btn${isListening ? " listening" : ""}`}
            >
              <svg
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill={isListening ? "currentColor" : "none"}
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
                <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                <line x1="12" y1="19" x2="12" y2="23" />
                <line x1="8" y1="23" x2="16" y2="23" />
              </svg>
              {isListening ? "Stop" : "Voice"}
            </button>
            <button suppressHydrationWarning type="submit" disabled={loading}>
              <span className="send-icon">{loading ? "⏳" : "🌿"}</span>
              <span>{loading ? "Thinking..." : "Ask"}</span>
            </button>
          </form>
          {isListening && (
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: "6px",
                marginTop: "5px",
                color: "#ef4444",
                fontSize: "0.78em",
                fontWeight: "600",
                paddingLeft: "4px",
              }}
            >
              <span
                style={{
                  width: "7px",
                  height: "7px",
                  borderRadius: "50%",
                  background: "#ef4444",
                  display: "inline-block",
                  animation: "pulse-red 1s ease-in-out infinite",
                }}
              ></span>
              Listening... speak your question now
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// --- MessageComponent ---

function MessageComponent({
  message,
  allMessages,
  onDelete,
  onCopy,
  onUndo,
  onBookmark,
  onShare,
  isCopied,
  isShared,
  onFollowUp,
}: {
  message: Message;
  allMessages: Message[];
  onDelete: () => void;
  onCopy: (text: string) => void;
  onUndo: () => void;
  onBookmark: () => void;
  onShare: () => void;
  isCopied: boolean;
  isShared: boolean;
  onFollowUp: (q: string) => void;
}) {
  const [hovered, setHovered] = useState(false);
  const [delHov, setDelHov] = useState(false);
  const [undoHov, setUndoHov] = useState(false);
  const [bookHov, setBookHov] = useState(false);
  const [shareHov, setShareHov] = useState(false);

  const formatAnswer = (content: string, lang?: string): string[] => {
    if (!content || !content.trim()) return [];
    // For Sinhala/Tamil: split on bullet markers or newlines (don't return as single block)
    if (lang === "si" || lang === "ta") {
      const lines = content
        .split(/\n|[•\-\*]\s+/)
        .map((l) => l.trim())
        .filter((l) => l.length > 10);
      return lines.length > 1 ? lines : [content.trim()];
    }
    const lines = content
      .split("\n")
      .map((l) => l.replace(/^[•\-\*]\s*/, "").trim())
      .filter((l) => l.length > 15);
    if (content.includes("•") && lines.length > 0) return lines;
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
  const answerPlainText =
    message.type === "answer"
      ? [
          message.content,
          message.personalizedTips
            ? `\nTips:\n${message.personalizedTips}`
            : "",
        ].join("")
      : message.content;

  const CopyIco = () => (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <rect x="9" y="9" width="13" height="13" rx="2" />
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </svg>
  );
  const CheckIco = () => (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
  const UndoIco = () => (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M3 7v6h6" />
      <path d="M3 13a9 9 0 1 0 2.83-6.36L3 9" />
    </svg>
  );
  const TrashIco = () => (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <polyline points="3 6 5 6 21 6" />
      <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
      <path d="M10 11v6M14 11v6" />
      <path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" />
    </svg>
  );
  const BookIco = ({ filled }: { filled?: boolean }) => (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill={filled ? "currentColor" : "none"}
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z" />
    </svg>
  );
  const ShareIco = () => (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="18" cy="5" r="3" />
      <circle cx="6" cy="12" r="3" />
      <circle cx="18" cy="19" r="3" />
      <line x1="8.59" y1="13.51" x2="15.42" y2="17.49" />
      <line x1="15.41" y1="6.51" x2="8.59" y2="10.49" />
    </svg>
  );

  const btnBase: React.CSSProperties = {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    width: "28px",
    height: "28px",
    borderRadius: "7px",
    border: "1px solid #e5e7eb",
    background: "#fff",
    cursor: "pointer",
    boxShadow: "0 1px 3px rgba(0,0,0,0.07)",
    transition: "all 0.15s",
  };

  return (
    <div
      className={`message message-${message.type}`}
      style={{
        position: "relative",
        marginBottom:
          message.type === "answer" && message.followUps?.length
            ? "32px"
            : undefined,
      }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      {hovered && (
        <div
          style={{
            position: "absolute",
            bottom: "-14px",
            right: "8px",
            display: "flex",
            gap: "4px",
            zIndex: 10,
          }}
        >
          <button
            onClick={() => onCopy(answerPlainText)}
            title={isCopied ? "Copied!" : "Copy"}
            style={{
              ...btnBase,
              color: isCopied ? "#059669" : "#6b7280",
              borderColor: isCopied ? "#6ee7b7" : "#e5e7eb",
            }}
          >
            {isCopied ? <CheckIco /> : <CopyIco />}
          </button>
          {message.type === "answer" && (
            <>
              <button
                onClick={onBookmark}
                title={message.bookmarked ? "Remove bookmark" : "Bookmark"}
                onMouseEnter={() => setBookHov(true)}
                onMouseLeave={() => setBookHov(false)}
                style={{
                  ...btnBase,
                  color: message.bookmarked
                    ? "#7c3aed"
                    : bookHov
                      ? "#7c3aed"
                      : "#6b7280",
                  borderColor:
                    message.bookmarked || bookHov ? "#c4b5fd" : "#e5e7eb",
                }}
              >
                <BookIco filled={message.bookmarked} />
              </button>
              <button
                onClick={onShare}
                title={isShared ? "Copied!" : "Share answer"}
                onMouseEnter={() => setShareHov(true)}
                onMouseLeave={() => setShareHov(false)}
                style={{
                  ...btnBase,
                  color: isShared
                    ? "#059669"
                    : shareHov
                      ? "#2563eb"
                      : "#6b7280",
                  borderColor: isShared
                    ? "#6ee7b7"
                    : shareHov
                      ? "#bfdbfe"
                      : "#e5e7eb",
                }}
              >
                {isShared ? <CheckIco /> : <ShareIco />}
              </button>
            </>
          )}
          <button
            onClick={onUndo}
            title="Undo last Q&A"
            onMouseEnter={() => setUndoHov(true)}
            onMouseLeave={() => setUndoHov(false)}
            style={{
              ...btnBase,
              color: undoHov ? "#f59e0b" : "#6b7280",
              borderColor: undoHov ? "#fcd34d" : "#e5e7eb",
            }}
          >
            <UndoIco />
          </button>
          <button
            onClick={onDelete}
            title="Delete this Q&A"
            onMouseEnter={() => setDelHov(true)}
            onMouseLeave={() => setDelHov(false)}
            style={{
              ...btnBase,
              color: delHov ? "#ef4444" : "#6b7280",
              borderColor: delHov ? "#fca5a5" : "#e5e7eb",
            }}
          >
            <TrashIco />
          </button>
        </div>
      )}

      <div className="message-content">
        {message.type === "question" ? (
          <div>{message.content}</div>
        ) : (
          <>
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
                Answer Summary
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

            {message.personalizedTips &&
              message.personalizedTips.trim().length > 0 && (
                <div style={{ marginBottom: "16px" }}>
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
                      Personalized Tips
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
                          {message.detectedDosha} Dosha
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
                      .map((line, idx, arr) => (
                        <div
                          key={idx}
                          style={{
                            display: "flex",
                            gap: "8px",
                            marginBottom: idx < arr.length - 1 ? "8px" : 0,
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
                  Answer translated to{" "}
                  {message.detectedLanguage === "ta" ? "Tamil" : "Sinhala"}
                </span>
              </div>
            )}

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
                  borderLeft: `4px solid ${message.validation.confidence >= 75 ? "#059669" : message.validation.confidence >= 50 ? "#f59e0b" : "#ef4444"}`,
                }}
              >
                <strong>
                  Confidence Score: {message.validation.confidence}%
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

            {message.citations && message.citations.length > 0 && (
              <div
                style={{
                  backgroundColor: "#f9fafb",
                  padding: "12px",
                  borderRadius: "8px",
                  border: "1px solid #e5e7eb",
                  marginBottom: "16px",
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
                  Reference Sources
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
                      <div style={{ color: "#6b7280", fontSize: "0.9em" }}>
                        {citation.chapter && citation.chapter !== "N/A" ? (
                          <>
                            Chapter {citation.chapter}
                            {citation.paragraph &&
                              citation.paragraph !== "N/A" && (
                                <> | Section {citation.paragraph}</>
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

            {message.followUps && message.followUps.length > 0 && (
              <div style={{ marginTop: "4px" }}>
                <div
                  style={{
                    fontSize: "0.78em",
                    color: "#9ca3af",
                    fontWeight: "600",
                    marginBottom: "6px",
                    textTransform: "uppercase",
                    letterSpacing: "0.5px",
                  }}
                >
                  Follow-up questions
                </div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
                  {message.followUps.map((q, i) => (
                    <button
                      key={i}
                      onClick={() => onFollowUp(q)}
                      style={{
                        padding: "5px 12px",
                        borderRadius: "16px",
                        border: "1px solid #d1fae5",
                        background: "#f0fdf4",
                        color: "#059669",
                        fontSize: "0.82em",
                        cursor: "pointer",
                        fontWeight: "500",
                        transition: "all 0.15s",
                      }}
                      onMouseEnter={(e) => {
                        (
                          e.currentTarget as HTMLButtonElement
                        ).style.background = "#059669";
                        (e.currentTarget as HTMLButtonElement).style.color =
                          "#fff";
                      }}
                      onMouseLeave={(e) => {
                        (
                          e.currentTarget as HTMLButtonElement
                        ).style.background = "#f0fdf4";
                        (e.currentTarget as HTMLButtonElement).style.color =
                          "#059669";
                      }}
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
