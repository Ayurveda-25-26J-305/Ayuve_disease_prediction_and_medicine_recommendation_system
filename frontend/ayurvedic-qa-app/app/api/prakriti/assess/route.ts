import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:5000";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    const response = await fetch(`${BACKEND_URL}/api/prakriti/assess`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "ngrok-skip-browser-warning": "true",
        "User-Agent": "python-requests/2.28.0",
      },
      body: JSON.stringify(body),
    });

    const text = await response.text();
    if (text.trim().startsWith("<") || !text.trim().startsWith("{")) {
      return NextResponse.json({ success: false, error: "Backend tunnel is offline" }, { status: 502 });
    }

    const data = JSON.parse(text);
    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    console.error("Proxy /api/prakriti/assess error:", error);
    return NextResponse.json({ success: false, error: "Failed to reach backend" }, { status: 500 });
  }
}
