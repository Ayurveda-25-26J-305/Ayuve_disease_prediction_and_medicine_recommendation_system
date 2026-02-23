import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:5000";

// Tell Next.js this route can take up to 300 seconds
export const maxDuration = 300;

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    // 5 minute timeout for the backend call
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 300000);

    const response = await fetch(`${BACKEND_URL}/api/ask`, {
      method: "POST",
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        "bypass-tunnel-reminder": "true",
        "ngrok-skip-browser-warning": "69420",
        "User-Agent": "python-requests/2.28.0",
      },
      body: JSON.stringify(body),
    });

    clearTimeout(timeoutId);

    const text = await response.text();
    console.log(`[ask] Response status: ${response.status}`);
    console.log(`[ask] Response preview: ${text.slice(0, 300)}`);

    if (text.trim().startsWith("<")) {
      console.error("[ask] Got HTML interstitial from tunnel");
      return NextResponse.json(
        { success: false, error: "Tunnel interstitial — open the ngrok URL in your browser first to dismiss it" },
        { status: 502 }
      );
    }

    const data = JSON.parse(text);
    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    console.error("Proxy /api/ask error:", error);
    const isTimeout = error instanceof Error && error.name === "AbortError";
    return NextResponse.json(
      { success: false, error: isTimeout ? "Backend timed out" : "Failed to reach backend" },
      { status: 500 }
    );
  }
}
