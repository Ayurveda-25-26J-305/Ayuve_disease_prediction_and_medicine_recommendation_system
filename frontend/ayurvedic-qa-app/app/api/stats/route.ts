import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:5000";

export async function GET(_request: NextRequest) {
  try {
    console.log(`[stats] Fetching from: ${BACKEND_URL}/api/stats`);
    const response = await fetch(`${BACKEND_URL}/api/stats`, {
      method: "GET",
      cache: "no-store",
      headers: {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "bypass-tunnel-reminder": "true",
        "ngrok-skip-browser-warning": "69420",
        "User-Agent": "python-requests/2.28.0",
      },
    });

    const text = await response.text();
    console.log(`[stats] Response status: ${response.status}`);
    console.log(`[stats] Response preview: ${text.slice(0, 200)}`);

    // Guard against HTML interstitial pages (ngrok/localtunnel warning pages)
    if (text.trim().startsWith("<")) {
      console.error("[stats] Got HTML instead of JSON — tunnel interstitial page");
      return NextResponse.json(
        { success: false, error: "Tunnel interstitial — visit the ngrok URL in browser first to dismiss" },
        { status: 502 }
      );
    }

    const data = JSON.parse(text);
    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    console.error("Proxy /api/stats error:", error);
    return NextResponse.json(
      { success: false, error: "Failed to reach backend" },
      { status: 500 }
    );
  }
}
