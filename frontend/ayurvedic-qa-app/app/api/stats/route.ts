import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";

export async function GET(_request: NextRequest) {
  try {
    const response = await fetch(`${BACKEND_URL}/api/stats`, {
      method: "GET",
      headers: {
        "bypass-tunnel-reminder": "true",
        "User-Agent": "NextJS-Proxy/1.0",
      },
    });

    const data = await response.json();
    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    console.error("Proxy /api/stats error:", error);
    return NextResponse.json(
      { success: false, error: "Failed to reach backend" },
      { status: 500 }
    );
  }
}
