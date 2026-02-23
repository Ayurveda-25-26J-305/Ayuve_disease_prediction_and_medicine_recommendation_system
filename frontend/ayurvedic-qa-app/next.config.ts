import type { NextConfig } from "next";

// Allow Next.js server-side fetch to reach ngrok tunnels
// (ngrok-free.dev uses a valid cert but its CA chain is not always trusted by Node.js on Windows)
process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0";

const nextConfig: NextConfig = {
  /* config options here */
};

export default nextConfig;
