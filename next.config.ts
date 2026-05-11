import type { NextConfig } from "next";

const allowedDevOrigins = (process.env.NEXT_ALLOWED_DEV_ORIGINS || "")
  .split(",")
  .map((origin) => origin.trim())
  .filter(Boolean);

const nextConfig: NextConfig = {
  reactStrictMode: true,
  allowedDevOrigins:
    allowedDevOrigins.length > 0
      ? allowedDevOrigins
      : ["http://192.168.0.146", "http://localhost:3000", "http://127.0.0.1:3000"],
};

export default nextConfig;
