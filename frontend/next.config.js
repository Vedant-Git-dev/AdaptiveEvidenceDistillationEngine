/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      // Proxy /api/* to the FastAPI sidecar on :8000.
      { source: "/api/:path*", destination: "http://localhost:8000/:path*" },
    ];
  },
};

module.exports = nextConfig;
