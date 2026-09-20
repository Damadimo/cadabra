/** Static export: the published site has no server, the demo reads files written by scripts/export_static.py. */
const nextConfig = {
  output: "export",
  images: { unoptimized: true },
  // `next dev` proxies the API to the local FastAPI demo so the live race works while developing.
  async rewrites() {
    return process.env.NODE_ENV === "development"
      ? [{ source: "/api/:path*", destination: "http://127.0.0.1:8000/api/:path*" }]
      : [];
  },
};
export default nextConfig;
