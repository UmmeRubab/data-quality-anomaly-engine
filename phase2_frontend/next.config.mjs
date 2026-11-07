/** Next.js forwards /api calls to the FastAPI backend.
 * Local development: http://127.0.0.1:8000
 * Vercel: set DATAGUARD_API_URL to the public Render backend URL.
 */
const backend = (process.env.DATAGUARD_API_URL || process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');
const nextConfig = {
  async rewrites() {
    return [{ source: '/api/:path*', destination: `${backend}/api/:path*` }];
  },
};
export default nextConfig;
