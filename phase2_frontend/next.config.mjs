
const nextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: '/api/index',
      },
    ];
  },
};

export default nextConfig;
