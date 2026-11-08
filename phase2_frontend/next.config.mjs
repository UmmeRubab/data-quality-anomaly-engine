const nextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: '/api/index/:path*',
      },
    ];
  },
};

export default nextConfig;
