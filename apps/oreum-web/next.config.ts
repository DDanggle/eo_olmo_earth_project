import type { NextConfig } from 'next';
// Vercel: 정적 자산(public/data 125 MB)은 CDN으로 서빙됨. 서버 기능은 쓰지 않음.
const securityHeaders = [
  { key: 'X-Content-Type-Options', value: 'nosniff' },
  { key: 'X-Frame-Options', value: 'DENY' },
  { key: 'Referrer-Policy', value: 'strict-origin-when-cross-origin' },
  { key: 'Permissions-Policy', value: 'camera=(), microphone=(), geolocation=()' },
];

// public/data 는 배포 커밋과 함께만 바뀌므로 CDN 에 오래 캐시하고, 브라우저는 매번 재검증한다.
// Vercel 은 새 배포마다 edge 캐시를 무효화하므로 stale 데이터가 남지 않는다.
const dataCacheHeaders = [
  { key: 'Cache-Control', value: 'public, max-age=0, s-maxage=31536000, must-revalidate' },
];

// 정적 export: 서버 기능이 없다(모든 데이터는 public/data 의 JSON·이미지). GitHub Pages 처럼 하위 경로에
// 놓일 때는 NEXT_PUBLIC_BASE_PATH(예: /oreum-tracker) 를 주고, 코드는 withBase() 로 그 접두를 붙인다.
const basePath = process.env.NEXT_PUBLIC_BASE_PATH || '';

const nextConfig: NextConfig = {
  output: 'export',
  basePath,
  images: { unoptimized: true },
  trailingSlash: true,
  reactStrictMode: true,
  poweredByHeader: false,
};
export default nextConfig;
