import type { Metadata } from 'next';
import { IBM_Plex_Mono, IBM_Plex_Sans_KR } from 'next/font/google';
import './globals.css';

const sans = IBM_Plex_Sans_KR({ variable: '--font-sans', subsets: ['latin'], weight: ['400', '500', '600', '700'] });
const mono = IBM_Plex_Mono({ variable: '--font-mono', subsets: ['latin'], weight: ['400', '600', '700'] });

export const metadata: Metadata = {
  title: { default: '오름 변화 추적 — Jeju Oreum Tracker', template: '%s · Oreum Tracker' },
  description:
    '제주 오름 243곳의 연간 변화를 frozen Earth-embedding 모델로 읽는 순서로 정리한 공개 추적 지도. 볼 수 없었던 곳은 볼 수 없었다고 표시합니다.',
  robots: { index: true, follow: true },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="ko">
      <body className={`${sans.variable} ${mono.variable}`}>{children}</body>
    </html>
  );
}
