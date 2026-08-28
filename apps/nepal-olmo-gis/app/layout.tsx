import type { Metadata } from 'next';
import { IBM_Plex_Mono, IBM_Plex_Sans_KR } from 'next/font/google';
import './globals.css';

// Ai2 종이 톤 전환(2026-08-28): 폰트만 IBM Plex로 교체하고 CSS 변수 이름은 유지함
// (--font-geist-* 를 참조하는 규칙 50여 곳을 건드리지 않기 위해).
const geistSans = IBM_Plex_Sans_KR({
  variable: '--font-geist-sans', subsets: ['latin'], weight: ['400', '500', '600', '700'],
});
const geistMono = IBM_Plex_Mono({
  variable: '--font-geist-mono', subsets: ['latin'], weight: ['400', '600', '700'],
});

export const metadata: Metadata = {
  title: 'OLMoEarth Nepal Live Twin',
  description: 'Satellite evidence, OLMoEarth embeddings, and a transparent hazard-simulation interface for Rasuwa, Nepal.',
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable}`}>{children}</body>
    </html>
  );
}
