import type { Metadata } from "next";
import "./globals.css";

export function generateMetadata(): Metadata {
  const description =
    "1인 사업자의 업무 흐름을 기록하고 비교하는 로컬 관찰 도구";

  return {
    title: "메모리 가드 | 매장 안전 스마트 케어",
    description,
    openGraph: {
      title: "메모리 가드",
      description,
      locale: "ko_KR",
      type: "website",
    },
    twitter: {
      card: "summary",
      title: "메모리 가드",
      description,
    },
  };
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ko">
      <body>{children}</body>
    </html>
  );
}
