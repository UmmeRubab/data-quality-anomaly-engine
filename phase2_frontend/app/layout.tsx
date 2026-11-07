import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'DataGuard AI | Research Dashboard',
  description: 'Controlled data quality and ML robustness experiments',
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
