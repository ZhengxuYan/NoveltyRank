
import './globals.css'
import { Inter } from 'next/font/google'
import Header from '@/components/Header'

const inter = Inter({ subsets: ['latin'] })

export const metadata = {
  title: 'Novelty Rank: AI Paper Discovery',
  description: 'Discover the most novel AI papers from arXiv, ranked by Siamese SciBERT.',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${inter.className} min-h-screen flex flex-col bg-slate-950 text-slate-200 antialiased selection:bg-indigo-500/30 selection:text-indigo-200`}>
        <Header />
        <main className="flex-1">
          {children}
        </main>
      </body>
    </html>
  )
}
