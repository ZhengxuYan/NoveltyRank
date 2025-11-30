
import Link from 'next/link';
import { Sparkles, Github } from 'lucide-react';

export default function Header() {
  return (
    <header className="fixed w-full top-0 z-50 glass">
      <div className="container mx-auto px-6 h-16 flex items-center justify-between">
        <div className="flex items-center gap-8">
          <Link href="/" className="flex items-center gap-3 group">
            <div className="p-2 rounded-lg bg-indigo-500/10 border border-indigo-500/20 group-hover:bg-indigo-500/20 transition-colors">
              <Sparkles className="w-5 h-5 text-indigo-400" />
            </div>
            <div className="flex flex-col">
              <span className="font-bold text-lg text-white leading-none tracking-tight">Novelty<span className="text-indigo-400">Rank</span></span>
              <span className="text-[10px] font-medium text-slate-500 uppercase tracking-widest mt-0.5">AI Paper Discovery</span>
            </div>
          </Link>
          
          <nav className="hidden md:flex items-center gap-1 text-sm font-medium text-slate-400">
            <Link href="/" className="px-4 py-2 rounded-full text-white bg-white/5 hover:bg-white/10 transition-colors">Leaderboard</Link>
          </nav>
        </div>
        
        <div className="flex items-center gap-4">
           <a 
            href="https://huggingface.co/datasets/JasonYan777/novelty-ranked-preprints" 
            target="_blank" 
            rel="noopener noreferrer"
            className="hidden sm:flex items-center gap-2 text-xs font-medium text-slate-400 hover:text-white transition-colors border border-slate-800 hover:border-slate-700 px-4 py-2 rounded-full bg-slate-900/50"
          >
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
            Dataset
          </a>
          <a
            href="https://github.com/ZhengxuYan/NoveltyRank"
            target="_blank"
            rel="noopener noreferrer"
            className="p-2 text-slate-400 hover:text-white transition-colors"
          >
            <Github className="w-5 h-5" />
          </a>
        </div>
      </div>
    </header>
  );
}
