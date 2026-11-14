import { useEffect, useState, type ReactNode } from 'react'
import { Link, NavLink, useLocation } from 'react-router-dom'
import { BookMarked, Check, ChevronDown, FlaskConical, Menu, Monitor, Moon, Search as SearchIcon, Sun, X } from 'lucide-react'
import { LESSON_ORDER, PAGES, REFERENCE, SITE, TRACKS, routeHref } from '../content'
import { useProgress } from './progress'
import { setThemeMode, useTheme } from './theme'
import { Search } from "./Search"
import { GithubIcon } from "./GithubIcon"

export function Layout({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false)
  const [searchOpen, setSearchOpen] = useState(false)
  const loc = useLocation()
  useEffect(() => { setOpen(false) }, [loc.pathname])
  const done = useProgress()
  const total = LESSON_ORDER.length
  const pct = total ? Math.round((LESSON_ORDER.filter((r) => done.has(r)).length / total) * 100) : 0

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null
      const typing = t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)
      if ((e.key === 'k' && (e.metaKey || e.ctrlKey)) || (e.key === '/' && !typing)) { e.preventDefault(); setSearchOpen(true) }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-30 flex h-14 items-center gap-3 border-b border-line bg-bg/85 px-4 backdrop-blur">
        <button className="rounded-md p-1.5 lg:hidden" onClick={() => setOpen((o) => !o)} aria-label="Toggle navigation">{open ? <X className="size-5" /> : <Menu className="size-5" />}</button>
        <Link to="/" className="flex items-center gap-2 font-bold tracking-tight">
          <img src="./favicon.svg" alt="" className="size-6" /> <span className="hidden sm:inline">{SITE.title}</span><span className="sm:hidden">LLM Scratch</span>
        </Link>
        <div className="ml-auto flex items-center gap-3">
          <button onClick={() => setSearchOpen(true)} className="flex items-center gap-2 rounded-lg border border-line bg-surface-2 px-2.5 py-1.5 text-sm text-muted hover:text-ink sm:w-56" aria-label="Search">
            <SearchIcon className="size-4" /> <span className="hidden sm:inline">Search</span>
            <kbd className="ml-auto hidden rounded border border-line px-1.5 font-mono text-[10px] sm:inline">⌘K</kbd>
          </button>
          <Link to={routeHref('projects')} className="hidden items-center gap-1.5 text-sm text-muted hover:text-ink md:flex"><FlaskConical className="size-4" /> Projects</Link>
          <Link to={routeHref('glossary')} className="hidden items-center gap-1.5 text-sm text-muted hover:text-ink md:flex"><BookMarked className="size-4" /> Glossary</Link>
          <div className="hidden items-center gap-2 lg:flex" title={`${pct}% of lessons complete`}>
            <div className="h-1.5 w-24 overflow-hidden rounded-full bg-surface-2"><div className="h-full bg-ok transition-all" style={{ width: `${pct}%` }} /></div>
            <span className="font-mono text-xs text-muted">{pct}%</span>
          </div>
          <ThemeToggle />
          <a href={SITE.repo} target="_blank" rel="noopener" className="hidden rounded-md p-1.5 text-muted hover:text-ink sm:block" aria-label="Source on GitHub"><GithubIcon className="size-4.5" /></a>
        </div>
      </header>
      <div className="flex">
        <aside className={`fixed inset-y-0 top-14 z-20 w-80 shrink-0 overflow-y-auto border-r border-line bg-bg scrollbar-thin transition-transform lg:sticky lg:h-[calc(100vh-3.5rem)] lg:translate-x-0 ${open ? 'translate-x-0' : '-translate-x-full'}`}>
          <Sidebar />
        </aside>
        {open && <div className="fixed inset-0 top-14 z-10 bg-black/40 lg:hidden" onClick={() => setOpen(false)} />}
        <main className="min-w-0 flex-1">{children}</main>
      </div>
      {searchOpen && <Search onClose={() => setSearchOpen(false)} />}
    </div>
  )
}

function Sidebar() {
  const done = useProgress()
  const loc = useLocation()
  const current = decodeURIComponent(loc.pathname).replace(/^\/+|\/+$/g, '')
  const currentModule = PAGES[current]?.module
  const [expanded, setExpanded] = useState<Record<string, boolean>>({})
  return (
    <nav className="px-3 py-4" aria-label="Curriculum">
      <div className="mb-3 grid grid-cols-2 gap-2">
        <Link to={routeHref('projects')} className="flex items-center gap-1.5 rounded-lg border border-line px-2.5 py-2 text-[13px] font-semibold hover:border-accent"><FlaskConical className="size-4 text-accent" /> Projects</Link>
        <Link to={routeHref('glossary')} className="flex items-center gap-1.5 rounded-lg border border-line px-2.5 py-2 text-[13px] font-semibold hover:border-accent"><BookMarked className="size-4 text-accent" /> Glossary</Link>
      </div>
      {TRACKS.map((t) => (
        <div key={t.title} className="mb-3">
          <p className="px-2 pt-2 pb-1 text-[11px] font-bold tracking-wider text-muted uppercase">{t.title}</p>
          {t.modules.map((m) => {
            const n = m.lessons.filter((l) => done.has(l.route)).length
            const isOpen = expanded[m.code] ?? currentModule === m.code
            return (
              <div key={m.code} className="mb-0.5">
                <button onClick={() => setExpanded((e) => ({ ...e, [m.code]: !isOpen }))} className="flex w-full items-start gap-2 rounded-lg px-2 py-2 text-left hover:bg-surface-2" aria-expanded={isOpen}>
                  <span className="mt-0.5 flex h-6 min-w-7 shrink-0 items-center justify-center rounded-md bg-accent-soft px-1 font-mono text-[11px] font-bold text-accent">{m.code}</span>
                  <span className="min-w-0 flex-1">
                    <span className="block text-sm leading-snug font-semibold">{m.title}</span>
                    <span className="mt-1 flex items-center gap-2">
                      <span className="h-1 flex-1 overflow-hidden rounded-full bg-surface-2"><span className="block h-full bg-ok transition-all" style={{ width: m.lessons.length ? `${(n / m.lessons.length) * 100}%` : 0 }} /></span>
                      <span className="font-mono text-[10px] text-muted">{n}/{m.lessons.length}</span>
                    </span>
                  </span>
                  <ChevronDown className={`mt-1 size-4 shrink-0 text-muted transition ${isOpen ? '' : '-rotate-90'}`} />
                </button>
                {isOpen && (
                  <ol className="mt-0.5 mb-2 ml-5 border-l border-line">
                    {m.lessons.map((l) => (
                      <li key={l.route}>
                        <NavLink to={routeHref(l.route)} end className={({ isActive }) => `-ml-px flex items-center gap-2 border-l-2 py-1.5 pr-2 pl-3 text-[13px] leading-snug ${isActive ? 'border-accent font-semibold text-accent' : 'border-transparent text-muted hover:text-ink'}`}>
                          <span className="flex-1">{l.label}</span>
                          {done.has(l.route) && <Check className="size-3.5 shrink-0 text-ok" aria-label="completed" />}
                        </NavLink>
                      </li>
                    ))}
                  </ol>
                )}
              </div>
            )
          })}
        </div>
      ))}
      <div className="mb-3">
        <p className="px-2 pt-2 pb-1 text-[11px] font-bold tracking-wider text-muted uppercase">Reference</p>
        <ol className="mt-0.5 ml-5 border-l border-line">
          {REFERENCE.map((l) => (
            <li key={l.route}>
              <NavLink to={routeHref(l.route)} end className={({ isActive }) => `-ml-px flex border-l-2 py-1.5 pr-2 pl-3 text-[13px] leading-snug ${isActive ? 'border-accent font-semibold text-accent' : 'border-transparent text-muted hover:text-ink'}`}>{l.label}</NavLink>
            </li>
          ))}
        </ol>
      </div>
    </nav>
  )
}

function ThemeToggle() {
  const { mode } = useTheme()
  const next = mode === 'system' ? 'dark' : mode === 'dark' ? 'light' : 'system'
  const Icon = mode === 'system' ? Monitor : mode === 'dark' ? Moon : Sun
  return (
    <button onClick={() => setThemeMode(next)} className="rounded-md p-1.5 text-muted hover:text-ink" aria-label={`Theme: ${mode}. Switch to ${next}`} title={`Theme: ${mode}`}>
      <Icon className="size-4.5" />
    </button>
  )
}
