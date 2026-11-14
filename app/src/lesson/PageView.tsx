import { useEffect, useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { ArrowLeft, ArrowRight, Check } from 'lucide-react'
import { PAGES, loadPageHtml, neighbours, routeHref, type PageMeta } from '../content'
import { setDone, useProgress } from '../app/progress'
import { Prose } from './Prose'

export function PageView({ route, meta }: { route: string; meta: PageMeta }) {
  const [html, setHtml] = useState<string | null>(null)
  const { hash } = useLocation()
  const done = useProgress().has(route)
  const { prev, next } = neighbours(route)
  const isLesson = meta.module !== null

  useEffect(() => { document.title = `${meta.title} · LLM From Scratch` }, [meta])
  useEffect(() => { let live = true; loadPageHtml(meta).then((h) => { if (live) setHtml(h) }); return () => { live = false } }, [meta])

  // scroll to the anchor (or top) once the content is in the DOM, and on in-page anchor changes
  useEffect(() => {
    if (html === null) return
    if (hash.length > 1) {
      const el = document.getElementById(decodeURIComponent(hash.slice(1)))
      if (el) { el.scrollIntoView({ block: 'start' }); return }
    }
    window.scrollTo(0, 0)
  }, [html, hash])

  return (
    <article className="mx-auto w-full min-w-0 max-w-4xl px-4 pb-24 sm:px-8" data-testid="page" data-route={route}>
      <header className="pt-8 pb-6">
        {isLesson && <p className="font-mono text-xs text-accent">{meta.module} · {meta.moduleTitle} · {meta.index === 0 ? 'Overview' : `Lesson ${meta.index}`}</p>}
        <h1 className="mt-1 text-3xl font-bold tracking-tight sm:text-4xl">{meta.title}</h1>
      </header>

      {html === null ? <p className="font-mono text-xs text-muted">Loading…</p> : <Prose html={html} meta={meta} />}

      <footer className="mt-12 grid gap-6">
        {isLesson && (
          <div className="flex flex-wrap items-center gap-3">
            <button onClick={() => setDone(route, !done)}
              className={`inline-flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-semibold transition ${done ? 'bg-ok/15 text-ok' : 'bg-accent text-white hover:opacity-90'}`}>
              <Check className="size-4" /> {done ? 'Completed' : 'Mark lesson complete'}
            </button>
          </div>
        )}
        <nav className="grid gap-3 sm:grid-cols-2" aria-label="Lesson navigation">
          {prev ? <NavCard to={routeHref(prev)} dir="prev" title={PAGES[prev].title} /> : <span />}
          {next && <NavCard to={routeHref(next)} dir="next" title={PAGES[next].title} />}
        </nav>
      </footer>
    </article>
  )
}

function NavCard({ to, dir, title }: { to: string; dir: 'prev' | 'next'; title: string }) {
  return (
    <Link to={to} className={`group rounded-xl border border-line bg-surface px-4 py-3 hover:border-accent ${dir === 'next' ? 'text-right sm:col-start-2' : ''}`}>
      <span className="flex items-center gap-1 text-xs text-muted" style={{ justifyContent: dir === 'next' ? 'flex-end' : 'flex-start' }}>
        {dir === 'prev' ? <><ArrowLeft className="size-3" /> Previous</> : <>Next <ArrowRight className="size-3" /></>}
      </span>
      <span className="font-semibold group-hover:text-accent">{title}</span>
    </Link>
  )
}
