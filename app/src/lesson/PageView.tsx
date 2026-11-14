import { useEffect, useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { ArrowLeft, ArrowRight, Check, Clock, Hammer, Pencil } from 'lucide-react'
import { PAGES, SITE, fmtMinutes, loadPageHtml, neighbours, routeHref, type PageMeta } from '../content'
import { setDone, useProgress } from '../app/progress'
import { Prose } from './Prose'
import { Toc } from './Toc'

export function PageView({ route, meta }: { route: string; meta: PageMeta }) {
  const [html, setHtml] = useState<string | null>(null)
  const { hash } = useLocation()
  const done = useProgress().has(route)
  const { prev, next } = neighbours(route)
  const isLesson = meta.module !== null
  const sections = meta.headings.filter((h) => h.level === 2)

  useEffect(() => { document.title = `${meta.title} · LLM From Scratch` }, [meta])
  useEffect(() => { let live = true; void loadPageHtml(meta).then((h) => { if (live) setHtml(h) }); return () => { live = false } }, [meta])

  // scroll to the anchor (or top) once the content is in the DOM, and on in-page anchor changes
  useEffect(() => {
    if (html === null) return
    if (hash.length > 1) {
      const el = document.getElementById(decodeURIComponent(hash.slice(1)))
      if (el) { el.scrollIntoView({ block: 'start' }); return }
    }
    window.scrollTo(0, 0)
  }, [html, hash])

  // warm the next lesson's chunk while this one is being read
  useEffect(() => {
    if (html === null || !next) return
    const id = window.setTimeout(() => { void loadPageHtml(PAGES[next]) }, 1500)
    return () => window.clearTimeout(id)
  }, [html, next])

  return (
    <div className="mx-auto flex w-full max-w-[76rem] justify-center gap-8">
      <article className="w-full min-w-0 max-w-4xl px-4 pb-24 sm:px-8" data-testid="page" data-route={route}>
        <header className="pt-8 pb-6">
          <p className="flex flex-wrap items-center gap-x-2 font-mono text-xs text-accent">
            {isLesson && <span>{meta.module} · {meta.moduleTitle} · {meta.index === 0 ? 'Overview' : `Lesson ${meta.index}`}</span>}
            <span className="flex items-center gap-1 text-muted"><Clock className="size-3" /> {fmtMinutes(meta.minutes)} read</span>
          </p>
          <h1 className="mt-1 text-3xl font-bold tracking-tight sm:text-4xl">{meta.title}</h1>
        </header>

        {html === null ? <p className="font-mono text-xs text-muted">Loading…</p> : <Prose html={html} meta={meta} route={route} />}

        {html !== null && meta.usedIn && meta.usedIn.length > 0 && (
          <section className="mt-12 rounded-xl border border-line bg-surface px-5 py-4" aria-label="Used in">
            <p className="mb-2 flex items-center gap-1.5 text-xs font-bold tracking-wider text-header uppercase"><Hammer className="size-3.5" /> Used in</p>
            <ul className="grid gap-1.5 text-[15px] leading-relaxed">
              {meta.usedIn.map((u) => (
                <li key={u.route + u.note}><Link to={routeHref(u.route)} className="font-semibold text-accent hover:underline">{u.title}</Link>{u.note && <span className="text-muted"> · {u.note}</span>}</li>
              ))}
            </ul>
          </section>
        )}

        <footer className="mt-12 grid gap-6 print:hidden">
          <div className="flex flex-wrap items-center gap-3">
            {isLesson && (
              <button onClick={() => setDone(route, !done)}
                className={`inline-flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-semibold transition ${done ? 'bg-ok/15 text-ok' : 'bg-accent text-white hover:opacity-90'}`}>
                <Check className="size-4" /> {done ? 'Completed' : 'Mark lesson complete'}
              </button>
            )}
            <a href={`${SITE.repo}/edit/main/docs/${meta.src}`} target="_blank" rel="noopener" className="ml-auto inline-flex items-center gap-1.5 text-sm text-muted hover:text-ink">
              <Pencil className="size-3.5" /> Edit this page
            </a>
          </div>
          <nav className="grid gap-3 sm:grid-cols-2" aria-label="Lesson navigation">
            {prev ? <NavCard to={routeHref(prev)} dir="prev" title={PAGES[prev].title} /> : <span />}
            {next && <NavCard to={routeHref(next)} dir="next" title={PAGES[next].title} />}
          </nav>
        </footer>
      </article>
      {sections.length > 2 && <Toc route={route} sections={sections} ready={html !== null} />}
    </div>
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
