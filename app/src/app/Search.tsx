import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { CornerDownLeft, Search as SearchIcon } from 'lucide-react'
import { PAGES } from '../content'

type Rec = { r: string; t: string; h: string; a: string; x: string }
let cache: Promise<Rec[]> | null = null
const loadIndex = () => (cache ??= import('../generated/search.json').then((m) => m.default as Rec[]))

function score(rec: Rec, terms: string[]): number {
  const t = rec.t.toLowerCase(), h = rec.h.toLowerCase(), x = rec.x.toLowerCase()
  let s = 0
  for (const q of terms) {
    let hit = 0
    if (t.includes(q)) hit += 6
    if (h.includes(q)) hit += 4
    if (x.includes(q)) hit += 1 + Math.min(3, x.split(q).length - 2) * 0.3
    if (!hit) return 0
    s += hit
  }
  return s
}

function snippet(x: string, terms: string[]) {
  const lx = x.toLowerCase()
  const i = Math.max(0, ...terms.map((q) => lx.indexOf(q)).filter((i) => i >= 0).slice(0, 1))
  const start = Math.max(0, i - 60)
  return (start ? '…' : '') + x.slice(start, start + 180) + (start + 180 < x.length ? '…' : '')
}

export function Search({ onClose }: { onClose: () => void }) {
  const [q, setQ] = useState('')
  const [idx, setIdx] = useState<Rec[] | null>(null)
  const [sel, setSel] = useState(0)
  const nav = useNavigate()
  const input = useRef<HTMLInputElement>(null)
  const list = useRef<HTMLUListElement>(null)
  useEffect(() => { void loadIndex().then(setIdx); input.current?.focus() }, [])
  // Esc closes from anywhere; the selected row stays in view while arrowing
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])
  useEffect(() => { list.current?.querySelector('[data-sel="true"]')?.scrollIntoView({ block: 'nearest' }) }, [sel])

  const terms = useMemo(() => q.toLowerCase().split(/\s+/).filter((w) => w.length > 1), [q])
  const results = useMemo(() => {
    if (!idx || !terms.length) return []
    return idx.map((r) => [score(r, terms), r] as const).filter(([s]) => s > 0).sort((a, b) => b[0] - a[0]).slice(0, 30).map(([, r]) => r)
  }, [idx, terms])

  const go = (r: Rec) => { onClose(); nav(`/${r.r}${r.a ? `#${r.a}` : ''}`) }

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center bg-black/50 px-4 pt-[10vh]" onClick={onClose}>
      <div className="flex max-h-[75vh] w-full max-w-2xl flex-col overflow-hidden rounded-2xl border border-line bg-surface shadow-2xl" onClick={(e) => e.stopPropagation()} role="dialog" aria-modal="true" aria-label="Search">
        <div className="flex items-center gap-3 border-b border-line px-4">
          <SearchIcon className="size-4 text-muted" />
          <input ref={input} value={q} onChange={(e) => { setQ(e.target.value); setSel(0) }} placeholder="Search lessons, e.g. rope, dpo, kv cache…"
            className="h-12 flex-1 bg-transparent text-[15px] outline-none placeholder:text-muted"
            onKeyDown={(e) => {
              if (e.key === 'ArrowDown') { e.preventDefault(); setSel((s) => Math.min(s + 1, results.length - 1)) }
              else if (e.key === 'ArrowUp') { e.preventDefault(); setSel((s) => Math.max(s - 1, 0)) }
              else if (e.key === 'Enter' && results[sel]) go(results[sel])
            }} />
          <kbd className="rounded border border-line px-1.5 font-mono text-[10px] text-muted">esc</kbd>
        </div>
        <ul ref={list} className="overflow-y-auto p-2">
          {!idx && <li className="px-3 py-6 text-center text-sm text-muted">Loading index…</li>}
          {idx && terms.length > 0 && results.length === 0 && <li className="px-3 py-6 text-center text-sm text-muted">No results for “{q}”.</li>}
          {idx && terms.length === 0 && <li className="px-3 py-6 text-center text-sm text-muted">Type to search {Object.keys(PAGES).length} pages.</li>}
          {results.map((r, i) => {
            const m = PAGES[r.r]
            return (
              <li key={`${r.r}#${r.a}`}>
                <button data-sel={i === sel} onClick={() => go(r)} onMouseEnter={() => setSel(i)}
                  className={`w-full rounded-lg px-3 py-2.5 text-left ${i === sel ? 'bg-accent-soft' : ''}`}>
                  <span className="flex items-center gap-2">
                    {m?.module && <span className="font-mono text-[10px] font-bold text-accent">{m.module}</span>}
                    <span className="text-sm font-semibold">{r.t}</span>
                    {r.h && <span className="truncate text-sm text-muted">› {r.h}</span>}
                    {i === sel && <CornerDownLeft className="ml-auto size-3.5 shrink-0 text-muted" />}
                  </span>
                  {r.x && <span className="mt-0.5 block text-[13px] leading-snug text-muted">{snippet(r.x, terms)}</span>}
                </button>
              </li>
            )
          })}
        </ul>
      </div>
    </div>
  )
}
