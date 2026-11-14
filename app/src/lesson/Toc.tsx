import { useEffect, useState } from 'react'

type Section = { id: string; text: string }

/** "On this page" rail for wide screens, with the section in view highlighted. */
export function Toc({ route, sections, ready }: { route: string; sections: Section[]; ready: boolean }) {
  const [active, setActive] = useState<string>('')

  useEffect(() => {
    if (!ready) return
    let raf = 0
    const update = () => {
      raf = 0
      let cur = ''
      for (const s of sections) {
        const el = document.getElementById(s.id)
        if (el && el.getBoundingClientRect().top <= 120) cur = s.id
      }
      setActive(cur)
    }
    const onScroll = () => { if (!raf) raf = requestAnimationFrame(update) }
    update()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => { window.removeEventListener('scroll', onScroll); if (raf) cancelAnimationFrame(raf) }
  }, [sections, ready])

  return (
    <aside className="hidden w-56 shrink-0 min-[1400px]:block print:hidden" aria-label="On this page">
      <div className="sticky top-20 max-h-[calc(100vh-6rem)] overflow-y-auto pt-8 pr-4 pb-8 scrollbar-thin">
        <p className="mb-2 text-[11px] font-bold tracking-wider text-muted uppercase">On this page</p>
        <ol className="border-l border-line">
          {sections.map((s, i) => (
            <li key={s.id}>
              <a href={`#/${route}#${s.id}`}
                className={`-ml-px flex gap-2 border-l-2 py-1 pr-1 pl-3 text-[12.5px] leading-snug ${active === s.id ? 'border-accent font-semibold text-accent' : 'border-transparent text-muted hover:text-ink'}`}>
                <span className="font-mono text-[10px] leading-[1.6] opacity-70">{i + 1}</span>
                <span>{s.text}</span>
              </a>
            </li>
          ))}
        </ol>
      </div>
    </aside>
  )
}
