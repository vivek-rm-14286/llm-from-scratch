import { memo, useEffect, useLayoutEffect, useRef } from 'react'
import { useTheme } from '../app/theme'
import type { PageMeta } from '../content'

const executedScripts = new Set<string>()
const tok = (n: string) => getComputedStyle(document.documentElement).getPropertyValue(n).trim()

/** Renders a compiled Markdown fragment and progressively enhances it (math, diagrams, code tools). */
export const Prose = memo(function Prose({ html, meta }: { html: string; meta: PageMeta }) {
  const ref = useRef<HTMLDivElement>(null)
  const { dark } = useTheme()

  useLayoutEffect(() => {
    const el = ref.current!
    el.innerHTML = html
    addCopyButtons(el)
    addAnnotations(el)
    if (meta.hasMath) void renderMath(el)
    if (meta.hasScript) runScripts(el)
  }, [html, meta])

  // diagrams depend on the theme, so they re-render when it changes
  useEffect(() => {
    if (meta.hasMermaid) void renderMermaid(ref.current!, dark)
  }, [html, meta, dark])

  // close annotation popovers on outside click
  useEffect(() => {
    const onDoc = (e: MouseEvent) => {
      const t = e.target as HTMLElement
      if (!t.closest('.annot, .annot-pop')) ref.current?.querySelectorAll('.annot-pop').forEach((p) => p.remove())
    }
    document.addEventListener('click', onDoc)
    return () => document.removeEventListener('click', onDoc)
  }, [])

  return <div ref={ref} className="prose-lf" />
})

async function renderMath(el: HTMLElement) {
  const [{ default: katex }] = await Promise.all([import('katex'), import('katex/dist/katex.min.css')])
  el.querySelectorAll<HTMLElement>('.arithmatex').forEach((node) => {
    if (node.dataset.done) return
    const display = node.tagName === 'DIV'
    const raw = (node.textContent ?? '').trim()
    const tex = raw.replace(/^\\\(|\\\)$/g, '').replace(/^\\\[|\\\]$/g, '').trim()
    try { katex.render(tex, node, { displayMode: display, throwOnError: false }) } catch { /* leave source */ }
    node.dataset.done = '1'
  })
}

let mermaidSeq = 0
async function renderMermaid(el: HTMLElement, dark: boolean) {
  const { default: mermaid } = await import('mermaid')
  mermaid.initialize({
    startOnLoad: false,
    securityLevel: 'strict',
    theme: 'base',
    fontFamily: 'Inter, ui-sans-serif, system-ui, sans-serif',
    themeVariables: {
      darkMode: dark,
      background: tok('--surface'),
      primaryColor: tok('--surface-2'),
      primaryTextColor: tok('--text'),
      primaryBorderColor: tok('--accent'),
      secondaryColor: tok('--accent-soft'),
      tertiaryColor: tok('--surface'),
      lineColor: tok('--muted'),
      textColor: tok('--text'),
      mainBkg: tok('--surface-2'),
      nodeBorder: tok('--accent'),
      clusterBkg: tok('--surface'),
      clusterBorder: tok('--border'),
      edgeLabelBackground: tok('--surface'),
      actorBkg: tok('--surface-2'),
      actorBorder: tok('--accent'),
      noteBkgColor: tok('--accent-soft'),
      noteBorderColor: tok('--accent'),
      fontSize: '14px',
    },
  })
  const nodes = [...el.querySelectorAll<HTMLElement>('pre.mermaid, div.mermaid-rendered')]
  for (const node of nodes) {
    const src = node.dataset.src ?? node.textContent ?? ''
    try {
      const { svg } = await mermaid.render(`mmd-${++mermaidSeq}`, src)
      const out = document.createElement('div')
      out.className = 'mermaid-rendered'
      out.dataset.src = src
      out.innerHTML = svg
      node.replaceWith(out)
    } catch {
      node.dataset.src = src
    }
  }
}

function addCopyButtons(el: HTMLElement) {
  el.querySelectorAll<HTMLElement>('div.highlight').forEach((block) => {
    if (block.querySelector(':scope > .copy-btn')) return
    const code = block.querySelector('pre code')
    if (!code) return
    const b = document.createElement('button')
    b.className = 'copy-btn'
    b.type = 'button'
    b.textContent = 'Copy'
    b.setAttribute('aria-label', 'Copy code')
    b.addEventListener('click', () => {
      const text = [...code.childNodes].map((n) => (n as HTMLElement).classList?.contains('annot') ? '' : n.textContent).join('')
      void navigator.clipboard?.writeText(text.replace(/\n$/, '')).then(() => {
        b.textContent = 'Copied'
        setTimeout(() => { b.textContent = 'Copy' }, 1200)
      })
    })
    block.appendChild(b)
  })
}

/** Code annotations: a `# (1)!` comment in a code block refers to item 1 of the list right after the block. */
function addAnnotations(el: HTMLElement) {
  el.querySelectorAll<HTMLElement>('div.highlight').forEach((block) => {
    const list = block.nextElementSibling
    if (!list || list.tagName !== 'OL') return
    const items = [...list.children] as HTMLElement[]
    let used = false
    block.querySelectorAll<HTMLElement>('code span[class^="c"]').forEach((c) => {
      const m = (c.textContent ?? '').match(/^(.*?)\((\d+)\)!\s*$/)
      if (!m) return
      const n = Number(m[2])
      const item = items[n - 1]
      if (!item) return
      used = true
      const rest = m[1].replace(/^\s*(#|\/\/|--)\s*$/, '').trimEnd()
      c.textContent = rest ? rest + ' ' : ''
      const btn = document.createElement('button')
      btn.type = 'button'
      btn.className = 'annot'
      btn.textContent = String(n)
      btn.setAttribute('aria-label', `Annotation ${n}`)
      btn.addEventListener('click', (e) => {
        e.stopPropagation()
        const open = block.querySelector(`.annot-pop[data-n="${n}"]`)
        block.querySelectorAll('.annot-pop').forEach((p) => p.remove())
        if (open) return
        const pop = document.createElement('div')
        pop.className = 'annot-pop'
        pop.dataset.n = String(n)
        pop.innerHTML = item.innerHTML
        const br = block.getBoundingClientRect(), r = btn.getBoundingClientRect()
        pop.style.top = `${r.bottom - br.top + 6}px`
        pop.style.left = `${Math.max(8, Math.min(r.left - br.left - 12, br.width - 328))}px`
        block.appendChild(pop)
      })
      c.after(btn)
    })
    if (used) list.classList.add('annot-list')
  })
}

/** Inline <script> blocks (the in-browser demo) run once per session; later visits re-initialise them. */
function runScripts(el: HTMLElement) {
  el.querySelectorAll('script').forEach((old) => {
    const code = old.textContent ?? ''
    if (executedScripts.has(code)) {
      const w = window as unknown as { __ocInit?: () => void }
      w.__ocInit?.()
      return
    }
    executedScripts.add(code)
    const s = document.createElement('script')
    s.textContent = code
    old.replaceWith(s)
  })
}
