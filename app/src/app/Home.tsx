import { useEffect } from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { ArrowRight, Code, Cpu, FlaskConical, Layers } from 'lucide-react'
import { LESSON_ORDER, TRACKS, routeHref } from '../content'
import { useProgress } from './progress'

export function Home() {
  const done = useProgress()
  useEffect(() => { document.title = 'LLM From Scratch: learn to build language models' }, [])
  const resume = LESSON_ORDER.find((r) => !done.has(r)) ?? LESSON_ORDER[0]
  return (
    <div className="mx-auto max-w-5xl px-4 pb-24 sm:px-8">
      <section className="pt-12 pb-10">
        <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="font-mono text-sm text-accent">PyTorch · from your first gradient to a 500M-parameter model and beyond</motion.p>
        <motion.h1 initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="mt-2 max-w-3xl text-4xl font-bold tracking-tight sm:text-5xl">
          Learn to build language models, from first principles to production.
        </motion.h1>
        <p className="mt-4 max-w-2xl text-lg leading-relaxed text-muted">
          A hands-on textbook in five tracks: <b className="text-ink">Fundamentals</b>, <b className="text-ink">Training Models</b>, <b className="text-ink">Applying Models</b>,{' '}
          <b className="text-ink">Decision Models</b> and <b className="text-ink">Projects</b>. Every lesson goes from intuition to math to <b className="text-ink">runnable PyTorch</b>,
          and four capstone projects take you from a <b className="text-ink">500M SLM trained from scratch</b> to RAG, fine-tuning and a policy that plays a game.
        </p>
        <div className="mt-7 flex flex-wrap gap-3">
          <Link to={routeHref(resume)} className="inline-flex items-center gap-2 rounded-lg bg-accent px-5 py-2.5 font-semibold text-white hover:opacity-90">
            {done.size ? 'Continue' : 'Start learning'} <ArrowRight className="size-4" />
          </Link>
          <Link to={routeHref('projects')} className="inline-flex items-center gap-2 rounded-lg border border-line px-5 py-2.5 font-semibold hover:border-accent"><FlaskConical className="size-4" /> Jump to the projects</Link>
          <Link to={routeHref('glossary')} className="inline-flex items-center gap-2 rounded-lg border border-line px-5 py-2.5 font-semibold hover:border-accent">Browse the glossary</Link>
        </div>
        <div className="mt-10 grid gap-3 sm:grid-cols-3">
          {[
            [Code, 'Built by hand', 'Backprop, BPE, attention, RoPE, LoRA, DPO, PPO and RAG, each implemented from scratch before you reach for a library.'],
            [Cpu, 'Runs on your machine', 'Every snippet was executed. Tiny presets run on a laptop; each page says when you need a GPU, and how big.'],
            [Layers, 'Same beats every lesson', "What you'll learn, intuition, math, code, pitfalls, check-your-understanding and further reading."],
          ].map(([Icon, t, d]) => {
            const I = Icon as typeof Cpu
            return (
              <div key={t as string} className="rounded-xl border border-line bg-surface p-4">
                <I className="size-5 text-accent" />
                <p className="mt-2 font-semibold">{t as string}</p>
                <p className="mt-1 text-sm text-muted">{d as string}</p>
              </div>
            )
          })}
        </div>
      </section>

      {TRACKS.map((t) => (
        <section key={t.title} aria-label={t.title} className="mt-10 grid gap-4">
          <div>
            <h2 className="text-2xl font-bold tracking-tight">{t.title}</h2>
            <p className="mt-1 text-muted">{t.blurb}</p>
          </div>
          {t.modules.map((m, i) => {
            const n = m.lessons.filter((l) => done.has(l.route)).length
            return (
              <motion.div key={m.code} initial={{ opacity: 0, y: 10 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: i * 0.03 }}
                className="rounded-2xl border border-line bg-surface p-5">
                <div className="flex flex-wrap items-baseline gap-x-3">
                  <span className="font-mono text-sm font-bold text-accent">{m.code}</span>
                  <h3 className="text-xl font-bold tracking-tight">{m.title}</h3>
                  <span className="ml-auto font-mono text-xs text-muted">{n}/{m.lessons.length} done</span>
                </div>
                <p className="mt-1 text-sm text-muted">{m.blurb}</p>
                <ol className="mt-3 flex flex-wrap gap-2">
                  {m.lessons.map((l, j) => (
                    <li key={l.route}>
                      <Link to={routeHref(l.route)} className={`inline-block rounded-full border px-3 py-1 text-[13px] hover:border-accent hover:text-accent ${done.has(l.route) ? 'border-ok/50 bg-ok/10' : 'border-line'}`}>
                        {j > 0 ? `${j}. ` : ''}{l.label}
                      </Link>
                    </li>
                  ))}
                </ol>
              </motion.div>
            )
          })}
        </section>
      ))}
    </div>
  )
}
