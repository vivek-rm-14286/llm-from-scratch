import manifest from './generated/manifest.json'

export type LessonRef = { route: string; label: string }
export type Module = { code: string; title: string; blurb: string; lessons: LessonRef[] }
export type Track = { title: string; code: string; blurb: string; modules: Module[] }
export type PageMeta = {
  src: string; label: string; title: string; id: string
  module: string | null; moduleTitle: string | null; index: number | null
  headings: { id: string; level: number; text: string }[]
  hasMath: boolean; hasMermaid: boolean; hasScript: boolean; hasTasks: boolean; minutes: number
  usedIn?: { route: string; title: string; note: string }[]
  tag?: string
}

export const SITE = manifest.site as { title: string; description: string; repo: string }
export const TRACKS = manifest.tracks as Track[]
export const REFERENCE = manifest.reference as LessonRef[]
export const PAGES = manifest.pages as unknown as Record<string, PageMeta>
/** Old routes that moved (kept so shared links keep working). */
export const REDIRECTS = (manifest as { redirects?: Record<string, string> }).redirects ?? {}
/** Curriculum order (lessons only, no reference pages). */
export const LESSON_ORDER: string[] = TRACKS.flatMap((t) => t.modules.flatMap((m) => m.lessons.map((l) => l.route)))
/** Full reading order including reference pages, used for prev/next. */
export const ORDER: string[] = manifest.order as string[]

export function moduleByCode(code: string | null): Module | undefined {
  if (!code) return undefined
  for (const t of TRACKS) for (const m of t.modules) if (m.code === code) return m
  return undefined
}

export function neighbours(route: string) {
  const i = ORDER.indexOf(route)
  return { prev: i > 0 ? ORDER[i - 1] : undefined, next: i >= 0 && i < ORDER.length - 1 ? ORDER[i + 1] : undefined }
}

const loaders = import.meta.glob('./generated/pages/*.html', { query: '?raw', import: 'default' }) as Record<string, () => Promise<string>>
export function loadPageHtml(meta: PageMeta): Promise<string> {
  const l = loaders[`./generated/pages/${meta.id}.html`]
  return l ? l() : Promise.resolve('<p>Missing page.</p>')
}

export const routeHref = (route: string) => `/${route}`

export function fmtMinutes(m: number): string {
  return m < 60 ? `${m} min` : `${Math.floor(m / 60)} h ${m % 60 ? `${m % 60} min` : ''}`.trim()
}
