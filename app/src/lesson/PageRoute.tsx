import { Link, Navigate, useLocation } from 'react-router-dom'
import { PAGES, REDIRECTS } from '../content'
import { PageView } from './PageView'

export function PageRoute() {
  const { pathname } = useLocation()
  const route = decodeURIComponent(pathname).replace(/^\/+|\/+$/g, '')
  const meta = PAGES[route]
  if (!meta && REDIRECTS[route]) return <Navigate to={`/${REDIRECTS[route]}`} replace />
  if (!meta) return <NotFound route={route} />
  return <PageView key={route} route={route} meta={meta} />
}

function NotFound({ route }: { route: string }) {
  return (
    <div className="mx-auto max-w-4xl px-4 pt-16 pb-24 sm:px-8">
      <p className="font-mono text-xs text-accent">404</p>
      <h1 className="mt-1 text-3xl font-bold tracking-tight">Page not found</h1>
      <p className="mt-4 text-muted">There is no page at <code className="inline-code">/{route}</code>. It may have moved when the curriculum was reorganised.</p>
      <div className="mt-6 flex flex-wrap gap-3">
        <Link to="/" className="inline-flex items-center rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-white hover:opacity-90">Go to the curriculum</Link>
        <button onClick={() => window.dispatchEvent(new Event('open-search'))} className="inline-flex items-center rounded-lg border border-line px-4 py-2 text-sm font-semibold hover:border-accent">Search the site</button>
      </div>
    </div>
  )
}
