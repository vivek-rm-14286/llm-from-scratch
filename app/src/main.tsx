import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { HashRouter, Navigate, Route, Routes, useLocation } from 'react-router-dom'
import { MotionConfig } from 'framer-motion'
import './index.css'
import { Layout } from './app/Layout'
import { Home } from './app/Home'
import { PageView } from './lesson/PageView'
import { PAGES } from './content'

function PageRoute() {
  const { pathname } = useLocation()
  const route = decodeURIComponent(pathname).replace(/^\/+|\/+$/g, '')
  const meta = PAGES[route]
  if (!meta) return <Navigate to="/" replace />
  return <PageView key={route} route={route} meta={meta} />
}

// Reuse the root if this module is re-executed (Vite HMR), instead of calling createRoot twice on one container.
const container = document.getElementById('root')! as HTMLElement & { __root?: ReturnType<typeof createRoot> }
container.__root ??= createRoot(container)
container.__root.render(
  <StrictMode>
    <MotionConfig reducedMotion="user">
      <HashRouter>
        <Layout>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="*" element={<PageRoute />} />
          </Routes>
        </Layout>
      </HashRouter>
    </MotionConfig>
  </StrictMode>,
)
