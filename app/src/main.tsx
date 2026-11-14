import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { HashRouter, Route, Routes } from 'react-router-dom'
import { MotionConfig } from 'framer-motion'
import './index.css'
import { Layout } from './app/Layout'
import { Home } from './app/Home'
import { PageRoute } from './lesson/PageRoute'

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
