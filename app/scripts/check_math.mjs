// Compile every math block in the generated pages with KaTeX in strict mode; exit 1 on any error.
import fs from 'node:fs'
import path from 'node:path'
import katex from 'katex'

const dir = new URL('../src/generated/pages/', import.meta.url).pathname
const decode = (s) => s.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&amp;/g, '&')
let n = 0
const bad = []
for (const f of fs.readdirSync(dir)) {
  const html = fs.readFileSync(path.join(dir, f), 'utf8')
  for (const m of html.matchAll(/<(span|div) class="arithmatex">([\s\S]*?)<\/\1>/g)) {
    n++
    const tex = decode(m[2]).trim().replace(/^\\\(|\\\)$/g, '').replace(/^\\\[|\\\]$/g, '').trim()
    try { katex.renderToString(tex, { displayMode: m[1] === 'div', throwOnError: true, strict: 'error' }) }
    catch (e) { bad.push(`${f}: ${String(e.message).slice(0, 160)}`) }
  }
}
console.log(`math: ${n} blocks, ${bad.length} errors`)
if (bad.length) { console.error(bad.join('\n')); process.exit(1) }
