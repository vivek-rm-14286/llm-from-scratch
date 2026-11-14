/* Theme toggle (system → dark → light) and per-viewer lesson progress.
   localStorage only; the site works fully without it. The theme key is shared
   with "Postgres, Visualized" on the same origin, so both sites stay in sync. */
(function () {
  var KEY = "llmfs-progress-v1";

  function readMode() {
    try { var t = localStorage.getItem("theme"); return t === "light" || t === "dark" ? t : "system"; } catch (e) { return "system"; }
  }
  function isDark(mode) {
    return mode === "dark" || (mode === "system" && window.matchMedia("(prefers-color-scheme: dark)").matches);
  }
  function applyTheme(mode) {
    var root = document.documentElement;
    if (mode === "system") delete root.dataset.theme; else root.dataset.theme = mode;
    if (document.body) document.body.setAttribute("data-md-color-scheme", isDark(mode) ? "slate" : "default");
    document.querySelectorAll("[data-theme-toggle]").forEach(function (b) {
      b.setAttribute("data-mode-current", mode);
      b.title = "Theme: " + mode;
    });
  }
  function setMode(mode) {
    try { if (mode === "system") localStorage.removeItem("theme"); else localStorage.setItem("theme", mode); } catch (e) {}
    applyTheme(mode);
  }

  function loadDone() {
    try { return new Set(JSON.parse(localStorage.getItem(KEY) || "[]")); } catch (e) { return new Set(); }
  }
  function saveDone(done) {
    try { localStorage.setItem(KEY, JSON.stringify(Array.from(done))); } catch (e) {}
  }

  function render() {
    var done = loadDone();
    // sidebar + home pills
    document.querySelectorAll("a[data-lesson]").forEach(function (a) {
      a.classList.toggle("is-done", done.has(a.getAttribute("data-lesson")));
    });
    document.querySelectorAll("[data-module]").forEach(function (m) {
      var links = m.querySelectorAll("a[data-lesson]");
      var n = 0;
      links.forEach(function (a) { if (done.has(a.getAttribute("data-lesson"))) n++; });
      var fill = m.querySelector(".lf-bar-fill");
      if (fill) fill.style.width = links.length ? (100 * n / links.length) + "%" : "0";
      var c = m.querySelector("[data-count]");
      if (c) c.textContent = n + "/" + links.length;
      var cd = m.querySelector("[data-count-done]");
      if (cd) cd.textContent = n + "/" + links.length + " done";
    });
    // header total (unique lessons in the sidebar)
    var all = new Set();
    document.querySelectorAll(".lf-sidebar a[data-lesson]").forEach(function (a) { all.add(a.getAttribute("data-lesson")); });
    var k = 0; all.forEach(function (u) { if (done.has(u)) k++; });
    var pct = all.size ? Math.round(100 * k / all.size) : 0;
    document.querySelectorAll("[data-progress-total]").forEach(function (f) { f.style.width = pct + "%"; });
    document.querySelectorAll("[data-progress-pct]").forEach(function (p) { p.textContent = pct + "%"; });
    // complete button
    document.querySelectorAll("[data-complete]").forEach(function (b) {
      var on = done.has(b.getAttribute("data-complete"));
      b.classList.toggle("is-done", on);
      var s = b.querySelector("span"); if (s) s.textContent = on ? "Completed" : "Mark lesson complete";
    });
    // home "continue" button → first unfinished lesson in curriculum order
    var cont = document.querySelector("[data-continue]");
    if (cont) {
      var order = Array.prototype.slice.call(document.querySelectorAll(".lf-sidebar a[data-lesson]"));
      var next = order.find(function (a) { return !done.has(a.getAttribute("data-lesson")); });
      if (next && done.size) {
        cont.setAttribute("href", next.getAttribute("href"));
        cont.firstChild.textContent = "Continue ";
      }
    }
  }

  function init() {
    applyTheme(readMode());
    var mq = window.matchMedia("(prefers-color-scheme: dark)");
    if (mq.addEventListener) mq.addEventListener("change", function () { applyTheme(readMode()); });
    document.addEventListener("click", function (e) {
      var t = e.target.closest("[data-theme-toggle]");
      if (t) {
        var m = readMode();
        setMode(m === "system" ? "dark" : m === "dark" ? "light" : "system");
        return;
      }
      var b = e.target.closest("[data-complete]");
      if (b) {
        var done = loadDone(), key = b.getAttribute("data-complete");
        if (done.has(key)) done.delete(key); else done.add(key);
        saveDone(done);
        render();
      }
    });
    // keep the active lesson visible in the sidebar
    var active = document.querySelector(".lf-sidebar a.is-active");
    if (active && active.scrollIntoView) active.scrollIntoView({ block: "center" });
    render();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
