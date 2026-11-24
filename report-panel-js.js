// "Read the technical report" opens the embedded report just below the hero.
  (function () {
    function panel() { return document.getElementById('report'); }
    function setExpanded(v) { document.querySelectorAll('.report-toggle').forEach(function (b) { b.setAttribute('aria-expanded', String(v)); }); }
    function open(scroll) {
      var p = panel(); if (!p) return false;
      var f = p.querySelector('iframe'); if (f && !f.getAttribute('src')) f.setAttribute('src', f.getAttribute('data-src'));
      p.hidden = false; setExpanded(true);
      if (scroll) { var smooth = !window.matchMedia('(prefers-reduced-motion: reduce)').matches; p.scrollIntoView({ behavior: smooth ? 'smooth' : 'auto', block: 'start' }); }
      return true;
    }
    function close() { var p = panel(); if (!p) return; p.hidden = true; setExpanded(false); var b = document.querySelector('.report-toggle'); if (b) b.focus(); }
    document.addEventListener('click', function (e) {
      if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
      var t = e.target.closest && e.target.closest('[data-open-report], [data-close-report]');
      if (!t) return;
      if (t.hasAttribute('data-close-report')) { close(); return; }
      if (t.classList.contains('report-toggle') && !panel().hidden) { e.preventDefault(); close(); return; }
      if (open(true)) e.preventDefault();
    });
    function fromHash() { if (location.hash === '#report') open(true); }
    window.addEventListener('hashchange', fromHash);
    window.addEventListener('load', fromHash);
  })();
