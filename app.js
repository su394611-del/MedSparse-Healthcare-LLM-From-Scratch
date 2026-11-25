/* MedSparse Healthcare project site: Vue 3 navigation and Mermaid diagram explorer.
   All essential links and descriptions are static HTML for indexing and
   keyboard accessibility. Diagrams have downloadable, local .mmd sources. */
(() => {
  'use strict';
  const diagrams = {
    hld: {
      file: 'High-Level-Architecture.mmd',
      title: 'High-Level Architecture',
      description: 'Biomedical datasets, audited tokenizer, random-initialized pretraining, full medical SFT and validation.'
    },
    lld: {
      file: 'Low-Level-Architecture.mmd',
      title: 'Low-Level Architecture',
      description: 'RMSNorm, rotary embeddings, grouped-query attention, top-2 sparse routing and SwiGLU experts.'
    },
    lifecycle: {
      file: 'Training-and-Checkpoint-Lifecycle.mmd',
      title: 'Training & Checkpoint Lifecycle',
      description: 'Checkpoint transfer between pretraining and SFT, correct resume semantics, and evaluation.'
    }
  };
  let mermaidPromise = null;
  let renderSerial = 0;
  let activeKey = 'hld';
  const sourceCache = new Map();

  function setStatus(message, isError = false) {
    const status = document.getElementById('diagram-message');
    if (!status) return;
    status.textContent = message;
    status.hidden = !message;
    status.style.color = isError ? '#ffc69b' : '';
  }

  async function loadMermaid() {
    if (!mermaidPromise) {
      mermaidPromise = import('https://cdn.jsdelivr.net/npm/mermaid@11.17.2/dist/mermaid.esm.min.mjs').then(module => {
        const mermaid = module.default;
        mermaid.initialize({
          startOnLoad: false,
          securityLevel: 'strict',
          theme: 'dark',
          themeVariables: {
            fontFamily: 'Inter, ui-sans-serif, system-ui, sans-serif',
            primaryColor: '#17334d',
            primaryTextColor: '#e9f3ff',
            primaryBorderColor: '#5688a4',
            lineColor: '#86cbe9',
            textColor: '#e9f3ff',
            clusterBkg: '#142c47',
            clusterBorder: '#496a89'
          },
          htmlLabels: false,
          flowchart: { curve: 'basis', htmlLabels: false }
        });
        return mermaid;
      });
    }
    return mermaidPromise;
  }

  async function loadSource(key) {
    if (sourceCache.has(key)) return sourceCache.get(key);
    const diagram = diagrams[key];
    const response = await fetch(`./diagrams/${diagram.file}`, { cache: 'no-cache' });
    if (!response.ok) throw new Error(`HTTP ${response.status} while loading ${diagram.file}`);
    const source = await response.text();
    sourceCache.set(key, source);
    return source;
  }

  async function displayDiagram(key) {
    if (!Object.hasOwn(diagrams, key)) return;
    activeKey = key;
    const requestId = ++renderSerial;
    const meta = diagrams[key];
    const heading = document.getElementById('diagram-title');
    const description = document.getElementById('diagram-description');
    const download = document.getElementById('diagram-download');
    const canvas = document.getElementById('diagram-svg');
    const sourceView = document.getElementById('diagram-source');
    const help = document.getElementById('diagram-help');
    if (heading) heading.textContent = meta.title;
    if (description) description.textContent = meta.description;
    if (download) download.href = `./diagrams/${meta.file}`;
    if (canvas) canvas.replaceChildren();
    setStatus('Loading ' + meta.title.toLowerCase() + '…');
    if (sourceView) sourceView.textContent = 'Loading Mermaid source…';

    try {
      const source = await loadSource(key);
      if (requestId !== renderSerial) return;
      if (sourceView) sourceView.textContent = source;
      const mermaid = await loadMermaid();
      if (requestId !== renderSerial) return;
      const result = await mermaid.render(`mixtral_${key}_${requestId}`, source);
      if (requestId !== renderSerial) return;
      if (canvas) {
        const svg = new DOMParser().parseFromString(result.svg, 'image/svg+xml').documentElement;
        if (svg.localName !== 'svg') throw new Error('Invalid diagram image');
        const bounds = svg.viewBox.baseVal;
        svg.setAttribute('width', String(bounds.width));
        svg.setAttribute('height', String(bounds.height));
        const image = document.createElement('img');
        image.alt = meta.title + ' diagram';
        image.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(new XMLSerializer().serializeToString(svg));
        canvas.replaceChildren(image);
      }
      setStatus('');
      if (help) help.textContent = 'Rendered with Mermaid.js. Open “View the editable Mermaid.js source” or download the .mmd file.';
    } catch (error) {
      if (requestId !== renderSerial) return;
      setStatus('The interactive diagram could not be rendered. Download its .mmd source or open the project Wiki.', true);
      if (help) help.textContent = `Details: ${error instanceof Error ? error.message : String(error)}`;
    }
  }

  async function copySource() {
    const pre = document.getElementById('diagram-source');
    const help = document.getElementById('diagram-help');
    if (!pre || !pre.textContent || pre.textContent.startsWith('Loading')) return;
    try {
      await navigator.clipboard.writeText(pre.textContent);
      if (help) help.textContent = 'Mermaid.js source copied to clipboard.';
    } catch (_) {
      document.querySelector('.diagram-source-details')?.setAttribute('open', '');
      if (help) help.textContent = 'Clipboard access is unavailable. The Mermaid source is visible below for manual copying.';
    }
  }

  document.querySelector('.menu-button')?.addEventListener('click', () => {
    const nav = document.querySelector('.mobile-nav');
    const isOpen = nav?.classList.toggle('is-open');
    document.querySelector('.menu-button')?.setAttribute('aria-expanded', String(Boolean(isOpen)));
  });
  const tabs = Array.from(document.querySelectorAll('[data-diagram]'));
  const panel = document.getElementById('diagram-render');

  function setTabStop(target) {
    tabs.forEach(tab => {
      tab.tabIndex = tab === target ? 0 : -1;
    });
  }

  function markTab(target) {
    setTabStop(target);
    for (const tab of tabs) {
      const selected = tab === target;
      tab.classList.toggle('active', selected);
      tab.setAttribute('aria-selected', String(selected));
    }
    panel?.setAttribute('aria-labelledby', target.id);
  }

  for (const [index, tab] of tabs.entries()) {
    tab.addEventListener('click', () => {
      markTab(tab);
      displayDiagram(tab.dataset.diagram);
    });
    tab.addEventListener('keydown', event => {
      const offsets = { ArrowLeft: -1, ArrowRight: 1 };
      let next;
      if (event.key === 'Home') next = tabs[0];
      else if (event.key === 'End') next = tabs[tabs.length - 1];
      else if (Object.hasOwn(offsets, event.key)) {
        next = tabs[(index + offsets[event.key] + tabs.length) % tabs.length];
      }
      if (!next) return;
      event.preventDefault();
      setTabStop(next);
      next.focus();
    });
  }
  const initialTab = tabs.find(tab => tab.dataset.diagram === activeKey);
  if (initialTab) markTab(initialTab);
  document.getElementById('copy-diagram-button')?.addEventListener('click', copySource);
  displayDiagram(activeKey);
})();
