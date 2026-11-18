// Static Mermaid.js viewer: Mermaid source remains separately editable.
const diagramFiles = [
  ["hld", "./diagrams/High-Level-Architecture.mmd"],
  ["lld", "./diagrams/Low-Level-Architecture.mmd"],
  ["lifecycle", "./diagrams/Training-and-Checkpoint-Lifecycle.mmd"],
  ["deployment", "./diagrams/GitHub-Pages-Auto-Deployment.mmd"],
];

async function init() {
  let mermaid;
  try {
    const library = await import('https://cdn.jsdelivr.net/npm/mermaid@11.12.2/dist/mermaid.esm.min.mjs');
    mermaid = library.default;
    mermaid.initialize({
      startOnLoad: false,
      securityLevel: 'strict',
      theme: 'neutral',
      flowchart: { useMaxWidth: true, htmlLabels: true, curve: 'basis' },
      maxTextSize: 60000,
    });
  } catch (error) {
    showError(`Could not load Mermaid.js. Check the network connection. ${error.message}`);
    return;
  }

  try {
    for (const [id, sourcePath] of diagramFiles) {
      const response = await fetch(sourcePath, {cache: 'no-cache'});
      if (!response.ok) throw new Error(`${sourcePath}: HTTP ${response.status}`);
      const code = await response.text();
      const node = document.querySelector(`[data-mermaid-id="${id}"]`);
      node.textContent = code;
      node.classList.add('mermaid');
      const sourceNode = document.getElementById(`${id}-source`);
      if (sourceNode) sourceNode.textContent = code;
    }
    await mermaid.run({ querySelector: '.mermaid', suppressErrors: false });
    document.getElementById('status').textContent = 'Diagrams loaded successfully';
    document.getElementById('status').classList.add('ready');
  } catch (error) {
    showError(`Diagram rendering failed: ${error.message}`);
  }
}

function showError(message) {
  const node = document.getElementById('status');
  node.textContent = message;
  node.classList.add('error');
}

for (const button of document.querySelectorAll('[data-copy]')) {
  button.addEventListener('click', async () => {
    const sourceNode = document.getElementById(`${button.dataset.copy}-source`);
    try {
      await navigator.clipboard.writeText(sourceNode.textContent);
      const original = button.textContent;
      button.textContent = 'Copied';
      setTimeout(() => { button.textContent = original; }, 1600);
    } catch (error) {
      button.textContent = 'Use source download';
    }
  });
}

init();
