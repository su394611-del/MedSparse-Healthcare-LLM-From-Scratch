# Publish the Mixtral Healthcare Vue.js project page

This package replaces only `docs/index.html`. Keep the existing `docs/diagrams/*.mmd` files because the page fetches these Mermaid sources at runtime. Existing `docs/assets/*` may remain in the repository; this page does not require them.

## Windows PowerShell

1. Open your local Git clone of `su394611-del/Mixtral-Healthcare-LLM-From-Scratch`.
2. Extract the ZIP in that clone's root folder, allowing `docs/index.html` to be replaced.
3. Run:

```powershell
git status
git add docs/index.html
git commit -m "Publish responsive Vue project page with Hugging Face models and Mermaid diagrams"
git push origin main
```

The existing `.github/workflows/deploy-pages.yml` deploys the `docs/` directory on each change to `docs/**`. Make sure GitHub Pages is configured to use **GitHub Actions**.

Published URL: https://su394611-del.github.io/Mixtral-Healthcare-LLM-From-Scratch/

Model links:
- https://huggingface.co/sudhanshu42844/mixtral-healthcare-pretrained-from-scratch
- https://huggingface.co/sudhanshu42844/mixtral-healthcare-medical-sft

Mermaid diagram source paths referenced by the page:
- `docs/diagrams/High-Level-Architecture.mmd`
- `docs/diagrams/Low-Level-Architecture.mmd`
- `docs/diagrams/Training-and-Checkpoint-Lifecycle.mmd`

Notes: Model training step counts are documented targets, not assertions of completed runs. The page does not disclose personal device details. Vue and Mermaid are loaded over HTTPS from public CDN packages; ordinary content and links remain available if those scripts cannot load.
