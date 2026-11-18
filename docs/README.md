# Architecture diagrams for GitHub Pages

This directory contains the published static site (`index.html`), styles, an ES module renderer, and editable `.mmd` files. The site loads Mermaid.js from a pinned CDN version.

- [High-level diagram](diagrams/High-Level-Architecture.mmd)
- [Low-level diagram](diagrams/Low-Level-Architecture.mmd)
- [Training lifecycle](diagrams/Training-and-Checkpoint-Lifecycle.mmd)
- [Automatic publishing](diagrams/GitHub-Pages-Auto-Deployment.mmd)

The deployment workflow lives at `.github/workflows/deploy-pages.yml`. GitHub Pages must be configured to use GitHub Actions as its publishing source. After setup, pushing changes to `docs/**` on `main` republishes the site automatically.

## Content scope

Stage 1 builds the native sparse-MoE from random parameters; Stage 2 fully fine-tunes it from the Stage 1 checkpoint. The diagrams are architecture and workflow illustrations, not proof of a completed clinical evaluation.
