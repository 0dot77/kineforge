# Kineforge

![Kineforge Logo](public/brand/kineforge-logo.svg)

`Kineforge is a node-based live media forge where body motion, AI vision, and GPU visuals are composed in real time.`

Kineforge is built for dancers and media artists who want to prototype interaction pipelines quickly in the browser.

## Brand Assets

- Logo lockup: `public/brand/kineforge-logo.svg`
- Symbol mark: `public/brand/kineforge-mark.svg`
- Canonical slug: `kineforge`
- Main repository: `https://github.com/0dot77/kineforge`

## Tech Stack

- Next.js (App Router)
- TypeScript
- Tailwind CSS v4
- React Flow (`@xyflow/react`)
- MediaPipe Tasks Vision (Face + Hand)
- Cloudflare Pages (static deploy)

## What You Can Do

- Capture live performer input from webcam
- Extract face landmarks and expression cues
- Extract hand landmarks and pinch/lift gestures
- Compose overlay and mapping stages
- Keep final stage output continuously visible in bottom-right PiP
- Keep frame/runtime status pinned in top-right status panel
- Monitor runtime metrics (FPS, frame cost, CPU proxy load, heap memory, WebGPU availability)
- Toggle per-node preview and expand preview panel under each node
- Double-click empty canvas to open node picker and spawn nodes

## Quick Controls

- `Live` icon: request camera access and toggle live processing
- `Trash` icon: clear all nodes and edges from the board
- Double-click empty canvas: open node picker at cursor position

## Development

```bash
git clone https://github.com/0dot77/kineforge.git
cd kineforge
npm install
npm run dev
```

Open: `http://localhost:4173`

## Production Build

```bash
npm run build
npm run start
```

`npm run build` exports a static site to `out/`.

## Cloudflare Pages Deploy

```bash
npm install
npm run build
npx wrangler login
npx wrangler pages project create kineforge --production-branch main
npm run deploy:pages
```

If the project already exists, skip `project create`.

## Notes

- Camera permission is required.
- Models are loaded from official MediaPipe model storage.
- CPU percentage shown in monitor is a main-thread frame-time proxy, not OS-level CPU usage.
