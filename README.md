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

## What You Can Do

- Capture live performer input from webcam
- Extract face landmarks and expression cues
- Extract hand landmarks and pinch/lift gestures
- Compose overlay and mapping stages
- Monitor reactive stage output in PiP
- Monitor runtime metrics (FPS, frame cost, CPU proxy load, heap memory, WebGPU availability)
- Toggle debug preview per node
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

## Notes

- Camera permission is required.
- Models are loaded from official MediaPipe model storage.
- CPU percentage shown in monitor is a main-thread frame-time proxy, not OS-level CPU usage.
