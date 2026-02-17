# MediaPipe Node Lab

Node-based browser prototype for dancers and media artists:

- webcam input node
- face extraction node (MediaPipe Face Landmarker)
- hand extraction node (MediaPipe Hand Landmarker)
- gesture mapping node
- stage output node with reactive visuals

## Tech Stack

- TypeScript
- Vite
- LiteGraph.js
- MediaPipe Tasks Vision

## Run (Dev)

```bash
cd /Users/taeyang/Developer/mediapipe-node-lab
npm install
npm run dev
```

Open:

`http://localhost:4173`

## Build

```bash
npm run build
npm run preview
```

## Graph Overview

Default graph is auto-created on startup:

`Webcam Source -> Face Extract`
`Webcam Source -> Hand Extract`
`(Webcam + Face + Hand) -> Landmark Overlay`
`(Face metrics + Hand metrics) -> Gesture Mapper -> Stage Output`

If the graph ever disappears, click `Reset Graph`.

## Notes

- Browser camera permission is required.
- Models are loaded from official MediaPipe model storage:
  - face: `face_landmarker.task`
  - hand: `hand_landmarker.task`
