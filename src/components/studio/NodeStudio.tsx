'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Background,
  BackgroundVariant,
  Controls,
  Handle,
  MiniMap,
  Node,
  NodeProps,
  NodeTypes,
  Position,
  ReactFlow,
  useEdgesState,
  useNodesState,
} from '@xyflow/react';
import type { Edge, ReactFlowInstance } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { Activity, Camera, Eye, EyeOff, Grip, Hand, ScanFace, Sparkles, Trash2, WandSparkles } from 'lucide-react';
import type { FaceLandmarker, HandLandmarker } from '@mediapipe/tasks-vision';

type NodeKind = 'source' | 'extract' | 'compose' | 'map' | 'output';

type NodeIcon = 'camera' | 'face' | 'hand' | 'overlay' | 'mapper' | 'stage';

type PreviewKey = 'webcam' | 'face' | 'hand' | 'overlay' | 'mapper' | 'stage';

interface StudioNodeData extends Record<string, unknown> {
  title: string;
  subtitle: string;
  kind: NodeKind;
  icon: NodeIcon;
  accent: string;
  previewSource: PreviewKey;
  metrics: string[];
  previewEnabled: boolean;
  previewUrl?: string;
  previewHint: string;
  hasInput: boolean;
  hasOutput: boolean;
}

type StudioNode = Node<StudioNodeData, 'studio'>;

interface NodeTemplate {
  templateId: string;
  title: string;
  subtitle: string;
  kind: NodeKind;
  icon: NodeIcon;
  accent: string;
  previewSource: PreviewKey;
  hasInput: boolean;
  hasOutput: boolean;
  searchTerms: string;
}

interface NodePickerState {
  open: boolean;
  panelX: number;
  panelY: number;
  flowX: number;
  flowY: number;
  query: string;
}

interface Landmark {
  x: number;
  y: number;
}

interface FaceMetrics {
  count: number;
  noseX: number;
  noseY: number;
  jaw: number;
}

interface HandMetrics {
  count: number;
  pinch: number;
  palmY: number;
}

interface ControlsMetrics {
  tilt: number;
  lift: number;
  pinch: number;
  jaw: number;
  presence: number;
}

interface RuntimeState {
  stream: MediaStream | null;
  rafId: number | null;
  faceLandmarker: FaceLandmarker | null;
  handLandmarker: HandLandmarker | null;
  modelsReady: boolean;
  frameCount: number;
  lastFrameTimestamp: number;
  lastUiUpdate: number;
  frameCostMs: number;
  smoothedControls: ControlsMetrics;
  previewCanvases: Record<PreviewKey, HTMLCanvasElement | null>;
}

interface RuntimeMonitor {
  fps: number;
  frameMs: number;
  loadPercent: number;
  memoryMb: number | null;
  webgpuSupported: boolean;
  webgpuLabel: string;
}

interface WebGpuAdapterInfoLike {
  vendor?: string;
  architecture?: string;
  device?: string;
  description?: string;
}

interface WebGpuAdapterLike {
  requestAdapterInfo?: () => Promise<WebGpuAdapterInfoLike>;
}

interface WebGpuLike {
  requestAdapter: () => Promise<WebGpuAdapterLike | null>;
}

const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm';
const FACE_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task';
const HAND_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task';

const PREVIEW_WIDTH = 320;
const PREVIEW_HEIGHT = 180;

const PREVIEW_HINTS: Record<PreviewKey, string> = {
  webcam: 'Camera preview appears here',
  face: 'Face landmarks preview',
  hand: 'Hand landmarks preview',
  overlay: 'Overlay preview',
  mapper: 'Control map scope',
  stage: 'Final output mirror',
};

const PREVIEW_OFF_HINT = 'Preview off · click eye icon';

const DEFAULT_METRICS_BY_SOURCE: Record<PreviewKey, string[]> = {
  webcam: ['idle', 'camera not started'],
  face: ['faces: 0', 'jaw: 0.000'],
  hand: ['hands: 0', 'pinch: 0%'],
  overlay: ['face mesh off', 'hand mesh off'],
  mapper: ['tilt: 0.00', 'lift: 0.00'],
  stage: ['presence: 0.00', 'radius: 0'],
};

const EMPTY_CONTROLS: ControlsMetrics = {
  tilt: 0,
  lift: 0,
  pinch: 0,
  jaw: 0,
  presence: 0,
};

const HAND_CONNECTIONS: Array<[number, number]> = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [5, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [9, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [13, 17],
  [17, 18],
  [18, 19],
  [19, 20],
  [0, 17],
];

const ICONS = {
  camera: Camera,
  face: ScanFace,
  hand: Hand,
  overlay: Activity,
  mapper: WandSparkles,
  stage: Sparkles,
};

const NODE_LIBRARY: NodeTemplate[] = [
  {
    templateId: 'camera-source',
    title: 'Webcam Source',
    subtitle: 'Live performer feed',
    kind: 'source',
    icon: 'camera',
    accent: '#bef264',
    previewSource: 'webcam',
    hasInput: false,
    hasOutput: true,
    searchTerms: 'camera webcam input source performer',
  },
  {
    templateId: 'face-extract',
    title: 'Face Extract',
    subtitle: 'Landmarks + expression cues',
    kind: 'extract',
    icon: 'face',
    accent: '#a3e635',
    previewSource: 'face',
    hasInput: true,
    hasOutput: true,
    searchTerms: 'face landmark expression extract',
  },
  {
    templateId: 'hand-extract',
    title: 'Hand Extract',
    subtitle: 'Landmarks + pinch strength',
    kind: 'extract',
    icon: 'hand',
    accent: '#84cc16',
    previewSource: 'hand',
    hasInput: true,
    hasOutput: true,
    searchTerms: 'hand pinch gesture extract',
  },
  {
    templateId: 'landmark-overlay',
    title: 'Landmark Overlay',
    subtitle: 'Composed face + hand mesh',
    kind: 'compose',
    icon: 'overlay',
    accent: '#65a30d',
    previewSource: 'overlay',
    hasInput: true,
    hasOutput: true,
    searchTerms: 'overlay compose mesh blend',
  },
  {
    templateId: 'gesture-mapper',
    title: 'Gesture Mapper',
    subtitle: 'Transforms movement to controls',
    kind: 'map',
    icon: 'mapper',
    accent: '#d9f99d',
    previewSource: 'mapper',
    hasInput: true,
    hasOutput: true,
    searchTerms: 'mapper control modulation transform',
  },
  {
    templateId: 'stage-output',
    title: 'Stage Output',
    subtitle: 'Realtime visual monitor feed',
    kind: 'output',
    icon: 'stage',
    accent: '#84cc16',
    previewSource: 'stage',
    hasInput: true,
    hasOutput: false,
    searchTerms: 'output stage monitor final render',
  },
];

const INITIAL_NODES: StudioNode[] = [
  {
    id: 'webcam',
    type: 'studio',
    position: { x: 70, y: 280 },
    data: {
      title: 'Webcam Source',
      subtitle: 'Live performer feed',
      kind: 'source',
      icon: 'camera',
      accent: '#bef264',
      previewSource: 'webcam',
      metrics: [...DEFAULT_METRICS_BY_SOURCE.webcam],
      previewEnabled: false,
      previewHint: PREVIEW_OFF_HINT,
      hasInput: false,
      hasOutput: true,
    },
  },
  {
    id: 'face',
    type: 'studio',
    position: { x: 470, y: 70 },
    data: {
      title: 'Face Extract',
      subtitle: 'Landmarks + expression cues',
      kind: 'extract',
      icon: 'face',
      accent: '#a3e635',
      previewSource: 'face',
      metrics: [...DEFAULT_METRICS_BY_SOURCE.face],
      previewEnabled: false,
      previewHint: PREVIEW_OFF_HINT,
      hasInput: true,
      hasOutput: true,
    },
  },
  {
    id: 'hand',
    type: 'studio',
    position: { x: 470, y: 490 },
    data: {
      title: 'Hand Extract',
      subtitle: 'Landmarks + pinch strength',
      kind: 'extract',
      icon: 'hand',
      accent: '#84cc16',
      previewSource: 'hand',
      metrics: [...DEFAULT_METRICS_BY_SOURCE.hand],
      previewEnabled: false,
      previewHint: PREVIEW_OFF_HINT,
      hasInput: true,
      hasOutput: true,
    },
  },
  {
    id: 'overlay',
    type: 'studio',
    position: { x: 880, y: 70 },
    data: {
      title: 'Landmark Overlay',
      subtitle: 'Composed face + hand mesh',
      kind: 'compose',
      icon: 'overlay',
      accent: '#65a30d',
      previewSource: 'overlay',
      metrics: [...DEFAULT_METRICS_BY_SOURCE.overlay],
      previewEnabled: false,
      previewHint: PREVIEW_OFF_HINT,
      hasInput: true,
      hasOutput: true,
    },
  },
  {
    id: 'mapper',
    type: 'studio',
    position: { x: 880, y: 490 },
    data: {
      title: 'Gesture Mapper',
      subtitle: 'Transforms movement to controls',
      kind: 'map',
      icon: 'mapper',
      accent: '#d9f99d',
      previewSource: 'mapper',
      metrics: [...DEFAULT_METRICS_BY_SOURCE.mapper],
      previewEnabled: false,
      previewHint: PREVIEW_OFF_HINT,
      hasInput: true,
      hasOutput: true,
    },
  },
  {
    id: 'stage',
    type: 'studio',
    position: { x: 1290, y: 280 },
    data: {
      title: 'Stage Output',
      subtitle: 'Realtime visual monitor feed',
      kind: 'output',
      icon: 'stage',
      accent: '#84cc16',
      previewSource: 'stage',
      metrics: [...DEFAULT_METRICS_BY_SOURCE.stage],
      previewEnabled: false,
      previewHint: PREVIEW_OFF_HINT,
      hasInput: true,
      hasOutput: false,
    },
  },
];

const INITIAL_EDGES: Edge[] = [
  {
    id: 'webcam-face',
    source: 'webcam',
    target: 'face',
    animated: true,
    style: { stroke: '#a3e635', strokeWidth: 2.5 },
  },
  {
    id: 'webcam-hand',
    source: 'webcam',
    target: 'hand',
    animated: true,
    style: { stroke: '#84cc16', strokeWidth: 2.5 },
  },
  {
    id: 'webcam-overlay',
    source: 'webcam',
    target: 'overlay',
    animated: true,
    style: { stroke: '#bef264', strokeWidth: 2.5 },
  },
  {
    id: 'face-overlay',
    source: 'face',
    target: 'overlay',
    animated: true,
    style: { stroke: '#a3e635', strokeWidth: 2.5 },
  },
  {
    id: 'hand-overlay',
    source: 'hand',
    target: 'overlay',
    animated: true,
    style: { stroke: '#84cc16', strokeWidth: 2.5 },
  },
  {
    id: 'face-mapper',
    source: 'face',
    target: 'mapper',
    animated: true,
    style: { stroke: '#d9f99d', strokeWidth: 2.5 },
  },
  {
    id: 'hand-mapper',
    source: 'hand',
    target: 'mapper',
    animated: true,
    style: { stroke: '#bef264', strokeWidth: 2.5 },
  },
  {
    id: 'overlay-stage',
    source: 'overlay',
    target: 'stage',
    animated: true,
    style: { stroke: '#84cc16', strokeWidth: 2.5 },
  },
  {
    id: 'mapper-stage',
    source: 'mapper',
    target: 'stage',
    animated: true,
    style: { stroke: '#d9f99d', strokeWidth: 2.5 },
  },
];

const MEDIAPIPE_RUNTIME_NOISE = [
  'Created TensorFlow Lite XNNPACK delegate for CPU.',
  'Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.',
  'Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.',
];

function isMediapipeRuntimeNoise(args: unknown[]): boolean {
  const text = args
    .map((arg) => {
      if (typeof arg === 'string') return arg;
      try {
        return JSON.stringify(arg);
      } catch {
        return String(arg);
      }
    })
    .join(' ');

  return MEDIAPIPE_RUNTIME_NOISE.some((pattern) => text.includes(pattern));
}

function createPreviewCanvas(): HTMLCanvasElement | null {
  if (typeof document === 'undefined') return null;
  const canvas = document.createElement('canvas');
  canvas.width = PREVIEW_WIDTH;
  canvas.height = PREVIEW_HEIGHT;
  return canvas;
}

function clamp(value: number, min = 0, max = 1): number {
  return Math.max(min, Math.min(max, value));
}

function clampNumber(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function computeStageRadius(controls: ControlsMetrics, maxDimension: number): number {
  const pinchInfluence = controls.pinch * 0.52;
  const jawInfluence = clamp(controls.jaw * 13, 0, 0.38);
  const base = maxDimension * (0.14 + pinchInfluence + jawInfluence);
  const withPresence = controls.presence > 0.05 ? base : base * 0.45;
  return clampNumber(withPresence, 24, maxDimension * 0.46);
}

function distance2d(a?: Landmark, b?: Landmark): number {
  if (!a || !b) return 0;
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function formatPercent(value: number): string {
  return `${Math.round(clamp(value, 0, 1) * 100)}%`;
}

function buildMetricsBySource(
  source: PreviewKey,
  runtimeFrame: number,
  faces: Landmark[][],
  hands: Landmark[][],
  controls: ControlsMetrics,
  fps: number,
  radius: number
): string[] {
  if (source === 'webcam') {
    return [`frame: ${runtimeFrame}`, `fps: ${fps.toFixed(1)}`];
  }
  if (source === 'face') {
    return [`faces: ${faces.length}`, `nose: ${controls.tilt.toFixed(2)}, ${(1 - controls.lift).toFixed(2)}`, `jaw: ${controls.jaw.toFixed(3)}`];
  }
  if (source === 'hand') {
    return [`hands: ${hands.length}`, `pinch: ${formatPercent(controls.pinch)}`, `lift: ${controls.lift.toFixed(2)}`];
  }
  if (source === 'overlay') {
    return [faces.length ? 'face mesh on' : 'face mesh off', hands.length ? 'hand mesh on' : 'hand mesh off'];
  }
  if (source === 'mapper') {
    return [`tilt: ${controls.tilt.toFixed(2)}`, `lift: ${controls.lift.toFixed(2)}`, `pinch: ${controls.pinch.toFixed(2)}`];
  }
  return [`presence: ${controls.presence.toFixed(2)}`, `radius: ${Math.round(radius)}`];
}

function drawCoverImage(
  ctx: CanvasRenderingContext2D,
  image: CanvasImageSource,
  targetWidth = ctx.canvas.width,
  targetHeight = ctx.canvas.height
): void {
  const sourceWidth =
    (image as HTMLCanvasElement).width ??
    (image as HTMLVideoElement).videoWidth ??
    (image as HTMLImageElement).naturalWidth ??
    0;
  const sourceHeight =
    (image as HTMLCanvasElement).height ??
    (image as HTMLVideoElement).videoHeight ??
    (image as HTMLImageElement).naturalHeight ??
    0;

  if (!sourceWidth || !sourceHeight || !targetWidth || !targetHeight) return;

  const sourceRatio = sourceWidth / sourceHeight;
  const targetRatio = targetWidth / targetHeight;

  let drawWidth: number;
  let drawHeight: number;
  let dx = 0;
  let dy = 0;

  if (sourceRatio > targetRatio) {
    drawHeight = targetHeight;
    drawWidth = drawHeight * sourceRatio;
    dx = (targetWidth - drawWidth) * 0.5;
  } else {
    drawWidth = targetWidth;
    drawHeight = drawWidth / sourceRatio;
    dy = (targetHeight - drawHeight) * 0.5;
  }

  ctx.drawImage(image, dx, dy, drawWidth, drawHeight);
}

function drawPoints(
  ctx: CanvasRenderingContext2D,
  landmarks: Landmark[],
  color: string,
  pointSize = 2,
  step = 5
): void {
  if (!landmarks.length) return;
  ctx.fillStyle = color;
  for (let i = 0; i < landmarks.length; i += step) {
    const point = landmarks[i];
    ctx.beginPath();
    ctx.arc(point.x * ctx.canvas.width, point.y * ctx.canvas.height, pointSize, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawHandConnections(ctx: CanvasRenderingContext2D, landmarks: Landmark[], color: string): void {
  if (!landmarks.length) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (const [from, to] of HAND_CONNECTIONS) {
    const a = landmarks[from];
    const b = landmarks[to];
    if (!a || !b) continue;
    ctx.moveTo(a.x * ctx.canvas.width, a.y * ctx.canvas.height);
    ctx.lineTo(b.x * ctx.canvas.width, b.y * ctx.canvas.height);
  }
  ctx.stroke();
}

function drawMapperScope(canvas: HTMLCanvasElement, controls: ControlsMetrics): void {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#020617';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const bars = [
    { label: 'tilt', value: clamp(Math.abs(controls.tilt), 0, 1), color: '#bef264' },
    { label: 'lift', value: clamp(controls.lift, 0, 1), color: '#a3e635' },
    { label: 'pinch', value: clamp(controls.pinch, 0, 1), color: '#84cc16' },
    { label: 'jaw', value: clamp(controls.jaw * 25, 0, 1), color: '#d9f99d' },
  ];

  bars.forEach((bar, index) => {
    const y = 26 + index * 36;
    ctx.fillStyle = 'rgba(148, 163, 184, 0.22)';
    ctx.fillRect(20, y, canvas.width - 40, 10);
    ctx.fillStyle = bar.color;
    ctx.fillRect(20, y, (canvas.width - 40) * bar.value, 10);

    ctx.fillStyle = '#cbd5e1';
    ctx.font = '11px var(--font-body)';
    ctx.fillText(`${bar.label}: ${bar.value.toFixed(2)}`, 20, y - 7);
  });
}

function canvasToPreviewUrl(canvas: HTMLCanvasElement): string {
  return canvas.toDataURL('image/jpeg', 0.68);
}

function StatusBadge({ status, label }: { status: 'idle' | 'loading' | 'live' | 'error'; label: string }) {
  const className =
    status === 'live'
      ? 'border-lime-300/40 bg-lime-500/15 text-lime-100'
      : status === 'loading'
        ? 'border-lime-200/35 bg-lime-300/10 text-lime-100'
        : status === 'error'
          ? 'border-rose-300/35 bg-rose-500/15 text-rose-100'
          : 'border-slate-300/30 bg-slate-500/15 text-slate-100';

  return <span className={`rounded-full border px-3 py-1 text-xs font-semibold ${className}`}>{label}</span>;
}

function MonitorStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-white/10 bg-slate-900/75 px-2.5 py-2">
      <p className="text-[9px] uppercase tracking-[0.16em] text-slate-400/90">{label}</p>
      <p className="mt-1 text-xs font-medium text-lime-100">{value}</p>
    </div>
  );
}

function ModernNodeCard({
  id,
  data,
  selected,
  onTogglePreview,
}: NodeProps<StudioNode> & { onTogglePreview: (nodeId: string) => void }) {
  const Icon = ICONS[data.icon];
  const isPreviewEnabled = data.previewEnabled;

  return (
    <div
      className="relative w-[330px] rounded-2xl border border-white/12 bg-slate-950/84 p-3 text-slate-100 shadow-[0_28px_95px_rgba(2,6,23,0.65)] backdrop-blur-xl"
      style={{ boxShadow: selected ? `0 0 0 2px ${data.accent}, 0 28px 95px rgba(2,6,23,0.65)` : undefined }}
    >
      {data.hasInput && (
        <Handle
          type="target"
          position={Position.Left}
          style={{ width: 12, height: 12, border: '2px solid #020617', background: data.accent }}
        />
      )}
      {data.hasOutput && (
        <Handle
          type="source"
          position={Position.Right}
          style={{ width: 12, height: 12, border: '2px solid #020617', background: data.accent }}
        />
      )}

      <div
        className="pointer-events-none absolute inset-0 rounded-2xl opacity-80"
        style={{ background: `radial-gradient(circle at 14% 12%, ${data.accent}30 0%, transparent 42%)` }}
      />

      <div className="relative flex items-start justify-between gap-3">
        <div>
          <p className="text-[10px] uppercase tracking-[0.2em] text-slate-300/70">{data.kind}</p>
          <h3 className="mt-1 text-base font-semibold text-white">{data.title}</h3>
          <p className="mt-1 text-xs text-slate-300/75">{data.subtitle}</p>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <button
            type="button"
            onClick={(event) => {
              event.stopPropagation();
              onTogglePreview(id);
            }}
            className={`grid h-9 w-9 place-items-center rounded-lg border text-slate-100 transition ${
              isPreviewEnabled
                ? 'border-lime-300/45 bg-lime-500/20 hover:bg-lime-400/25'
                : 'border-white/12 bg-slate-900/80 hover:bg-slate-800/90'
            }`}
            title={isPreviewEnabled ? 'Hide preview' : 'Show preview'}
            aria-label={isPreviewEnabled ? 'Hide preview' : 'Show preview'}
          >
            {isPreviewEnabled ? <Eye size={15} strokeWidth={2.2} /> : <EyeOff size={15} strokeWidth={2.2} />}
          </button>
          <div
            className="grid h-9 w-9 place-items-center rounded-lg border border-white/10 bg-slate-900/80"
            style={{ color: data.accent }}
          >
            <Icon size={17} strokeWidth={2.1} />
          </div>
        </div>
      </div>

      <div className="relative mt-3 grid gap-1.5">
        {data.metrics.map((metric) => (
          <div
            key={metric}
            className="rounded-md border border-white/8 bg-slate-900/75 px-2.5 py-1.5 text-[11px] text-slate-100/90"
          >
            {metric}
          </div>
        ))}
      </div>

      <div className="relative mt-3 overflow-hidden rounded-xl border border-white/12 bg-slate-900/80">
        {data.previewUrl ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={data.previewUrl} alt={`${data.title} preview`} className="aspect-video h-auto w-full object-cover" />
        ) : (
          <div className="grid aspect-video place-items-center px-3 text-center text-xs text-slate-400">{data.previewHint}</div>
        )}
      </div>
    </div>
  );
}

export function NodeStudio() {
  const [nodes, setNodes, onNodesChange] = useNodesState<StudioNode>(INITIAL_NODES);
  const [edges, setEdges, onEdgesChange] = useEdgesState(INITIAL_EDGES);
  const [status, setStatus] = useState<'idle' | 'loading' | 'live' | 'error'>('idle');
  const [statusText, setStatusText] = useState('Idle');
  const [debugText, setDebugText] = useState('No frames yet. Tap the Live icon to start camera access.');
  const [runtimeMonitor, setRuntimeMonitor] = useState<RuntimeMonitor>({
    fps: 0,
    frameMs: 0,
    loadPercent: 0,
    memoryMb: null,
    webgpuSupported: false,
    webgpuLabel: 'Detecting...',
  });
  const [pipSize, setPipSize] = useState({ width: 480, height: 330 });
  const [isResizingPip, setIsResizingPip] = useState(false);
  const [nodePicker, setNodePicker] = useState<NodePickerState>({
    open: false,
    panelX: 16,
    panelY: 16,
    flowX: 0,
    flowY: 0,
    query: '',
  });

  const nodesRef = useRef<StudioNode[]>(INITIAL_NODES);
  const nodeIdCounterRef = useRef(1);
  const lastPaneClickRef = useRef(0);
  const reactFlowInstanceRef = useRef<ReactFlowInstance<StudioNode, Edge> | null>(null);
  const pickerInputRef = useRef<HTMLInputElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const stageCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const sourceCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const tasksModuleRef = useRef<typeof import('@mediapipe/tasks-vision') | null>(null);
  const resizeStartRef = useRef<{ x: number; y: number; width: number; height: number } | null>(null);

  const runtimeRef = useRef<RuntimeState>({
    stream: null,
    rafId: null,
    faceLandmarker: null,
    handLandmarker: null,
    modelsReady: false,
    frameCount: 0,
    lastFrameTimestamp: 0,
    lastUiUpdate: 0,
    frameCostMs: 0,
    smoothedControls: { ...EMPTY_CONTROLS },
    previewCanvases: {
      webcam: createPreviewCanvas(),
      face: createPreviewCanvas(),
      hand: createPreviewCanvas(),
      overlay: createPreviewCanvas(),
      mapper: createPreviewCanvas(),
      stage: createPreviewCanvas(),
    },
  });

  const toggleNodePreview = useCallback(
    (nodeId: string) => {
      setNodes((prevNodes) => {
        const nextNodes = prevNodes.map((node) => {
          if (node.id !== nodeId) return node;
          const nextEnabled = !node.data.previewEnabled;
          return {
            ...node,
            data: {
              ...node.data,
              previewEnabled: nextEnabled,
              previewUrl: nextEnabled ? node.data.previewUrl : undefined,
              previewHint: nextEnabled ? PREVIEW_HINTS[node.data.previewSource] : PREVIEW_OFF_HINT,
            },
          };
        });
        nodesRef.current = nextNodes;
        return nextNodes;
      });
    },
    [setNodes]
  );

  const nodeTypes = useMemo<NodeTypes>(
    () => ({
      studio: (props) => <ModernNodeCard {...(props as NodeProps<StudioNode>)} onTogglePreview={toggleNodePreview} />,
    }),
    [toggleNodePreview]
  );

  const filteredLibrary = useMemo(() => {
    const query = nodePicker.query.trim().toLowerCase();
    if (!query) return NODE_LIBRARY;
    return NODE_LIBRARY.filter((item) =>
      `${item.title} ${item.subtitle} ${item.kind} ${item.searchTerms}`.toLowerCase().includes(query)
    );
  }, [nodePicker.query]);

  const closeNodePicker = useCallback(() => {
    setNodePicker((prev) => ({
      ...prev,
      open: false,
      query: '',
    }));
  }, []);

  const spawnNodeFromTemplate = useCallback(
    (template: NodeTemplate) => {
      const nextCounter = nodeIdCounterRef.current;
      nodeIdCounterRef.current += 1;
      const offset = (nextCounter % 6) * 10;
      const nodeId = `${template.templateId}-${nextCounter}`;

      setNodes((prevNodes) => {
        const nextNodes = [
          ...prevNodes,
          {
            id: nodeId,
            type: 'studio' as const,
            position: { x: nodePicker.flowX + offset, y: nodePicker.flowY + offset },
            data: {
              title: template.title,
              subtitle: template.subtitle,
              kind: template.kind,
              icon: template.icon,
              accent: template.accent,
              previewSource: template.previewSource,
              metrics: [...DEFAULT_METRICS_BY_SOURCE[template.previewSource]],
              previewEnabled: false,
              previewHint: PREVIEW_OFF_HINT,
              hasInput: template.hasInput,
              hasOutput: template.hasOutput,
            },
          },
        ];
        nodesRef.current = nextNodes;
        return nextNodes;
      });

      setDebugText(
        `Added "${template.title}" node at x:${Math.round(nodePicker.flowX)} y:${Math.round(nodePicker.flowY)} (double-click canvas to add more).`
      );
      closeNodePicker();
    },
    [closeNodePicker, nodePicker.flowX, nodePicker.flowY, setNodes]
  );

  const handlePaneClick = useCallback((event: React.MouseEvent) => {
    const now = performance.now();
    if (now - lastPaneClickRef.current > 280) {
      lastPaneClickRef.current = now;
      return;
    }
    lastPaneClickRef.current = 0;

    const instance = reactFlowInstanceRef.current;
    if (!instance) return;

    const flowPoint = instance.screenToFlowPosition({
      x: event.clientX,
      y: event.clientY,
    });
    const panelWidth = 360;
    const panelHeight = 450;
    const panelX = clampNumber(event.clientX + 14, 12, window.innerWidth - panelWidth - 12);
    const panelY = clampNumber(event.clientY + 14, 12, window.innerHeight - panelHeight - 12);

    setNodePicker({
      open: true,
      panelX,
      panelY,
      flowX: flowPoint.x,
      flowY: flowPoint.y,
      query: '',
    });
  }, []);

  const setStudioStatus = useCallback((nextStatus: 'idle' | 'loading' | 'live' | 'error', message: string) => {
    setStatus(nextStatus);
    setStatusText(message);
  }, []);

  const ensureModels = useCallback(async () => {
    const runtime = runtimeRef.current;
    if (runtime.modelsReady && runtime.faceLandmarker && runtime.handLandmarker) {
      return;
    }

    setStudioStatus('loading', 'Loading MediaPipe models...');

    if (!tasksModuleRef.current) {
      tasksModuleRef.current = await import('@mediapipe/tasks-vision');
    }

    const { FilesetResolver, FaceLandmarker, HandLandmarker } = tasksModuleRef.current;
    const vision = await FilesetResolver.forVisionTasks(WASM_URL);

    runtime.faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: FACE_MODEL_URL },
      runningMode: 'VIDEO',
      numFaces: 1,
      outputFaceBlendshapes: true,
    });

    runtime.handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: HAND_MODEL_URL },
      runningMode: 'VIDEO',
      numHands: 2,
    });

    runtime.modelsReady = true;
  }, [setStudioStatus]);

  const stopLive = useCallback(() => {
    const runtime = runtimeRef.current;

    if (runtime.rafId !== null) {
      cancelAnimationFrame(runtime.rafId);
      runtime.rafId = null;
    }

    if (runtime.stream) {
      for (const track of runtime.stream.getTracks()) {
        track.stop();
      }
      runtime.stream = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    runtime.frameCostMs = 0;
    setRuntimeMonitor((prev) => ({
      ...prev,
      fps: 0,
      frameMs: 0,
      loadPercent: 0,
    }));
    setStudioStatus('idle', 'Camera stopped');
  }, [setStudioStatus]);

  useEffect(() => {
    return () => {
      stopLive();
    };
  }, [stopLive]);

  useEffect(() => {
    nodesRef.current = nodes;
  }, [nodes]);

  useEffect(() => {
    if (!nodePicker.open) return;
    const rafId = requestAnimationFrame(() => {
      pickerInputRef.current?.focus();
    });
    return () => cancelAnimationFrame(rafId);
  }, [nodePicker.open]);

  useEffect(() => {
    if (!nodePicker.open) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        closeNodePicker();
      }
      if (event.key === 'Enter' && filteredLibrary[0]) {
        event.preventDefault();
        spawnNodeFromTemplate(filteredLibrary[0]);
      }
    };

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [closeNodePicker, filteredLibrary, nodePicker.open, spawnNodeFromTemplate]);

  useEffect(() => {
    let cancelled = false;

    const detectWebGpu = async () => {
      const nav = navigator as Navigator & { gpu?: WebGpuLike };
      if (!nav.gpu) {
        if (!cancelled) {
          setRuntimeMonitor((prev) => ({
            ...prev,
            webgpuSupported: false,
            webgpuLabel: 'Unavailable',
          }));
        }
        return;
      }

      try {
        const adapter = await nav.gpu.requestAdapter();
        if (!adapter) {
          if (!cancelled) {
            setRuntimeMonitor((prev) => ({
              ...prev,
              webgpuSupported: false,
              webgpuLabel: 'No adapter',
            }));
          }
          return;
        }

        let label = 'Adapter ready';
        if (adapter.requestAdapterInfo) {
          try {
            const info = await adapter.requestAdapterInfo();
            const parts = [info.vendor, info.architecture, info.device, info.description].filter((value) => Boolean(value));
            if (parts.length > 0) {
              label = parts.join(' · ');
            }
          } catch {
            // keep default label
          }
        }

        if (!cancelled) {
          setRuntimeMonitor((prev) => ({
            ...prev,
            webgpuSupported: true,
            webgpuLabel: label,
          }));
        }
      } catch {
        if (!cancelled) {
          setRuntimeMonitor((prev) => ({
            ...prev,
            webgpuSupported: false,
            webgpuLabel: 'Detection failed',
          }));
        }
      }
    };

    void detectWebGpu();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const originalWarn = console.warn.bind(console);
    const originalError = console.error.bind(console);
    const originalInfo = console.info.bind(console);

    console.warn = (...args: unknown[]) => {
      if (isMediapipeRuntimeNoise(args)) return;
      originalWarn(...args);
    };

    console.error = (...args: unknown[]) => {
      if (isMediapipeRuntimeNoise(args)) return;
      originalError(...args);
    };

    console.info = (...args: unknown[]) => {
      if (isMediapipeRuntimeNoise(args)) return;
      originalInfo(...args);
    };

    return () => {
      console.warn = originalWarn;
      console.error = originalError;
      console.info = originalInfo;
    };
  }, []);

  const drawPipStage = useCallback(
    (sourceFrame: HTMLCanvasElement, faces: Landmark[][], hands: Landmark[][], controls: ControlsMetrics, fps: number) => {
      const stage = stageCanvasRef.current;
      if (!stage) return;

      const dpr = window.devicePixelRatio || 1;
      const cssWidth = Math.max(220, Math.floor(stage.clientWidth));
      const cssHeight = Math.max(150, Math.floor(stage.clientHeight));
      const realWidth = Math.floor(cssWidth * dpr);
      const realHeight = Math.floor(cssHeight * dpr);

      if (stage.width !== realWidth || stage.height !== realHeight) {
        stage.width = realWidth;
        stage.height = realHeight;
      }

      const ctx = stage.getContext('2d');
      if (!ctx) return;

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, cssWidth, cssHeight);

      ctx.fillStyle = '#070d1d';
      ctx.fillRect(0, 0, cssWidth, cssHeight);

      ctx.save();
      ctx.globalAlpha = 0.92;
      drawCoverImage(ctx, sourceFrame, cssWidth, cssHeight);
      ctx.restore();

      if (faces[0]) {
        drawPoints(ctx, faces[0], '#bef264', 1.35, 5);
      }

      for (const hand of hands) {
        drawHandConnections(ctx, hand, '#84cc16');
        drawPoints(ctx, hand, '#d9f99d', 1.8, 1);
      }

      const centerX = cssWidth * clamp(0.5 + controls.tilt * 1.1, 0.08, 0.92);
      const centerY = cssHeight * clamp(0.73 - controls.lift * 0.83, 0.1, 0.9);
      const radius = computeStageRadius(controls, Math.min(cssWidth, cssHeight));

      ctx.save();
      ctx.globalCompositeOperation = 'screen';
      const gradient = ctx.createRadialGradient(centerX, centerY, 8, centerX, centerY, radius);
      gradient.addColorStop(0, 'hsla(84, 95%, 70%, 0.5)');
      gradient.addColorStop(0.55, 'hsla(88, 88%, 56%, 0.22)');
      gradient.addColorStop(1, 'hsla(84, 70%, 25%, 0)');
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();

      ctx.fillStyle = 'rgba(2, 6, 23, 0.65)';
      ctx.fillRect(10, 10, 130, 42);
      ctx.fillStyle = '#d9f99d';
      ctx.font = '11px var(--font-body)';
      ctx.fillText(`fps: ${fps.toFixed(1)}`, 20, 28);
      ctx.fillText(`presence: ${controls.presence.toFixed(2)}`, 20, 44);
    },
    []
  );

  const updateNodeSnapshots = useCallback(
    (
      frameCanvas: HTMLCanvasElement,
      faces: Landmark[][],
      hands: Landmark[][],
      controls: ControlsMetrics,
      fps: number,
      radius: number
    ) => {
      const runtime = runtimeRef.current;
      const previews = runtime.previewCanvases;
      const nodeSnapshot = nodesRef.current;
      const activeSources = new Set<PreviewKey>();
      for (const node of nodeSnapshot) {
        if (node.data.previewEnabled) {
          activeSources.add(node.data.previewSource);
        }
      }

      const previewUrls: Partial<Record<PreviewKey, string>> = {};

      const ensurePreviewContext = (
        key: PreviewKey
      ): { canvas: HTMLCanvasElement; ctx: CanvasRenderingContext2D } | null => {
        if (!activeSources.has(key)) return null;
        if (!previews[key]) {
          previews[key] = createPreviewCanvas();
        }
        const canvas = previews[key];
        if (!canvas) return null;
        const ctx = canvas.getContext('2d');
        if (!ctx) return null;
        return { canvas, ctx };
      };

      const webcamPreview = ensurePreviewContext('webcam');
      if (webcamPreview) {
        webcamPreview.ctx.clearRect(0, 0, PREVIEW_WIDTH, PREVIEW_HEIGHT);
        drawCoverImage(webcamPreview.ctx, frameCanvas, PREVIEW_WIDTH, PREVIEW_HEIGHT);
        previewUrls.webcam = canvasToPreviewUrl(webcamPreview.canvas);
      }

      const facePreview = ensurePreviewContext('face');
      if (facePreview) {
        facePreview.ctx.clearRect(0, 0, PREVIEW_WIDTH, PREVIEW_HEIGHT);
        drawCoverImage(facePreview.ctx, frameCanvas, PREVIEW_WIDTH, PREVIEW_HEIGHT);
        if (faces[0]) {
          drawPoints(facePreview.ctx, faces[0], '#bef264', 1.4, 5);
        }
        previewUrls.face = canvasToPreviewUrl(facePreview.canvas);
      }

      const handPreview = ensurePreviewContext('hand');
      if (handPreview) {
        handPreview.ctx.clearRect(0, 0, PREVIEW_WIDTH, PREVIEW_HEIGHT);
        drawCoverImage(handPreview.ctx, frameCanvas, PREVIEW_WIDTH, PREVIEW_HEIGHT);
        for (const hand of hands) {
          drawHandConnections(handPreview.ctx, hand, '#84cc16');
          drawPoints(handPreview.ctx, hand, '#d9f99d', 1.9, 1);
        }
        previewUrls.hand = canvasToPreviewUrl(handPreview.canvas);
      }

      const overlayPreview = ensurePreviewContext('overlay');
      if (overlayPreview) {
        overlayPreview.ctx.clearRect(0, 0, PREVIEW_WIDTH, PREVIEW_HEIGHT);
        drawCoverImage(overlayPreview.ctx, frameCanvas, PREVIEW_WIDTH, PREVIEW_HEIGHT);
        if (faces[0]) {
          drawPoints(overlayPreview.ctx, faces[0], '#bef264', 1.4, 5);
        }
        for (const hand of hands) {
          drawHandConnections(overlayPreview.ctx, hand, '#84cc16');
          drawPoints(overlayPreview.ctx, hand, '#d9f99d', 1.9, 1);
        }
        previewUrls.overlay = canvasToPreviewUrl(overlayPreview.canvas);
      }

      const mapperPreview = ensurePreviewContext('mapper');
      if (mapperPreview) {
        drawMapperScope(mapperPreview.canvas, controls);
        previewUrls.mapper = canvasToPreviewUrl(mapperPreview.canvas);
      }

      const stagePreview = ensurePreviewContext('stage');
      const stageCanvas = stageCanvasRef.current;
      if (stagePreview && stageCanvas) {
        stagePreview.ctx.clearRect(0, 0, PREVIEW_WIDTH, PREVIEW_HEIGHT);
        drawCoverImage(stagePreview.ctx, stageCanvas, PREVIEW_WIDTH, PREVIEW_HEIGHT);
        previewUrls.stage = canvasToPreviewUrl(stagePreview.canvas);
      }

      setNodes((prevNodes) =>
        prevNodes.map((node) => {
          const source = node.data.previewSource;
          const isPreviewEnabled = node.data.previewEnabled;
          const nextDataBase = {
            ...node.data,
            previewEnabled: isPreviewEnabled,
            previewUrl: isPreviewEnabled ? (previewUrls[source] ?? node.data.previewUrl) : undefined,
            previewHint: isPreviewEnabled ? PREVIEW_HINTS[source] : PREVIEW_OFF_HINT,
            metrics: buildMetricsBySource(source, runtime.frameCount, faces, hands, controls, fps, radius),
          };

          return {
            ...node,
            data: nextDataBase,
          };
        })
      );
    },
    [setNodes]
  );

  const startLive = useCallback(async () => {
    const runtime = runtimeRef.current;

    try {
      await ensureModels();

      if (!videoRef.current) {
        throw new Error('Missing hidden video element.');
      }

      if (runtime.rafId !== null) {
        return;
      }

      if (!runtime.stream) {
        runtime.stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: 'user',
          },
          audio: false,
        });
      }

      videoRef.current.srcObject = runtime.stream;
      await videoRef.current.play();

      setStudioStatus('live', 'Live capture running');

      const runFrame = (timestamp: number) => {
        const frameStart = performance.now();

        if (!videoRef.current || !runtime.stream || !runtime.faceLandmarker || !runtime.handLandmarker) {
          runtime.rafId = requestAnimationFrame(runFrame);
          return;
        }

        if (!sourceCanvasRef.current) {
          sourceCanvasRef.current = document.createElement('canvas');
        }

        const frameCanvas = sourceCanvasRef.current;
        const videoWidth = videoRef.current.videoWidth || 1280;
        const videoHeight = videoRef.current.videoHeight || 720;

        if (frameCanvas.width !== videoWidth || frameCanvas.height !== videoHeight) {
          frameCanvas.width = videoWidth;
          frameCanvas.height = videoHeight;
        }

        const frameCtx = frameCanvas.getContext('2d');
        if (!frameCtx) {
          runtime.rafId = requestAnimationFrame(runFrame);
          return;
        }

        frameCtx.clearRect(0, 0, videoWidth, videoHeight);
        frameCtx.save();
        frameCtx.translate(videoWidth, 0);
        frameCtx.scale(-1, 1);
        frameCtx.drawImage(videoRef.current, 0, 0, videoWidth, videoHeight);
        frameCtx.restore();

        const faceResult = runtime.faceLandmarker.detectForVideo(frameCanvas, timestamp);
        const handResult = runtime.handLandmarker.detectForVideo(frameCanvas, timestamp);

        const faces = (faceResult.faceLandmarks ?? []) as Landmark[][];
        const hands = (handResult.landmarks ?? []) as Landmark[][];

        const facePrimary = faces[0];
        const handPrimary = hands[0];

        const nose = facePrimary?.[1] ?? facePrimary?.[4] ?? facePrimary?.[0];
        const upperLip = facePrimary?.[13];
        const lowerLip = facePrimary?.[14];
        const wrist = handPrimary?.[0];
        const thumbTip = handPrimary?.[4];
        const indexTip = handPrimary?.[8];

        const faceMetrics: FaceMetrics = {
          count: faces.length,
          noseX: nose?.x ?? 0.5,
          noseY: nose?.y ?? 0.5,
          jaw: clamp(distance2d(upperLip, lowerLip), 0, 0.085),
        };

        const pinchDistance = distance2d(thumbTip, indexTip);
        const pinchNormalized = clamp((0.125 - pinchDistance) / 0.085, 0, 1);
        const handMetrics: HandMetrics = {
          count: hands.length,
          pinch: pinchNormalized,
          palmY: wrist?.y ?? 0.5,
        };

        const rawControls: ControlsMetrics = {
          tilt: faceMetrics.noseX - 0.5,
          lift: 1 - handMetrics.palmY,
          pinch: handMetrics.pinch,
          jaw: faceMetrics.jaw,
          presence: Math.max(faceMetrics.count ? 1 : 0, handMetrics.count ? 1 : 0),
        };

        const previousControls = runtime.smoothedControls;
        const smoothingWhenTracking = 0.22;
        const smoothingWhenMissingHand = 0.42;
        const controls: ControlsMetrics = {
          tilt: previousControls.tilt + (rawControls.tilt - previousControls.tilt) * smoothingWhenTracking,
          lift: previousControls.lift + (rawControls.lift - previousControls.lift) * smoothingWhenTracking,
          pinch:
            previousControls.pinch +
            ((handMetrics.count > 0 ? rawControls.pinch : 0) - previousControls.pinch) *
              (handMetrics.count > 0 ? 0.26 : smoothingWhenMissingHand),
          jaw: previousControls.jaw + (rawControls.jaw - previousControls.jaw) * 0.2,
          presence: previousControls.presence + (rawControls.presence - previousControls.presence) * 0.3,
        };
        runtime.smoothedControls = controls;

        const delta = runtime.lastFrameTimestamp > 0 ? timestamp - runtime.lastFrameTimestamp : 0;
        runtime.lastFrameTimestamp = timestamp;
        runtime.frameCount += 1;
        const fps = delta > 0 ? 1000 / delta : 0;

        drawPipStage(frameCanvas, faces, hands, controls, fps);

        if (timestamp - runtime.lastUiUpdate > 180) {
          runtime.lastUiUpdate = timestamp;
          const stageCanvas = stageCanvasRef.current;
          const maxDim = stageCanvas ? Math.min(stageCanvas.clientWidth || 0, stageCanvas.clientHeight || 0) : 320;
          const radius = computeStageRadius(controls, Math.max(240, maxDim));
          updateNodeSnapshots(frameCanvas, faces, hands, controls, fps, radius);

          const frameCostRaw = performance.now() - frameStart;
          runtime.frameCostMs =
            runtime.frameCostMs > 0 ? runtime.frameCostMs + (frameCostRaw - runtime.frameCostMs) * 0.22 : frameCostRaw;
          const memory = (performance as Performance & { memory?: { usedJSHeapSize?: number } }).memory;
          const memoryMb = memory?.usedJSHeapSize ? memory.usedJSHeapSize / (1024 * 1024) : null;
          const loadPercent = clampNumber((runtime.frameCostMs / 16.7) * 100, 0, 260);

          setRuntimeMonitor((prev) => ({
            ...prev,
            fps: Number.isFinite(fps) ? Number(fps.toFixed(1)) : 0,
            frameMs: Number(runtime.frameCostMs.toFixed(2)),
            loadPercent: Number(loadPercent.toFixed(1)),
            memoryMb: memoryMb !== null ? Number(memoryMb.toFixed(1)) : null,
          }));

          const activePreviewNodes = nodesRef.current
            .filter((node) => node.data.previewEnabled)
            .map((node) => `${node.data.title} (${node.id})`);

          setDebugText(
            JSON.stringify(
              {
                frame: runtime.frameCount,
                fps: Number.isFinite(fps) ? Number(fps.toFixed(2)) : 0,
                activePreviewNodes,
                face: {
                  count: faces.length,
                  jaw: Number(controls.jaw.toFixed(4)),
                },
                hand: {
                  count: hands.length,
                  pinch: Number(controls.pinch.toFixed(4)),
                  lift: Number(controls.lift.toFixed(4)),
                  pinchDistance: Number(pinchDistance.toFixed(4)),
                },
                controls,
              },
              null,
              2
            )
          );
        }

        runtime.rafId = requestAnimationFrame(runFrame);
      };

      runtime.rafId = requestAnimationFrame(runFrame);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setStudioStatus('error', 'Failed to start');
      setDebugText(message);
      console.error(error);
    }
  }, [drawPipStage, ensureModels, setStudioStatus, updateNodeSnapshots]);

  const toggleLive = useCallback(() => {
    if (status === 'live' || status === 'loading') {
      stopLive();
      return;
    }
    void startLive();
  }, [startLive, status, stopLive]);

  const resetLayout = useCallback(() => {
    stopLive();
    nodesRef.current = [];
    nodeIdCounterRef.current = 1;
    setNodes([]);
    setEdges([]);
    setStudioStatus('idle', 'Graph cleared');
    setDebugText('Graph cleared. Double-click empty canvas to spawn new nodes.');
  }, [setEdges, setNodes, setStudioStatus, stopLive]);

  const startResizePip = useCallback(
    (event: React.PointerEvent<HTMLButtonElement>) => {
      event.preventDefault();
      resizeStartRef.current = {
        x: event.clientX,
        y: event.clientY,
        width: pipSize.width,
        height: pipSize.height,
      };
      setIsResizingPip(true);
    },
    [pipSize.height, pipSize.width]
  );

  useEffect(() => {
    if (!isResizingPip) return;

    const onMove = (event: PointerEvent) => {
      const start = resizeStartRef.current;
      if (!start) return;

      const nextWidth = clampNumber(start.width + (event.clientX - start.x), 360, 980);
      const nextHeight = clampNumber(start.height + (event.clientY - start.y), 240, 760);

      setPipSize({ width: nextWidth, height: nextHeight });
    };

    const onEnd = () => {
      setIsResizingPip(false);
      resizeStartRef.current = null;
    };

    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onEnd);

    return () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onEnd);
    };
  }, [isResizingPip]);

  return (
    <div className="fixed inset-0 overflow-hidden bg-[radial-gradient(circle_at_14%_-12%,rgba(255,255,255,0.05),transparent_36%),radial-gradient(circle_at_86%_112%,rgba(132,204,22,0.08),transparent_38%),linear-gradient(180deg,#010204_0%,#05080f_52%,#03050b_100%)]">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onPaneClick={handlePaneClick}
        onInit={(instance) => {
          reactFlowInstanceRef.current = instance;
        }}
        nodeTypes={nodeTypes}
        defaultViewport={{ x: -100, y: -40, zoom: 0.78 }}
        minZoom={0.35}
        maxZoom={1.3}
        fitView={false}
        connectOnClick={false}
        panOnScroll
        selectionOnDrag
        proOptions={{ hideAttribution: true }}
      >
        <Background variant={BackgroundVariant.Dots} gap={26} size={1.15} color="rgba(98, 104, 119, 0.45)" />
        <Controls
          className="!rounded-xl !border !border-white/20 !bg-slate-900/75 !text-slate-100"
          showInteractive={false}
        />
        <MiniMap
          className="!rounded-xl !border !border-white/20 !bg-slate-900/75"
          nodeColor={(node) => (node.data as StudioNodeData).accent}
          maskColor="rgba(2, 6, 23, 0.48)"
        />
      </ReactFlow>

      <div className="pointer-events-none fixed left-4 top-4 z-30 flex items-center gap-2">
        <button
          type="button"
          onClick={toggleLive}
          className={`pointer-events-auto relative grid h-10 w-10 place-items-center rounded-xl border transition ${
            status === 'live'
              ? 'border-lime-300/45 bg-lime-500/20 text-lime-100 hover:bg-lime-400/25'
              : status === 'loading'
                ? 'border-amber-300/45 bg-amber-500/18 text-amber-100 hover:bg-amber-400/22'
                : status === 'error'
                  ? 'border-rose-300/45 bg-rose-500/20 text-rose-100 hover:bg-rose-400/24'
                  : 'border-white/15 bg-slate-950/85 text-slate-100 hover:bg-slate-900/95'
          }`}
          title={status === 'live' || status === 'loading' ? 'Stop live capture' : 'Start live capture'}
          aria-label={status === 'live' || status === 'loading' ? 'Stop live capture' : 'Start live capture'}
        >
          <Camera size={14} />
          <span
            className={`absolute right-[6px] top-[6px] h-2.5 w-2.5 rounded-full ${
              status === 'live' ? 'animate-pulse bg-lime-300' : status === 'loading' ? 'bg-amber-300' : 'bg-slate-400'
            }`}
          />
        </button>
        <button
          type="button"
          onClick={resetLayout}
          className="pointer-events-auto grid h-9 w-9 place-items-center rounded-xl border border-white/15 bg-slate-950/85 text-slate-200 transition hover:bg-slate-900/95"
          title="Clear all nodes"
          aria-label="Clear all nodes"
        >
          <Trash2 size={14} />
        </button>
      </div>

      <div className="pointer-events-none absolute left-4 top-[58px] z-20 rounded-xl border border-lime-300/20 bg-slate-950/75 px-3 py-2 text-[11px] text-lime-100/90 backdrop-blur-xl">
        Double-click empty canvas to summon Kineforge node picker
      </div>

      {nodePicker.open && (
        <div className="fixed inset-0 z-40" onMouseDown={closeNodePicker}>
          <div className="absolute inset-0 bg-black/35 backdrop-blur-[1px]" />
          <div
            className="absolute w-[360px] rounded-2xl border border-white/12 bg-slate-950/92 p-3 shadow-[0_36px_140px_rgba(0,0,0,0.7)] backdrop-blur-2xl"
            style={{ left: nodePicker.panelX, top: nodePicker.panelY }}
            onMouseDown={(event) => event.stopPropagation()}
          >
            <p className="text-[10px] uppercase tracking-[0.24em] text-lime-300/85">Kineforge Node Picker</p>
            <p className="mt-1 text-sm font-medium text-white">Add Node At Cursor</p>
            <p className="mt-1 text-xs text-slate-300/75">Type to search, Enter for first result, ESC to close.</p>

            <input
              ref={pickerInputRef}
              value={nodePicker.query}
              onChange={(event) =>
                setNodePicker((prev) => ({
                  ...prev,
                  query: event.target.value,
                }))
              }
              placeholder="Search nodes..."
              className="mt-3 w-full rounded-xl border border-white/14 bg-slate-900/85 px-3 py-2 text-sm text-slate-100 outline-none ring-lime-300/40 placeholder:text-slate-400 focus:border-lime-300/45 focus:ring-2"
            />

            <div className="mt-3 max-h-[292px] overflow-auto space-y-2 pr-1">
              {filteredLibrary.length > 0 ? (
                filteredLibrary.map((item) => (
                  <button
                    key={item.templateId}
                    type="button"
                    onClick={() => spawnNodeFromTemplate(item)}
                    className="w-full rounded-xl border border-white/10 bg-slate-900/70 px-3 py-2 text-left transition hover:border-lime-300/45 hover:bg-slate-800/90"
                  >
                    <p className="text-sm font-medium text-slate-100">{item.title}</p>
                    <p className="mt-0.5 text-xs text-slate-300/80">{item.subtitle}</p>
                    <p className="mt-1 text-[10px] uppercase tracking-[0.18em] text-lime-200/80">{item.kind}</p>
                  </button>
                ))
              ) : (
                <div className="rounded-xl border border-white/10 bg-slate-900/70 px-3 py-4 text-center text-xs text-slate-400">
                  검색 결과가 없습니다.
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      <aside
        className="fixed bottom-5 right-5 z-30"
        style={{ width: pipSize.width, height: pipSize.height, touchAction: 'none' }}
      >
        <div className="relative flex h-full flex-col overflow-hidden rounded-2xl border border-white/12 bg-slate-950/86 shadow-[0_28px_120px_rgba(2,6,23,0.74)] backdrop-blur-2xl">
          <div className="flex items-center justify-between border-b border-white/10 px-3 py-2.5">
            <div>
              <p className="text-[10px] uppercase tracking-[0.18em] text-lime-200/85">PiP Monitor</p>
              <p className="text-xs text-slate-300/80">우하단 실시간 결과 모니터</p>
            </div>
            <div className="flex items-center gap-2">
              <StatusBadge status={status} label={statusText} />
            </div>
          </div>

          <div className="border-b border-white/10 bg-slate-950/88 p-2.5">
            <div className="grid grid-cols-2 gap-2">
              <MonitorStat label="FPS" value={runtimeMonitor.fps.toFixed(1)} />
              <MonitorStat label="Frame ms" value={runtimeMonitor.frameMs.toFixed(2)} />
              <MonitorStat label="CPU Proxy" value={`${runtimeMonitor.loadPercent.toFixed(0)}%`} />
              <MonitorStat label="Heap" value={runtimeMonitor.memoryMb !== null ? `${runtimeMonitor.memoryMb.toFixed(1)} MB` : 'n/a'} />
            </div>
            <p
              className={`mt-2 text-[10px] ${
                runtimeMonitor.webgpuSupported ? 'text-lime-200/90' : 'text-rose-200/90'
              }`}
            >
              WebGPU: {runtimeMonitor.webgpuSupported ? runtimeMonitor.webgpuLabel : 'Unavailable'}
            </p>
          </div>

          <canvas ref={stageCanvasRef} className="min-h-0 flex-1 w-full border-b border-white/10 bg-slate-950" />

          <pre className="h-[30%] overflow-auto bg-slate-950/95 p-3 text-[11px] text-lime-100/95">{debugText}</pre>
        </div>

        <button
          type="button"
          onPointerDown={startResizePip}
          className="absolute bottom-1 right-1 grid h-7 w-7 place-items-center rounded-md border border-white/15 bg-slate-900/85 text-slate-200/90 transition hover:bg-slate-800/95"
          title="Resize monitor"
          aria-label="Resize monitor"
        >
          <Grip size={13} />
        </button>
      </aside>

      <video ref={videoRef} className="hidden" autoPlay playsInline muted />
    </div>
  );
}
