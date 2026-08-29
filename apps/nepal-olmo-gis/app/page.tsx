'use client';

import { AttributionControl, LngLatBounds, Map as MapLibreMap, NavigationControl, Popup, setWorkerUrl } from 'maplibre-gl';
import type { MapLayerMouseEvent } from 'maplibre-gl';
import type { Feature, FeatureCollection } from 'geojson';
import Image from 'next/image';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { CSSProperties } from 'react';
import 'maplibre-gl/dist/maplibre-gl.css';

type SceneRecord = {
  id: string;
  sensor: string;
  acquired_at: string;
  state: string;
  image: string;
  coordinates: [[number, number], [number, number], [number, number], [number, number]];
  source_sha256: string;
};

type ScenarioPoint = {
  id: string;
  display_label: string;
  map_label: string;
  stage: number;
  marker_color: string;
  in_event_chain: boolean;
  name: string;
  coordinates: [number, number];
  role: string;
  place: string;
  source?: string;
  source_url?: string;
  evidence_level?: string;
  story?: string;
  story_ko?: string;
  distance_from_a_km: number;
};

type ScheduledScene = {
  id?: string;
  sensor: string;
  acquired_at: string;
  state: string;
  detail?: string | null;
  evidence_uri?: string | null;
};

type IncidentUpdate = {
  occurred_at_utc: string;
  status: string;
  relation: string;
  title: string;
  summary: string;
  source: string;
  source_url: string;
};

type TransferRow = { region: string; auroc: number; placebo_auroc: number; patches: number };
type SusceptibilityRow = { region: string; olmo_auroc: number; raw_auroc: number; verdict: string };
type ResearchBlock = {
  integration_disclaimer: string;
  nepal_embedding: { status: string; baseline: string; placebo_count: number; claim: string };
  confirmatory_transfer: {
    status: string; regions: number; wins_reuse_vs_raw_strong: number; strong_wins: number;
    reuse_region_macro: number; raw_strong_region_macro: number; absolute_gap: number;
    relative_gain_pct: number; non_win_regions: string[]; claim_boundary: string | string[];
  };
  historical_event_delta_pilot: { rows: TransferRow[]; contract: string; claim_boundary: string };
  pre_event_susceptibility_probe: { rows: SusceptibilityRow[]; overall: string; claim_boundary: string };
  physics: { current: string; proposed_primary: string; independent_check: string; downstream_hydraulics: string; coupling_rule: string };
  evaluation_arms: { id: string; label: string }[];
  evaluation_metrics: Record<string, string>;
};

type LiveObservation = {
  sensor: string;
  acquired_at: string;
  catalog_status: string;
  product_name: string | null;
  publication_utc: string | null;
  cloud_cover_tile_pct: number | null;
  materialization_status: string;
  selection_preflight_valid: boolean;
  materialization_seal_valid: boolean;
  period_readiness: { sentinel1?: number; sentinel2_l2a?: number };
  olmo_ready: boolean;
  claim_boundary: string;
};

type CurrentDecision = {
  status: 'candidate_ready' | 'embed_ready' | 'hold' | 'wait_observation';
  action: string;
  reason: string;
  next_gate: string;
  allowed_claim: string;
};

type Scenario = {
  generated_at: string;
  event: { name: string; occurred_at: string; cause_status: string; evidence_status: string };
  points: ScenarioPoint[];
  scene_records: SceneRecord[];
  scheduled_scenes: ScheduledScene[];
  live_observation: LiveObservation | null;
  olmoearth: { input_contract: string; anchors: number; embedding_status: string; post_event_delta: string | Record<string, unknown> };
  decision: CurrentDecision;
  ops_log?: { event_id?: string; time_utc: string; source: string; type: string; priority: 'green' | 'orange' | 'blue'; summary: string }[];
  incident_updates: IncidentUpdate[];
  research: ResearchBlock;
  downstream_visual: {
    purpose: string;
    records: { label: string; acquired_at: string; item_id: string; mgrs_tile: string; tile_cloud_pct: number; image: string; image_sha256: string }[];
  };
  simulation: { route_points: number; claim: string; scientific_upgrade?: string };
};

type Hydrography = {
  type: 'FeatureCollection';
  features: Feature[];
  simulation_route: [number, number][];
};

type FlowExports = WebAssembly.Exports & {
  memory: WebAssembly.Memory;
  clear_route: () => void;
  set_route_point: (index: number, lon: number, lat: number) => void;
  reset: (seed: number) => void;
  step: (dt: number, speed: number) => void;
  particles_ptr: () => number;
  particle_count: () => number;
  abi_version: () => number;
};

// 타임라인 항목은 scenario.json에서 파생한다. 이전 버전은 이 목록을 하드코딩해서
// 실제 장면 8개 중 6개만 보였고 07-23의 센서를 잘못 표기했다.
type TimelineItem = {
  id: string;
  kind: 'scene' | 'event' | 'scheduled';
  date: string;      // "03 JUL"
  iso: string;       // 정렬용
  sensor: string;    // "S2" | "S1" | "EVENT"
  state: string;     // READY | IMPACT | PENDING | PLANNED
  selectable: boolean;
};

const MONTHS = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'];
const shortDate = (iso: string) => {
  const d = new Date(iso);
  return `${String(d.getUTCDate()).padStart(2, '0')} ${MONTHS[d.getUTCMonth()]}`;
};
const shortSensor = (sensor: string) => (sensor.includes('-2') || sensor.startsWith('S2') ? 'S2' : sensor.includes('-1') || sensor.startsWith('S1') ? 'S1' : sensor.toUpperCase());
const kstStamp = (iso: string | null) => iso
  ? new Intl.DateTimeFormat('en-GB', { timeZone: 'Asia/Seoul', month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false }).format(new Date(iso)).toUpperCase()
  : '—';

// 배경 지도 — 2026-08-28.
//
// 왜 OSM 직결을 못 쓰는가: tile.openstreetmap.org 는 앱 직접 사용을 금지함. 실측하면 모든
// 줌의 타일이 동일한 6,933 B로 오고 헤더에 `x-blocked: Access denied` / `x-totp: INVALID`가
// 붙음. **http 200으로 오는 것이 함정**이라 로그에 오류가 남지 않고 지도만 검게 남았음.
//
// 왜 벡터 대신 raster 인가: 이전 시도는 CARTO Dark Matter 벡터(93레이어) + 클라이언트
// 음영기복(raster-dem 디코딩)이었고 화면이 심하게 버벅였음. raster는 사전 렌더라 레이어가
// 2장이면 끝이고 GPU 부담이 훨씬 작음.
//
// 네팔 랑탕 z12 실측 (전부 200, 해시 상이 — 차단 함정 없음):
//   CARTO dark_all        5.2 KB   가장 가벼움. 앱 톤(#10241e)과 맞음
//   Esri World_Hillshade 24.3 KB   사전 렌더 음영기복 — 산악 입체감
//   Esri World_Imagery   11.7 KB   위성영상. EO 연구 맥락에 주제적으로 맞음
//   MapTiler outdoor-v2    403     키의 허용 도메인 설정이 맞아야 열림
//
// 배포 화면의 성립을 API key에 맡기지 않는다. 외부 raster는 지명/음영 context를 더할 뿐이고,
// 실제 선택 S2 장면은 별도 DOM backdrop으로 항상 렌더한다.

const lightRasterStyle = {
  version: 8 as const,
  sources: {
    hillshade: {
      type: 'raster' as const,
      tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/Elevation/World_Hillshade/MapServer/tile/{z}/{y}/{x}'],
      tileSize: 256,
      maxzoom: 16,
      attribution: 'Hillshade © Esri',
    },
    // CARTO 무료 raster는 2026-08 현재 "API KEY REQUIRED" 워터마크 타일을 반환함(실측).
    // 키 없이 쓸 수 있는 Esri 참조 라벨로 교체함.
    labels: {
      type: 'raster' as const,
      tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}'],
      tileSize: 256,
      maxzoom: 19,
      attribution: 'Labels © Esri',
    },
    // 3D 지형 — AWS 공개 Terrarium DEM (키 불필요). 이전에 버벅였던 것은 93레이어 벡터
    // 스타일 + 클라이언트 음영기복 디코딩 조합이었음. raster 2장 + terrain 은 부담이 다름.
    terrainDem: {
      type: 'raster-dem' as const,
      tiles: ['https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png'],
      tileSize: 256,
      encoding: 'terrarium' as const,
      maxzoom: 14,
      attribution: 'DEM: Mapzen/AWS Terrain Tiles',
    },
  },
  layers: [
    { id: 'map-background', type: 'background' as const, paint: { 'background-color': 'rgba(16, 36, 30, 0.18)' } },
    { id: 'hillshade', type: 'raster' as const, source: 'hillshade',
      paint: { 'raster-opacity': 0.85, 'raster-saturation': -0.6 } },
    { id: 'labels', type: 'raster' as const, source: 'labels',
      paint: { 'raster-opacity': 0.55 } },
  ],
};

// MapTiler 벡터 — 2026-08-28 사용자가 origin 제한을 고쳐 키가 열림 (localhost origin → 200 실측).
// 키가 없거나 스타일 로드가 실패하면 Esri raster 폴백으로 자동 강등함 (성립을 키에 맡기지 않음).
const MAPTILER_KEY = process.env.NEXT_PUBLIC_MAPTILER_KEY;
const maptilerStyleUrl = MAPTILER_KEY
  ? `https://api.maptiler.com/maps/outdoor-v2/style.json?key=${MAPTILER_KEY}`
  : null;
const basemapStyle = lightRasterStyle;
// 3D 지형 시점 — S2 장면(image source)은 terrain 위에 드레이프되므로 pitch 와 정합함.
const TERRAIN_PITCH = 52;
// MapTiler 스타일에는 우리 DEM 소스가 없으므로 3D 전환 시 동적으로 주입함.
const TERRAIN_DEM_SPEC = {
  type: 'raster-dem' as const,
  tiles: ['https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png'],
  tileSize: 256,
  encoding: 'terrarium' as const,
  maxzoom: 14,
  attribution: 'DEM: Mapzen/AWS Terrain Tiles',
};

setWorkerUrl('/maplibre-gl-worker.mjs');

const prefersReducedMotion = () =>
  typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

export default function Home() {
  const mapNode = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const flowSpeedRef = useRef(0.034);
  const flowPlayingRef = useRef(true);
  const wasmRef = useRef<FlowExports | null>(null);
  // 첫 화면은 사건 전체다. style reload가 전체 회랑을 다시 A/B 2.56 km
  // 장면으로 덮어쓰지 못하게, 사용자의 scene-focus 의도를 별도로 기억한다.
  const userSelectedSceneRef = useRef(false);
  const railsRef = useRef({ left: true, right: true });

  const [mapReady, setMapReady] = useState(false);
  const [styleRevision, setStyleRevision] = useState(0);
  // WebGL2가 없는 브라우저에서 MapLibre 생성자가 던지는 예외가 앱 전체를 죽이던
  // 결함의 방어. 'unsupported'면 지도 대신 정적 장면 이미지로 강등 표시한다.
  const [mapStatus, setMapStatus] = useState<'init' | 'ready' | 'unsupported'>('init');
  const [scenario, setScenario] = useState<Scenario | null>(null);
  const [hydrography, setHydrography] = useState<Hydrography | null>(null);
  // 데이터 로드와 WASM은 별개 채널이다. 이전 버전은 scenario fetch 실패를
  // wasmStatus='failed'로 표시해 "시뮬레이션이 죽었다"는 오보를 냈다.
  const [dataStatus, setDataStatus] = useState<'loading' | 'ready' | 'failed'>('loading');
  const [wasmStatus, setWasmStatus] = useState<'loading' | 'ready' | 'failed'>('loading');
  const [activeSceneId, setActiveSceneId] = useState<string | null>(null);
  // 2D(수직 정사영 — 판독·비교용) / 3D(지형 드레이프 — 회랑 실감용) 전환.
  const [viewDim, setViewDim] = useState<'2d' | '3d'>('2d');
  const [storyOpen, setStoryOpen] = useState(() => typeof window !== 'undefined' && window.location.hash === '#story');
  const [storyLang, setStoryLang] = useState<'en' | 'ko'>('en');
  const [swipe, setSwipe] = useState(52);
  const viewDimRef = useRef<'2d' | '3d'>('2d');
  const [selectedPoint, setSelectedPoint] = useState('E');
  const [overlayOpacity, setOverlayOpacity] = useState(0.78);
  const [showAnchors, setShowAnchors] = useState(true);
  const [flowPlaying, setFlowPlaying] = useState(true);
  const [flowSpeed, setFlowSpeed] = useState(0.034);
  const [leftOpen, setLeftOpen] = useState(true);
  const [rightOpen, setRightOpen] = useState(true);
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => { railsRef.current = { left: leftOpen, right: rightOpen }; }, [leftOpen, rightOpen]);

  // 좁은 화면에서는 패널을 기본으로 접는다 (지도가 주인공).
  // rAF로 페인트 뒤에 미룬다 — effect 내 동기 setState는 연쇄 렌더를 유발한다(lint).
  useEffect(() => {
    const id = requestAnimationFrame(() => {
      if (window.innerWidth < 1100) { setLeftOpen(false); setRightOpen(false); }
    });
    return () => cancelAnimationFrame(id);
  }, []);

  useEffect(() => {
    let cancelled = false;
    Promise.all([
      fetch('/data/scenario.json').then((r) => { if (!r.ok) throw new Error(`scenario ${r.status}`); return r.json() as Promise<Scenario>; }),
      fetch('/data/hydrography.geojson').then((r) => { if (!r.ok) throw new Error(`hydrography ${r.status}`); return r.json() as Promise<Hydrography>; }),
    ]).then(([nextScenario, nextHydrography]) => {
      if (cancelled) return;
      setScenario(nextScenario);
      setHydrography(nextHydrography);
      setDataStatus('ready');
      // 초기 장면 = 최신 **광학(S2)**. 최신 전체로 하면 S1 레이더(평균 밝기 35/255,
      // 사실상 검은 이미지)가 화면을 덮어 "지도가 안 나온다"로 보인다 — 실제 발생한 문제.
      const ready = [...nextScenario.scene_records].sort((a, b) => a.acquired_at.localeCompare(b.acquired_at));
      const latestOptical = [...ready].reverse().find((s) => shortSensor(s.sensor) === 'S2');
      setActiveSceneId((current) => current ?? latestOptical?.id ?? ready[ready.length - 1]?.id ?? null);
    }).catch(() => { if (!cancelled) setDataStatus('failed'); });
    return () => { cancelled = true; };
  }, [reloadKey]);

  // ── 타임라인: 전부 scenario.json에서 파생 ──
  const timeline = useMemo<TimelineItem[]>(() => {
    if (!scenario) return [];
    const items: TimelineItem[] = scenario.scene_records.map((s) => ({
      id: s.id, kind: 'scene', iso: s.acquired_at, date: shortDate(s.acquired_at),
      sensor: shortSensor(s.sensor),
      state: s.state === 'live_partial' ? 'LIVE·PART' : s.state === 'live_ready' ? 'LIVE' : 'READY',
      selectable: true,
    }));
    items.push({
      id: 'event', kind: 'event', iso: scenario.event.occurred_at,
      date: shortDate(scenario.event.occurred_at), sensor: 'EVENT', state: 'IMPACT', selectable: false,
    });
    scenario.scheduled_scenes.forEach((s, i) => items.push({
      id: `scheduled-${i}`, kind: 'scheduled', iso: s.acquired_at, date: shortDate(s.acquired_at),
      sensor: shortSensor(s.sensor),
      state: s.state === 'missed_coverage' ? 'MISSED' : s.state === 'planned' ? 'PLANNED' : s.state === 'catalog_published_cloudy' ? 'CATALOG' : 'PENDING',
      selectable: false,
    }));
    return items.sort((a, b) => a.iso.localeCompare(b.iso));
  }, [scenario]);

  const points = useMemo(() => scenario?.points ?? [], [scenario]);
  const researchPoints = useMemo<FeatureCollection>(() => ({
    type: 'FeatureCollection',
    features: points.map((p) => ({
      type: 'Feature', properties: {
        id: p.id, name: p.name, display_label: p.display_label, map_label: p.map_label, stage: p.stage,
        marker_color: p.marker_color, in_event_chain: p.in_event_chain,
      },
      geometry: { type: 'Point', coordinates: p.coordinates },
    })),
  }), [points]);

  useEffect(() => {
    if (!mapNode.current || mapRef.current) return;
    // 생성 전에 능력을 직접 조사한다 — MapLibre v6은 supported()가 없다.
    const probe = document.createElement('canvas');
    const gl = probe.getContext('webgl2');
    // 진단 — 화면을 볼 수 없을 때 한 번의 새로고침으로 원인을 가리기 위한 로그.
    const box = mapNode.current.getBoundingClientRect();
    const cs = getComputedStyle(mapNode.current);
    console.log('[diag] container layout | client =', mapNode.current.clientWidth + 'x' + mapNode.current.clientHeight,
                '| rect =', Math.round(box.width) + 'x' + Math.round(box.height),
                '| position =', cs.position, '| inset =', cs.inset,
                '| maplibreCss =', !!Array.from(document.styleSheets).find((sh) => {
                  try { return Array.from(sh.cssRules).some((r) => (r as CSSStyleRule).selectorText?.includes('maplibregl-canvas')); }
                  catch { return false; }
                }));
    console.log('[diag] webgl2 =', !!gl,
                '| container =', Math.round(box.width) + 'x' + Math.round(box.height),
                '| style = local-scene-backdrop + lightRasterStyle');
    if (!gl) { console.error('[diag] WebGL2 미지원 → 지도를 만들지 않고 종료함'); queueMicrotask(() => setMapStatus('unsupported')); return; }
    // CSS 시트 순서와 무관하게 컨테이너 크기를 보장한다 (인라인 = 최우선).
    // 실측: maplibre-gl.css 의 .maplibregl-map { position:relative } 가 로드 순서에 따라
    // .map-stage { position:absolute; inset:0 } 를 덮어 clientHeight 가 0 이 됐음.
    Object.assign(mapNode.current.style, {
      position: 'absolute', top: '0', right: '0', bottom: '0', left: '0',
      width: '100%', height: '100%',
    });
    try {
      const map = new MapLibreMap({
        container: mapNode.current,
        style: maptilerStyleUrl ?? basemapStyle,
        // 첫 인상은 A/B 한 점이 아니라 E→F 전체 사건 사슬이다. 이후 사용자가
        // SATELLITE FRAME/타임라인을 고를 때만 2.56 km 장면으로 들어간다.
        center: [85.33, 28.10],
        zoom: 9.35,
        pitch: 0,
        bearing: 0,
        maxPitch: 72,
        // 서비스 범위를 네팔·티베트 국경 회랑으로 잠금 — Trishuli 하류(Galchhi)에서
        // Kyirong(티베트) 상류까지. 언색호 lake_watch 회랑(국경 북쪽 ~20km)을 포함함.
        maxBounds: [[83.2, 26.6], [87.8, 29.8]],
        minZoom: 7,
        attributionControl: false,
      });
      map.addControl(new NavigationControl({ showCompass: true }), 'bottom-right');
      map.addControl(new AttributionControl({ compact: true }), 'bottom-right');
      map.on('styledata', () => console.log('[diag] styledata — 레이어',
        map.getStyle()?.layers?.length ?? 0, '개'));
      map.on('style.load', () => setStyleRevision((revision) => revision + 1));
      // MapTiler 스타일이 죽으면(403/네트워크) Esri raster 로 강등 — 화면 성립을 키에 맡기지 않음.
      let fellBack = false;
      map.on('error', (e) => {
        const msg = String((e as { error?: { message?: string } }).error?.message ?? '');
        if (!fellBack && maptilerStyleUrl && /style|403|Forbidden|Failed to fetch/i.test(msg)) {
          fellBack = true;
          console.warn('[diag] MapTiler 스타일 실패 → Esri 폴백:', msg);
          map.setStyle(basemapStyle as unknown as Parameters<typeof map.setStyle>[0]);
        }
      });
      map.on('load', () => {
        // 초기 캔버스가 컨테이너보다 작게 잡히는 버그(실측 1440x300 vs 1440x813) 방지.
        map.resize();
        // WebGL 3D 지형 — Terrarium DEM. 기본은 2D(판독·비교 좌표계), 3D는 토글로 켬.
        if (viewDimRef.current === '3d') {
          try {
            if (!map.getSource('terrainDem')) map.addSource('terrainDem', TERRAIN_DEM_SPEC);
            map.setTerrain({ source: 'terrainDem', exaggeration: 1.3 });
          }
          catch (e) { console.warn('[diag] terrain 활성화 실패 — 평면 유지', e); }
        }
        // 진단: MapLibre가 실제로 무엇을 재는지 — private이지만 원인 확정용.
        const anyMap = map as unknown as { _container?: HTMLElement; _containerDimensions?: () => [number, number] };
        console.log('[diag] maplibre 내부 | sameContainer =', anyMap._container === mapNode.current,
                    '| _containerDimensions =', JSON.stringify(anyMap._containerDimensions?.()),
                    '| container.clientWH =', (anyMap._container?.clientWidth ?? -1) + 'x' + (anyMap._container?.clientHeight ?? -1));
        const b = map.getCanvas();
        console.log('[diag] load 완료 | canvas =', b.width + 'x' + b.height,
                    '| 소스 =', Object.keys(map.getStyle()?.sources ?? {}).join(','));
        setMapReady(true); setMapStatus('ready');
      });
      let tileCount = 0;
      map.on('data', (e) => {
        // e.tile 이 있으면 타일 한 장이 실제로 도착한 것이다.
        if ((e as { tile?: unknown }).tile) {
          tileCount += 1;
          if (tileCount <= 3) console.log('[diag] tile 도착:', (e as { sourceId?: string }).sourceId);
        }
      });
      map.on('idle', () => {
        const c = map.getCanvas();
        console.log('[diag] idle | 타일', tileCount, '장 | canvas =', c.width + 'x' + c.height,
                    '| 레이어', map.getStyle()?.layers?.length ?? 0);
      });
      // 외부 context tile 실패는 진단만 남긴다. 실제 S2 backdrop과 로컬 evidence layer는 독립이다.
      map.on('error', (e) => {
        const msg = e?.error?.message ?? String(e);
        console.error('[map] error:', msg);
      });
      mapRef.current = map;
      // 컨테이너가 나중에 커지면 MapLibre는 스스로 캔버스를 늘리지 않는다.
      // 실측: container 1440x813 인데 canvas 1440x300 이라 지도가 얇은 띠로만 그려졌다.
      // 주의: 조건 없이 resize()를 호출하면 ResizeObserver가 자기 자신을 다시 깨워
      // 무한 루프가 된다(실측: headless Chrome이 5분간 종료되지 않았음).
      // 컨테이너 크기가 **실제로** 바뀐 경우에만 한 번 호출한다.
      let lastW = 0, lastH = 0;
      const ro = new ResizeObserver((entries) => {
        const r = entries[0]?.contentRect;
        if (!r) return;
        const w = Math.round(r.width), h = Math.round(r.height);
        if (w === lastW && h === lastH) return;
        lastW = w; lastH = h;
        if (w === 0 || h === 0) return;
        map.resize();
        const c = map.getCanvas();
        console.log('[diag] resize', w + 'x' + h, '→ canvas =', c.width + 'x' + c.height);
      });
      ro.observe(mapNode.current);
      return () => { ro.disconnect(); map.remove(); mapRef.current = null; };
    } catch {
      queueMicrotask(() => setMapStatus('unsupported'));
      return;
    }
  }, []);

  // 연구 지점 레이어 — points가 데이터에서 오므로 로드 후에 붙인다.
  useEffect(() => {
    const map = mapRef.current;
    if (!mapReady || !map || !map.isStyleLoaded() || points.length === 0) return;
    if (!map.getSource('research-points')) {
      map.addSource('research-points', { type: 'geojson', data: researchPoints });
      map.addLayer({
        id: 'point-halo', type: 'circle', source: 'research-points',
        paint: {
          'circle-radius': ['case', ['==', ['get', 'id'], 'E'], 21, ['==', ['get', 'id'], 'A'], 18, 12],
          'circle-color': ['get', 'marker_color'],
          'circle-opacity': ['case', ['==', ['get', 'id'], 'C'], 0.08, 0.2], 'circle-stroke-width': 1,
          'circle-stroke-color': ['get', 'marker_color'],
        },
      });
      map.addLayer({
        id: 'point-core', type: 'circle', source: 'research-points',
        paint: {
          'circle-radius': ['case', ['==', ['get', 'id'], 'E'], 7, ['==', ['get', 'id'], 'A'], 6, 4.5],
          'circle-color': ['get', 'marker_color'],
          'circle-stroke-width': 2, 'circle-stroke-color': '#081411',
        },
      });
      map.addLayer({
        id: 'point-label', type: 'symbol', source: 'research-points',
        layout: {
          'text-field': ['get', 'map_label'], 'text-size': 11, 'text-offset': [0, 1.55],
          'text-anchor': 'top', 'text-font': ['Noto Sans Regular'], 'text-allow-overlap': true,
        },
        paint: {
          'text-color': ['get', 'marker_color'], 'text-halo-color': '#071713', 'text-halo-width': 1.4,
        },
      });
    }
    const onPointClick = (event: MapLayerMouseEvent) => {
      const id = event.features?.[0]?.properties?.id;
      if (!id) return;
      setSelectedPoint(String(id));
      const pt = points.find((x) => x.id === String(id));
      if (pt) {
        // 점별 실측 위성 창 — A/B는 rasuwagadhi 앵커 창, D/E는 발원 수색 창.
        // C(원거리 참조)는 물질화 창이 없어 썸네일 없음.
        const win = ({ A: 'rasuwagadhi', B: 'rasuwagadhi', D: 'source', E: 'source', F: 'bidur' } as Record<string, string>)[pt.id];
        const thumbs = win
          ? `<div class="pp-thumbs">`
            + `<figure><img src="/data/story/anchors/${win}_pre.png" alt="pre"/><figcaption>PRE 08-12</figcaption></figure>`
            + `<figure><img src="/data/story/anchors/${win}_post.png" alt="post"/><figcaption>POST 08-27</figcaption></figure>`
            + `</div>`
          : '';
        new Popup({ closeButton: true, maxWidth: '320px', className: 'story-popup' })
          .setLngLat(pt.coordinates)
          .setHTML(`<p class="pp-eyebrow" style="color:${pt.marker_color}">${pt.display_label}${pt.id === 'C' ? ' · OUTSIDE EVENT CHAIN' : ''}</p>`
            + `<h3>${pt.name}</h3><p class="pp-place">${pt.place}</p>`
            + thumbs
            + (pt.story ? `<p class="pp-story">${pt.story}</p>` : '')
            + `<p class="pp-src">${pt.source_url ? `<a href="${pt.source_url}" target="_blank" rel="noreferrer">${pt.source ?? 'source'} ↗</a>` : (pt.source ?? '')}</p>`)
          .addTo(map);
      }
    };
    const onPointEnter = () => { map.getCanvas().style.cursor = 'pointer'; };
    const onPointLeave = () => { map.getCanvas().style.cursor = ''; };
    map.on('click', 'point-core', onPointClick);
    map.on('mouseenter', 'point-core', onPointEnter);
    map.on('mouseleave', 'point-core', onPointLeave);
    return () => {
      map.off('click', 'point-core', onPointClick);
      map.off('mouseenter', 'point-core', onPointEnter);
      map.off('mouseleave', 'point-core', onPointLeave);
    };
  }, [mapReady, points, researchPoints, styleRevision]);

  useEffect(() => {
    const map = mapRef.current;
    if (!mapReady || !map || !map.isStyleLoaded() || !hydrography || map.getSource('hydrography')) return;
    const before = map.getLayer('point-halo') ? 'point-halo' : undefined;
    map.addSource('hydrography', { type: 'geojson', data: hydrography as FeatureCollection });
    map.addLayer({ id: 'river-casing', type: 'line', source: 'hydrography', paint: { 'line-color': '#06100e', 'line-width': 8, 'line-opacity': 0.82 } }, before);
    map.addLayer({ id: 'river-route', type: 'line', source: 'hydrography', paint: { 'line-color': '#5fffd7', 'line-width': 2.2, 'line-opacity': 0.8, 'line-dasharray': [1.2, 1.6] } }, before);
  }, [hydrography, mapReady, styleRevision]);

  useEffect(() => {
    const map = mapRef.current;
    if (!mapReady || !map || !map.isStyleLoaded()) return;
    fetch('/data/olmo-input-anchors.geojson').then((r) => r.json() as Promise<FeatureCollection>).then((anchors) => {
      if (!map.isStyleLoaded() || map.getSource('olmo-anchors')) return;
      const before = map.getLayer('point-halo') ? 'point-halo' : undefined;
      map.addSource('olmo-anchors', { type: 'geojson', data: anchors });
      map.addLayer({ id: 'olmo-anchor-fill', type: 'fill', source: 'olmo-anchors', paint: { 'fill-color': '#5fffd7', 'fill-opacity': 0.045 } }, before);
      map.addLayer({ id: 'olmo-anchor-line', type: 'line', source: 'olmo-anchors', paint: { 'line-color': '#b7ffe9', 'line-width': 1, 'line-opacity': 0.52, 'line-dasharray': [3, 2] } }, before);
    }).catch(() => undefined);
  }, [mapReady, styleRevision]);

  // fitBounds 패딩은 실제로 열려 있는 패널에 맞춘다.
  // 이전 버전은 패널이 항상 보인다고 가정한 고정 패딩을 썼다.
  const scenePadding = useCallback(() => {
    const wide = window.innerWidth > 1100;
    const { left, right } = railsRef.current;
    return {
      top: 96,
      bottom: window.innerWidth > 720 ? 158 : 190,
      left: wide && left ? 372 : 24,
      right: wide && right ? 372 : 24,
    };
  }, []);

  const fitScene = useCallback((scene: SceneRecord, duration = 900) => {
    const map = mapRef.current;
    if (!map) return;
    const [topLeft, , bottomRight] = scene.coordinates;
    map.fitBounds(
      new LngLatBounds([topLeft[0], bottomRight[1]], [bottomRight[0], topLeft[1]]),
      { padding: scenePadding(), maxZoom: 15.1, pitch: viewDimRef.current === '3d' ? TERRAIN_PITCH : 0, bearing: viewDimRef.current === '3d' ? -18 : 0, duration: prefersReducedMotion() ? 0 : duration },
    );
  }, [scenePadding]);

  useEffect(() => {
    const map = mapRef.current;
    if (!mapReady || !map || !map.isStyleLoaded() || !scenario || !activeSceneId) return;
    const scene = scenario.scene_records.find((item) => item.id === activeSceneId);
    if (!scene) return;
    if (map.getLayer('satellite-scene')) map.removeLayer('satellite-scene');
    if (map.getSource('satellite-scene')) map.removeSource('satellite-scene');
    const before = map.getLayer('point-halo') ? 'point-halo' : undefined;
    map.addSource('satellite-scene', { type: 'image', url: scene.image, coordinates: scene.coordinates });
    map.addLayer({ id: 'satellite-scene', type: 'raster', source: 'satellite-scene', paint: { 'raster-opacity': overlayOpacity, 'raster-fade-duration': 120, 'raster-saturation': 0.12, 'raster-contrast': 0.08, 'raster-resampling': 'nearest' } }, before);
    if (!userSelectedSceneRef.current) {
      // 첫 화면은 단일 A/B 위성창이 아니라 SOURCE→DOWNSTREAM 사건 전체를 보여준다.
      // 사용자가 A/B를 붕괴 원점으로 오독한 직접 원인이 초기 2.56 km 자동 줌이었다.
      map.fitBounds(new LngLatBounds([85.105, 27.885], [85.55, 28.31]), {
        padding: scenePadding(), maxZoom: 10.8, duration: 0,
      });
    } else {
      fitScene(scene, 700);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeSceneId, mapReady, scenario, fitScene, styleRevision]);

  useEffect(() => {
    const map = mapRef.current;
    if (!mapReady || !map || !map.getLayer('satellite-scene')) return;
    map.setPaintProperty('satellite-scene', 'raster-opacity', overlayOpacity);
  }, [mapReady, overlayOpacity]);

  useEffect(() => {
    const map = mapRef.current;
    if (!mapReady || !map || !map.getLayer('olmo-anchor-fill')) return;
    map.setLayoutProperty('olmo-anchor-fill', 'visibility', showAnchors ? 'visible' : 'none');
    map.setLayoutProperty('olmo-anchor-line', 'visibility', showAnchors ? 'visible' : 'none');
  }, [mapReady, showAnchors]);

  useEffect(() => { flowPlayingRef.current = flowPlaying; }, [flowPlaying]);
  useEffect(() => { flowSpeedRef.current = flowSpeed; }, [flowSpeed]);

  useEffect(() => {
    const map = mapRef.current;
    const canvas = canvasRef.current;
    if (!mapReady || !map || !canvas || !hydrography) return;
    let cancelled = false;
    let animationFrame = 0;
    let lastTime = performance.now();

    const start = async () => {
      try {
        const response = await fetch('/wasm/nepal_flow.wasm');
        const instantiated = await WebAssembly.instantiateStreaming(response, {});
        const wasm = instantiated.instance.exports as FlowExports;
        wasmRef.current = wasm;
        if (wasm.abi_version() !== 1) throw new Error('Unexpected WASM ABI');
        wasm.clear_route();
        hydrography.simulation_route.forEach(([lon, lat], index) => wasm.set_route_point(index, lon, lat));
        wasm.reset(20260826);
        setWasmStatus('ready');
        const context = canvas.getContext('2d');
        if (!context) throw new Error('Canvas unavailable');

        const draw = (now: number) => {
          if (cancelled) return;
          const dpr = Math.min(window.devicePixelRatio || 1, 2);
          const width = canvas.clientWidth;
          const height = canvas.clientHeight;
          if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
            canvas.width = Math.round(width * dpr);
            canvas.height = Math.round(height * dpr);
          }
          context.setTransform(dpr, 0, 0, dpr, 0, 0);
          context.clearRect(0, 0, width, height);
          const dt = Math.min((now - lastTime) / 1000, 0.05);
          lastTime = now;
          if (flowPlayingRef.current) wasm.step(dt, flowSpeedRef.current);
          const count = wasm.particle_count();
          const values = new Float32Array(wasm.memory.buffer, wasm.particles_ptr(), count * 3);
          context.globalCompositeOperation = 'lighter';
          context.shadowColor = '#5fffd7';
          context.shadowBlur = 8;
          for (let index = 0; index < count; index += 1) {
            const screen = map.project([values[index * 3], values[index * 3 + 1]]);
            if (screen.x < 0 || screen.y < 0 || screen.x > width || screen.y > height) continue;
            context.globalAlpha = values[index * 3 + 2] * 0.8;
            context.fillStyle = index % 7 === 0 ? '#ffb45f' : '#5fffd7';
            context.beginPath();
            context.arc(screen.x, screen.y, index % 7 === 0 ? 1.8 : 1.15, 0, Math.PI * 2);
            context.fill();
          }
          context.globalAlpha = 1;
          context.shadowBlur = 0;
          animationFrame = requestAnimationFrame(draw);
        };
        animationFrame = requestAnimationFrame(draw);
      } catch {
        setWasmStatus('failed');
      }
    };
    start();
    return () => { cancelled = true; wasmRef.current = null; cancelAnimationFrame(animationFrame); };
  }, [hydrography, mapReady]);

  const activeScene = scenario?.scene_records.find((item) => item.id === activeSceneId) ?? null;
  const latestOpticalScene = useMemo(() => {
    const optical = scenario?.scene_records.filter((scene) => shortSensor(scene.sensor) === 'S2') ?? [];
    return [...optical].sort((a, b) => b.acquired_at.localeCompare(a.acquired_at))[0] ?? null;
  }, [scenario]);
  const setDimension = (dim: '2d' | '3d') => {
    setViewDim(dim); viewDimRef.current = dim;
    const map = mapRef.current;
    if (!map) return;
    try {
      if (dim === '3d') {
        if (!map.getSource('terrainDem')) map.addSource('terrainDem', TERRAIN_DEM_SPEC);
        map.setTerrain({ source: 'terrainDem', exaggeration: 1.3 });
        map.easeTo({ pitch: TERRAIN_PITCH, bearing: -18, duration: prefersReducedMotion() ? 0 : 800 });
      } else {
        map.setTerrain(null);
        map.easeTo({ pitch: 0, bearing: 0, duration: prefersReducedMotion() ? 0 : 800 });
      }
    } catch (e) { console.warn('[diag] terrain 전환 실패', e); }
  };

  const backdropScene = activeScene && shortSensor(activeScene.sensor) === 'S2' ? activeScene : latestOpticalScene;
  const missedCoverage = scenario?.scheduled_scenes.find((scene) => scene.state === 'missed_coverage') ?? null;
  const nextScheduled = scenario?.scheduled_scenes.find((scene) => scene.state !== 'missed_coverage') ?? null;
  const nextRadar = scenario?.scheduled_scenes.find((scene) => scene.state !== 'missed_coverage' && shortSensor(scene.sensor) === 'S1') ?? null;
  const liveObservation = scenario?.live_observation ?? null;
  const decision = scenario?.decision ?? null;
  const transfer = scenario?.research.confirmatory_transfer ?? null;
  const livePeriodText = liveObservation
    ? `S1 ${liveObservation.period_readiness?.sentinel1 ?? '?'}⁄4 · S2 ${liveObservation.period_readiness?.sentinel2_l2a ?? '?'}⁄4`
    : '—';
  const liveReadinessLabel = !liveObservation
    ? 'NO LIVE OBSERVATION'
    : liveObservation.olmo_ready
      ? 'OLMo INPUT SEALED'
      : liveObservation.materialization_status === 'partial_cube_contract_failed'
        ? 'PIXELS READY · CUBE INCOMPLETE'
        : liveObservation.materialization_status === 'selected_not_materialized'
          ? 'SCENE SELECTED · MATERIALIZE WAIT'
          : 'INPUT CONTRACT BLOCKED';
  const selectedCard = points.find((item) => item.id === selectedPoint) ?? points[0] ?? null;
  const eventPoints = points.filter((point) => point.in_event_chain);
  const controlPoints = points.filter((point) => !point.in_event_chain);
  const bidurPre = scenario?.downstream_visual.records.find((record) => record.label === 'pre') ?? null;
  const bidurPost = scenario?.downstream_visual.records.find((record) => record.label === 'post') ?? null;

  const focusPoint = (id: string) => {
    setSelectedPoint(id);
    const card = points.find((item) => item.id === id);
    if (!card) return;
    mapRef.current?.flyTo({
      center: card.coordinates, zoom: id === 'C' ? 10.5 : id === 'F' ? 13.2 : 14,
      pitch: viewDimRef.current === '3d' ? TERRAIN_PITCH : 0, bearing: viewDimRef.current === '3d' ? -18 : 0,
      duration: prefersReducedMotion() ? 0 : 1100,
    });
  };

  const fitCorridor = () => {
    userSelectedSceneRef.current = false;
    mapRef.current?.fitBounds(new LngLatBounds([85.105, 27.885], [85.55, 28.31]), {
      padding: scenePadding(), pitch: viewDimRef.current === '3d' ? TERRAIN_PITCH : 0, bearing: viewDimRef.current === '3d' ? -18 : 0, duration: prefersReducedMotion() ? 0 : 1100,
    });
  };
  const replayEventChain = () => {
    wasmRef.current?.reset(20260826);
    setFlowPlaying(true);
    fitCorridor();
  };

  // 타임라인 키보드 탐색: ←/→ 로 READY 장면 사이 이동.
  // River corridor 미니 도식 — 검증된 OSM centerline(78점)을 그대로 축소해 그림.
  // 앵커 4곳(라수와가디→티무레→샤브루베시→둔체)을 실좌표로 route에 투영함.
  const corridorSketch = useMemo(() => {
    const route = hydrography?.simulation_route;
    if (!route || route.length < 2) return null;
    const lons = route.map((p) => p[0]); const lats = route.map((p) => p[1]);
    const minLon = Math.min(...lons), maxLon = Math.max(...lons);
    const minLat = Math.min(...lats), maxLat = Math.max(...lats);
    const W = 260, H = 84, PAD = 8;
    const sx = (lon: number) => PAD + ((lon - minLon) / (maxLon - minLon || 1)) * (W - 2 * PAD);
    const sy = (lat: number) => PAD + ((maxLat - lat) / (maxLat - minLat || 1)) * (H - 2 * PAD);
    const path = route.map((pt, i) => `${i === 0 ? 'M' : 'L'}${sx(pt[0]).toFixed(1)},${sy(pt[1]).toFixed(1)}`).join(' ');
    const anchors: { name: string; lon: number; lat: number }[] = [
      { name: 'Rasuwagadhi', lon: 85.378, lat: 28.276 },
      { name: 'Timure', lon: 85.363, lat: 28.235 },
      { name: 'Syabrubesi', lon: 85.347, lat: 28.164 },
      { name: 'Dhunche', lon: 85.296, lat: 28.102 },
      { name: 'Trishuli Bazar', lon: 85.1357, lat: 27.9162 },
    ];
    // route 위 최근접점에 스냅해 앵커가 강 선 위에 앉게 함
    const dots = anchors.map((a) => {
      let best = route[0]; let bd = Infinity;
      for (const pt of route) {
        const d = (pt[0] - a.lon) ** 2 + (pt[1] - a.lat) ** 2;
        if (d < bd) { bd = d; best = pt; }
      }
      return { name: a.name, x: sx(best[0]), y: sy(best[1]) };
    });
    return { W, H, path, dots };
  }, [hydrography]);

  // STORY 오버레이 — Snow Fall식 스크롤리텔링: IntersectionObserver로 섹션 표시,
  // 진행 바는 스크롤 비율. prefers-reduced-motion 이면 항상 표시 상태로 시작함.
  const storyRef = useRef<HTMLDivElement>(null);
  const [storyProgress, setStoryProgress] = useState(0);
  useEffect(() => {
    if (!storyOpen) return;
    const root = storyRef.current;
    if (!root) return;
    const reduced = prefersReducedMotion();
    const sections = Array.from(root.querySelectorAll<HTMLElement>('.story-step'));
    if (reduced) { sections.forEach((s) => s.classList.add('in-view')); }
    const io = reduced ? null : new IntersectionObserver((entries) => {
      entries.forEach((e) => { if (e.isIntersecting) e.target.classList.add('in-view'); });
    }, { root, threshold: 0.25 });
    if (io) sections.forEach((s) => io.observe(s));
    const onScroll = () => {
      const max = root.scrollHeight - root.clientHeight;
      setStoryProgress(max > 0 ? Math.min(1, root.scrollTop / max) : 0);
    };
    root.addEventListener('scroll', onScroll, { passive: true });
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setStoryOpen(false); };
    window.addEventListener('keydown', onKey);
    return () => { io?.disconnect(); root.removeEventListener('scroll', onScroll); window.removeEventListener('keydown', onKey); };
  }, [storyOpen]);

  const sceneById = useCallback((id: string) => scenario?.scene_records.find((s) => s.id === id) ?? null, [scenario]);

  const readyIds = useMemo(() => timeline.filter((t) => t.selectable).map((t) => t.id), [timeline]);
  const onTimelineKey = (event: React.KeyboardEvent) => {
    if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
    event.preventDefault();
    if (!activeSceneId || readyIds.length === 0) return;
    const at = readyIds.indexOf(activeSceneId);
    const next = event.key === 'ArrowRight' ? Math.min(at + 1, readyIds.length - 1) : Math.max(at - 1, 0);
    userSelectedSceneRef.current = true;
    setActiveSceneId(readyIds[next]);
  };

  return (
    <main className="app-shell">
      {/* 장면은 지도 안의 지리참조 레이어('satellite-scene' image source)로만 그린다.
          이전의 DOM 고정 backdrop은 ① 드래그해도 움직이지 않고 ② WASM flow(지도 좌표)와
          어긋나며 ③ 지도 캔버스와 basemap을 가렸다. WebGL2가 없을 때만 정적 이미지로
          내려간다(아래 map-fallback). */}
      <div ref={mapNode} className="map-stage" aria-label="Rasuwagadhi satellite and simulation map" />
      {mapStatus !== 'unsupported' && <canvas ref={canvasRef} className="flow-canvas" aria-hidden="true" />}
      <div className="terrain-wash" aria-hidden="true" />
      {mapStatus === 'unsupported' && (
        <div className="map-fallback">
          {backdropScene && <Image src={backdropScene.image} alt="" fill unoptimized className="map-fallback-img" />}
          <div className="map-fallback-note" role="status">
            <strong>Interactive map unavailable — WebGL2 is off in this browser.</strong>
            <span>Showing the selected scene as a static image. Timeline and panels still work.
            Enable hardware acceleration (chrome://settings/system) or try another browser for the full map.</span>
          </div>
        </div>
      )}

      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark"><span /></div>
          <div><p className="eyebrow">AI2 / PLANETARY INTELLIGENCE PROTOTYPE</p><h1>OLMoEarth <span>Live Twin</span></h1></div>
        </div>
        <div className="map-mode-switch" role="group" aria-label="Map focus">
          <button onClick={() => { userSelectedSceneRef.current = true; if (activeScene) fitScene(activeScene, 900); }} disabled={!activeScene || mapStatus !== 'ready'}>SATELLITE FRAME</button>
          <button onClick={fitCorridor} disabled={mapStatus !== 'ready'}>EVENT CHAIN</button>
        </div>
        <button className="story-launch" onClick={() => setStoryOpen(true)}>STORY</button>
        <div className="map-mode-switch dim-switch" role="group" aria-label="View dimension">
          <button className={viewDim === '2d' ? 'is-active' : ''} onClick={() => setDimension('2d')} disabled={mapStatus !== 'ready'}>2D</button>
          <button className={viewDim === '3d' ? 'is-active' : ''} onClick={() => setDimension('3d')} disabled={mapStatus !== 'ready'}>3D</button>
        </div>
        <div className="event-status"><span className="live-dot" /><div><strong>RASUWA · NEPAL</strong><small>{scenario ? `${shortDate(scenario.event.occurred_at)} 2026 · INVESTIGATION` : 'LOADING'}</small></div></div>
      </header>

      {dataStatus === 'failed' && (
        <div className="data-error" role="alert">
          <strong>Snapshot data failed to load.</strong>
          <span>scenario.json / hydrography.geojson could not be fetched.</span>
          <button onClick={() => { setDataStatus('loading'); setReloadKey((k) => k + 1); }}>Retry</button>
        </div>
      )}

      <button
        className={`rail-toggle left ${leftOpen ? 'open' : ''}`}
        aria-expanded={leftOpen}
        aria-label={leftOpen ? 'Hide area panel' : 'Show area panel'}
        onClick={() => setLeftOpen((v) => !v)}
      >{leftOpen ? '⟨' : '⟩'}<em>AOI</em></button>

      <button
        className={`rail-toggle right ${rightOpen ? 'open' : ''}`}
        aria-expanded={rightOpen}
        aria-label={rightOpen ? 'Hide evidence panel' : 'Show evidence panel'}
        onClick={() => setRightOpen((v) => !v)}
      ><em>EVIDENCE</em>{rightOpen ? '⟩' : '⟨'}</button>

      {leftOpen && (
      <aside className="left-rail glass-panel">
        <div className="panel-heading"><span>01</span><div><p>EVENT ANATOMY</p><strong>Source → downstream</strong></div></div>
        <div className="coordinate-list">
          {eventPoints.map((point) => (
            <button key={point.id} style={{ '--point-color': point.marker_color } as CSSProperties} className={selectedPoint === point.id ? 'coordinate active' : 'coordinate'} onClick={() => focusPoint(point.id)}>
              <span>{point.stage}</span>
              <div>
                <em className="point-role">{point.display_label}</em><strong>{point.name}</strong><small>{point.coordinates[1].toFixed(6)}, {point.coordinates[0].toFixed(6)}</small>
                {selectedPoint === point.id && point.story && <p className="point-story">{point.story}</p>}
              </div>
              <em>{point.id}</em>
            </button>
          ))}
          {points.length === 0 && <p className="rail-empty">{dataStatus === 'loading' ? 'Loading points…' : 'No points in snapshot.'}</p>}
        </div>
        {controlPoints.length > 0 && <div className="control-group"><span>OUTSIDE THE EVENT CHAIN</span>{controlPoints.map((point) => (
          <button key={point.id} style={{ '--point-color': point.marker_color } as CSSProperties} className={selectedPoint === point.id ? 'coordinate control active' : 'coordinate control'} onClick={() => focusPoint(point.id)}>
            <span>Ø</span><div><em className="point-role">{point.display_label}</em><strong>{point.name}</strong><small>~{point.distance_from_a_km.toFixed(0)} km away · placebo only</small>{selectedPoint === point.id && point.story && <p className="point-story">{point.story}</p>}</div><em>{point.id}</em>
          </button>
        ))}</div>}
        {selectedCard && (
          <div className="selected-place">
            <span>{selectedCard.id === 'A' ? 'REFERENCE IMPACT WINDOW' : `${selectedCard.distance_from_a_km.toFixed(2)} km FROM IMPACT A`}</span>
            <strong>{selectedCard.place}</strong>
          </div>
        )}
        <p className="audit-note"><b>Red E is the collapse source estimate.</b> Orange A is the impact window. Gray C is a negative control and never belongs to the flood path.</p>
        <div className="layer-controls">
          <label htmlFor="overlay-opacity"><span>Satellite overlay</span><b>{Math.round(overlayOpacity * 100)}%</b></label>
          <input id="overlay-opacity" type="range" min="0" max="1" step="0.02" value={overlayOpacity} onChange={(event) => setOverlayOpacity(Number(event.target.value))} />
          <button className={showAnchors ? 'toggle active' : 'toggle'} onClick={() => setShowAnchors((value) => !value)} aria-pressed={showAnchors}><i /> OLMo input windows</button>
        </div>
        {/* River corridor 도식 — 강 모양과 앵커 순서를 지도 줌과 무관하게 항상 보여줌.
            선은 검증된 OSM centerline 그대로이고 개형/모식도가 아님. */}
        {corridorSketch && (
          <div className="corridor-sketch">
            <span className="ops-title">RIVER CORRIDOR · Bhote Koshi → Trishuli</span>
            <svg viewBox={`0 0 ${corridorSketch.W} ${corridorSketch.H}`} role="img"
                 aria-label="Bhote Koshi to Trishuli corridor with four anchors from Rasuwagadhi down to Dhunche">
              <path d={corridorSketch.path} fill="none" stroke="var(--blue)" strokeWidth="1.8"
                    strokeLinecap="round" strokeLinejoin="round" />
              {corridorSketch.dots.map((d, i) => (
                <g key={d.name}>
                  <circle cx={d.x} cy={d.y} r="3.4"
                          fill={i === 0 ? 'var(--orange)' : 'var(--surface)'}
                          stroke={i === 0 ? 'var(--orange)' : 'var(--blue)'} strokeWidth="1.6" />
                  <text x={d.x + 7} y={d.y + 3.5} fontSize="8.5"
                        fontFamily="var(--font-geist-mono)" fill="var(--muted)">{d.name}</text>
                </g>
              ))}
              <text x={corridorSketch.W - 8} y={corridorSketch.H - 6} textAnchor="end"
                    fontSize="8" fontFamily="var(--font-geist-mono)" fill="var(--muted)">▼ downstream</text>
            </svg>
          </div>
        )}
        {/* EarthRanger식 이벤트 피드 — 파이프라인이 한 일과 거부한 일의 감사 로그.
            레코드는 catalog/preflight/manifest/report의 시간 기준과 evidence URI에서 파생함. */}
        {scenario?.ops_log && scenario.ops_log.length > 0 && (
          <div className="ops-log">
            <span className="ops-title">OPERATIONS LOG</span>
            <div className="ops-scroll">
              {scenario.ops_log.map((e, i) => (
                <div key={i} className={`ops-row ${e.priority}`}>
                  <i aria-hidden="true" />
                  <div>
                    <b>{e.type}</b> <em>{e.time_utc.slice(5, 16).replace('T', ' ')}Z · {e.source}</em>
                    <small>{e.summary}</small>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
        <div className="map-legend-inline">
          <span><i className="mint" />Verified river route</span>
          <span><i className="white" />OLMo input 2.56 km</span>
          <span><i className="amber" />Unverified / pending</span>
        </div>
      </aside>
      )}

      {rightOpen && (
      <aside className="right-rail glass-panel">
        <div className="panel-heading"><span>02</span><div><p>AI EVIDENCE</p><strong>What works now</strong></div></div>
        <div className="olmo-outcomes">
          <article className="ready"><span>OLMo BASELINE</span><strong>5 ANCHORS READY</strong><small>S1+S2 × 4 periods · 768-d embeddings sealed</small></article>
          <article className="win"><span>TRANSFER EVIDENCE</span><strong>{transfer ? `${transfer.wins_reuse_vs_raw_strong}/${transfer.regions} REGIONS WON` : 'LOADING'}</strong><small>{transfer ? `region-macro ${transfer.reuse_region_macro.toFixed(3)} vs ${transfer.raw_strong_region_macro.toFixed(3)} · +${transfer.absolute_gap.toFixed(3)}` : 'confirmatory summary'}</small></article>
          <article className="wait"><span>NEPAL LIVE CHANGE</span><strong>WAITING FOR S1</strong><small>{livePeriodText} · baseline value remains usable</small></article>
        </div>
        {decision && (
          <div className={`decision-card compact ${decision.status}`} role="status">
            <span>LIVE NEPAL GATE · NOT THE WHOLE MODEL</span>
            <strong>{decision.action}</strong>
            <p>{decision.reason}</p>
            <small><b>NEXT GATE</b>{decision.next_gate}</small>
            <em>{decision.allowed_claim}</em>
          </div>
        )}
        <div className="compare-strip">
          <div className="scene-preview">
            {activeScene ? <Image src={activeScene.image} alt={`${activeScene.sensor} pre-event observation`} fill unoptimized sizes="150px" /> : <span className="loading-grid" />}
            <span>{activeScene && scenario && activeScene.acquired_at >= scenario.event.occurred_at ? 'POST' : 'PRE'} · {activeScene?.acquired_at.slice(0, 10) ?? (dataStatus === 'loading' ? 'LOADING' : '—')}</span>
          </div>
          <div className="compare-arrow" aria-hidden="true">→</div>
          <div className="scene-preview pending-preview">
            <span className="waiting-cross" />
            <span>POST · {liveObservation?.catalog_status === 'published' ? 'CATALOG / CUBE WAIT' : nextScheduled ? `${shortSensor(nextScheduled.sensor)} ${nextScheduled.acquired_at.slice(5, 10)}` : 'PENDING'}</span>
          </div>
        </div>
        {liveObservation && (
          <div className="live-observation" role="status">
            <span>LIVE CATALOG UPDATE</span>
            <strong>{shortSensor(liveObservation.sensor)} {shortDate(liveObservation.acquired_at)} · {liveObservation.catalog_status.toUpperCase()}</strong>
            <small>{kstStamp(liveObservation.publication_utc)} KST · TILE CLOUD {liveObservation.cloud_cover_tile_pct?.toFixed(2) ?? '—'}%</small>
            <em>{liveReadinessLabel} · {livePeriodText}</em>
          </div>
        )}
        {missedCoverage && (
          <div className="coverage-miss" role="status">
            <span>FOOTPRINT AUDIT</span>
            <strong>{shortSensor(missedCoverage.sensor)} {shortDate(missedCoverage.acquired_at)} · MISSED AOI</strong>
            <small>2 nearby products · 0 contained Langtang Lirung</small>
            <em>Not a publication delay. The next S1 gate is {nextRadar ? `${kstStamp(nextRadar.acquired_at)} KST` : 'not yet scheduled'}.</em>
          </div>
        )}
        <div className="pipeline-stack">
          <div className="pipeline-row ready"><span>OE</span><div><strong>Frozen representation</strong><small>sealed baseline · retrieval · downstream probes</small></div><b>READY</b></div>
          <div className="pipeline-row ready"><span>DV</span><div><strong>Bidur downstream pair</strong><small>{bidurPost ? `S2 ${bidurPost.acquired_at.slice(0, 10)} · tile ${bidurPost.mgrs_tile}` : 'visual audit'}</small></div><b>{bidurPost ? 'READY' : 'AUDIT'}</b></div>
          <div className="pipeline-row ready"><span>8R</span><div><strong>Cross-region transfer</strong><small>{transfer ? `${transfer.strong_wins} strong wins · ${transfer.non_win_regions.length} non-wins` : 'confirmatory'}</small></div><b>MEASURED</b></div>
          <div className="pipeline-row pending"><span>ΔN</span><div><strong>Nepal live embedding delta</strong><small>post S2 exists · final S1 period absent</small></div><b>WAIT S1</b></div>
          <div className={`pipeline-row ${wasmStatus === 'ready' ? 'preview' : 'pending'}`}><span>Φ</span><div><strong>Physics ensemble</strong><small>r.avaflow primary · D-Claw check · satellite likelihood</small></div><b>NEXT BUILD</b></div>
        </div>
        {/* O/E/P/H 4-layer 계약 — 설계 문서의 관측/증거/물리/공식 분리를 UI에 명시함.
            P·H는 아직 산출물이 없으므로 회색 placeholder로 정직하게 표시함. */}
        <div className="layer-contract">
          <span>LAYER CONTRACT</span>
          <div className="layer-contract-row on"><b>O</b><span>Observation — S1 VV/VH · S2 12-band · masks</span><em>ACTIVE</em></div>
          <div className={`layer-contract-row ${(typeof scenario?.olmoearth?.post_event_delta === 'object' && (scenario.olmoearth.post_event_delta as Record<string, unknown>).live_mode) ? 'on' : 'off'}`}><b>E</b><span>OLMo evidence — 768-d embedding · Δz · neighbours</span><em>{(typeof scenario?.olmoearth?.post_event_delta === 'object' && (scenario.olmoearth.post_event_delta as Record<string, unknown>).live_mode) ? 'ACTIVE' : liveObservation?.olmo_ready ? 'EMBED WAIT' : 'PENDING'}</em></div>
          <div className="layer-contract-row off"><b>P</b><span>Physics — r.avaflow ensemble · D-Claw check</span><em>DESIGNED</em></div>
          <div className="layer-contract-row off"><b>H</b><span>Human/official — Charter · CEMS · USGS review</span><em>EXTERNAL</em></div>
        </div>
        <div className="flow-control">
          <button onClick={replayEventChain} aria-label="Replay the event-chain corridor animation">REPLAY CHAIN</button>
          <div>
            <label htmlFor="flow-speed"><span>{flowPlaying ? 'ROUTE PLAYING' : 'ROUTE PAUSED'}</span><b>{(flowSpeed / 0.034).toFixed(1)}×</b></label>
            <input id="flow-speed" type="range" min="0.012" max="0.09" step="0.002" value={flowSpeed} onChange={(event) => setFlowSpeed(Number(event.target.value))} />
          </div>
        </div>
        <button className="flow-pause" onClick={() => setFlowPlaying((value) => !value)}>{flowPlaying ? 'PAUSE PARTICLES' : 'RESUME PARTICLES'}</button>
        <div className="truth-box"><span>CLAIM BOUNDARY</span><p>Particles follow the verified OSM Bhote Koshi→Trishuli centerline. They show interface flow, not flood depth, arrival time, or hazard.</p></div>
      </aside>
      )}

      <section className="timeline glass-panel" aria-label="Satellite acquisition timeline" onKeyDown={onTimelineKey}>
        <div className="timeline-title">
          <span>03</span>
          <div>
            <p>SCENE TIMELINE · ←/→</p>
            <strong>
              {activeScene
                ? `${shortDate(activeScene.acquired_at)} · ${shortSensor(activeScene.sensor)} · ON MAP${shortSensor(activeScene.sensor) === 'S1' ? ' · RADAR IS DARK BY NATURE' : ''}`
                : dataStatus === 'loading' ? 'LOADING SNAPSHOT' : 'NO SCENE SELECTED'}
            </strong>
          </div>
        </div>
        <div className="scene-track">
          {timeline.map((scene) => scene.selectable ? (
            <button
              key={scene.id}
              className={scene.id === activeSceneId ? 'scene active' : 'scene'}
              onClick={() => { userSelectedSceneRef.current = true; setActiveSceneId(scene.id); }}
              aria-pressed={scene.id === activeSceneId}
            >
              <span className={`scene-node ${scene.state.toLowerCase()}`} /><strong>{scene.date}</strong><small>{scene.sensor}</small><em>{scene.state}</em>
            </button>
          ) : (
            <div
              key={scene.id}
              className={`scene static ${scene.kind}`}
              title={scene.kind === 'event' ? scenario?.event.name : 'Not yet acquirable — cannot be shown on the map'}
            >
              <span className={`scene-node ${scene.state.toLowerCase()}`} /><strong>{scene.date}</strong><small>{scene.sensor}</small><em>{scene.state}</em>
            </div>
          ))}
          {timeline.length === 0 && <p className="rail-empty">{dataStatus === 'failed' ? 'Timeline unavailable.' : 'Loading acquisitions…'}</p>}
        </div>
      </section>

      {storyOpen && (() => {
        const ko = storyLang === 'ko';
        return (
        <div className="story-overlay" ref={storyRef} role="dialog" aria-label="How to read this service">
          <div className="story-progress" style={{ width: `${storyProgress * 100}%` }} />
          <div className="story-lang" role="group" aria-label="Story language">
            <button className={!ko ? 'is-active' : ''} onClick={() => setStoryLang('en')}>EN</button>
            <button className={ko ? 'is-active' : ''} onClick={() => setStoryLang('ko')}>한국어</button>
          </div>
          <button className="story-close" onClick={() => setStoryOpen(false)} aria-label="Close story">×</button>

          <section className="story-hero story-step">
            <p className="story-dateline">RASUWA, NEPAL · 26 AUG 2026 · {ko ? '실측 갱신' : 'EVIDENCE UPDATED'} {scenario ? kstStamp(scenario.generated_at) : '—'} KST</p>
            <h1>{ko ? '산에서 시작해, 강에서 보였다.' : 'It began on a mountain. It appeared in a river.'}</h1>
            <p className="story-lede">{ko
              ? 'Langtang Lirung의 암반–빙하 붕괴, Rasuwagadhi 국경 충격, 그리고 약 47 km 하류 Bidur의 Sentinel-2 변화까지 하나의 검증 사슬로 잇는다. 질문은 “AI가 재해를 예언했나”가 아니다. OLMoEarth가 여러 센서와 지역을 공통 표현으로 묶고, 물리 앙상블 중 관측과 맞는 설명을 더 빨리 찾게 할 수 있는가이다.'
              : 'This system links the Langtang Lirung rock–ice collapse, the Rasuwagadhi border impact and a measured Sentinel-2 change about 47 km downstream at Bidur. The question is not whether AI foretold the disaster. It is whether OlmoEarth can place different sensors and locations in one representation space—and help select which physical explanations agree with observation.'}</p>
            <div className="hero-answer"><span>{ko ? '현재 답' : 'CURRENT ANSWER'}</span><strong>{ko ? '상류–하류 관측 사슬은 성립. 네팔 live 임베딩 판정은 S1 대기.' : 'OBSERVATION CHAIN CLOSED · NEPAL LIVE EMBEDDING WAITS FOR S1'}</strong></div>
          </section>

          <section className="story-section story-step story-wide">
            <p className="story-kicker">01 · {ko ? '사건 구조' : 'EVENT ANATOMY'} — <em>{ko ? '문자가 아니라 역할' : 'roles, not letters'}</em></p>
            <h2>{ko ? '붕괴 원점과 충격 지점은 다르다' : 'The collapse source is not the impact window'}</h2>
            <div className="event-chain-cards">{eventPoints.map((point) => <article key={point.id} style={{ '--point-color': point.marker_color } as CSSProperties}><b>{point.stage}</b><span>{point.display_label}</span><strong>{point.name}</strong><small>{point.id} · {point.distance_from_a_km.toFixed(1)} km from impact A</small><p>{ko ? point.story_ko : point.story}</p></article>)}</div>
            <div className="control-explainer"><b>Ø · C · NEGATIVE CONTROL</b><p>{ko ? controlPoints[0]?.story_ko : controlPoints[0]?.story}</p></div>
            <p className="story-caption">{ko ? 'E(빨강)는 공개 자료 기반 발원 수색점이며 정확한 방출 폴리곤은 아님. D(보라)는 위치 미공개 언색호의 잠정 탐색점. A(주황)와 B(노랑)는 국경 충격/노출 창. F(파랑)는 하류 관측 창. C(회색)는 사건 경로 밖.' : 'E (red) is a public-evidence source-search estimate, not a surveyed release polygon. D (purple) is a provisional lake search zone. A (orange) and B (yellow) are border impact/exposure windows. F (blue) is downstream observation. C (grey) is outside the event chain.'}</p>
          </section>

          <section className="story-section story-step story-wide">
            <p className="story-kicker">02 · {ko ? '위성 증거' : 'SATELLITE EVIDENCE'} — <em>{ko ? '시간과 거리를 함께 보기' : 'time × distance'}</em></p>
            <h2>{ko ? '국경에서 보인 변화가 하류에서도 보인다' : 'A border change now has a downstream counterpart'}</h2>
            <div className="evidence-pairs">
              <article><header><span>A · IMPACT</span><strong>Rasuwagadhi</strong></header>{sceneById('s2-2026-08-12') && sceneById('s2-2026-08-27') && <div className="story-swipe compact" style={{ ['--swipe' as string]: `${swipe}%` }}><img src={sceneById('s2-2026-08-27')!.image} alt="Rasuwagadhi Sentinel-2 post-event" /><div className="swipe-clip"><img src={sceneById('s2-2026-08-12')!.image} alt="Rasuwagadhi Sentinel-2 pre-event" /></div><div className="swipe-bar" /><span className="swipe-label pre">08-12</span><span className="swipe-label post">08-27</span><input type="range" min={0} max={100} value={swipe} aria-label="Compare Rasuwagadhi before and after" onChange={(e) => setSwipe(Number(e.target.value))} /></div>}<p>{ko ? '현재 OLMo 5-anchor 시계열의 중심 창.' : 'The centre of the current five-anchor Olmo time series.'}</p></article>
              <article><header><span>F · DOWNSTREAM</span><strong>Bidur / Trishuli</strong></header><div className="fixed-pair">{bidurPre && <figure><img src={bidurPre.image} alt="Bidur Sentinel-2 before event" /><figcaption>PRE · 08-12</figcaption></figure>}{bidurPost && <figure><img src={bidurPost.image} alt="Bidur Sentinel-2 after event" /><figcaption>POST · 08-27</figcaption></figure>}</div><p>{ko ? '기존 Rasuwagadhi 타일 밖, 인접 45RUL에서 새로 회수한 실제 2.56 km 창.' : 'A real 2.56 km pair recovered from adjacent MGRS tile 45RUL, missed by the original Rasuwagadhi-only catalog.'}</p></article>
            </div>
            <div className="distance-matrix">
              {['source', 'rasuwagadhi', 'timure', 'syabrubesi', 'dhunche', 'bidur'].map((name, i) => <div key={name}><b>{i + 1}</b><span>{name === 'source' ? 'SOURCE · Langtang Lirung' : name.toUpperCase()}</span><figure><img src={`/data/story/anchors/${name}_pre.png`} alt={`${name} before`} /></figure><i>→</i><figure><img src={`/data/story/anchors/${name}_post.png`} alt={`${name} after`} /></figure></div>)}
            </div>
            <p className="story-caption">{ko ? '모든 행은 8/12→8/27 Sentinel-2. 장면 차이는 후보 관측이며 피해 라벨이 아님. Source와 Dhunche는 구름/눈 제약이 큼. 8/24 보라색 프레임은 실제 색이 아니라 Sentinel-1 VV/VH/대비 false-colour.' : 'Every row is Sentinel-2 from 12→27 Aug. Differences are candidate observations, not damage labels. Source and Dhunche remain cloud/snow limited. The purple 24 Aug frame is Sentinel-1 VV/VH/contrast false colour—not surface colour.'}</p>
          </section>

          <section className="story-section story-step story-wide olmo-proof">
            <p className="story-kicker">03 · OLMoEarth — <em>{ko ? '이미 되는 것부터' : 'lead with what already works'}</em></p>
            <h2>{ko ? 'OLMo의 가치는 HOLD 하나가 아니다' : 'OlmoEarth is more than one blocked live delta'}</h2>
            <div className="olmo-proof-grid">
              <article className="ready"><b>01</b><span>{ko ? '네팔 기준 표현' : 'NEPAL BASELINE'}</span><strong>5 × 768-d READY</strong><p>{ko ? '5개 앵커의 S1+S2×4기간 baseline/placebo 임베딩 봉인. 지금도 유사지역 검색·선형 probe·사전/사후 비교 기준으로 사용 가능.' : 'Sealed S1+S2×4-period baseline/placebo embeddings for five anchors—usable now for retrieval, linear probes and a pre/post reference.'}</p></article>
              <article className="win"><b>02</b><span>{ko ? '확증 전이' : 'CONFIRMATORY TRANSFER'}</span><strong>{transfer ? `${transfer.wins_reuse_vs_raw_strong}/${transfer.regions} WINS · +${transfer.absolute_gap.toFixed(3)}` : 'LOADING'}</strong><p>{transfer ? `Frozen reuse region-macro ${transfer.reuse_region_macro.toFixed(3)} vs raw UNet3D ${transfer.raw_strong_region_macro.toFixed(3)} (${transfer.relative_gain_pct.toFixed(1)}% relative).` : ''}</p></article>
              <article className="pilot"><b>03</b><span>{ko ? '사건 변화 파일럿' : 'EVENT-DELTA PILOT'}</span><strong>2 / 3 STRONG</strong><p>{ko ? '관련 S2-only M66에서 Hokkaido·Hiroshima 분리, Dominica 약함. 가능성 증거이지 네팔 검증값은 아님.' : 'Related S2-only M66 separated Hokkaido and Hiroshima; Dominica was weak. Feasibility evidence, not Nepal validation.'}</p></article>
              <article className="wait"><b>04</b><span>{ko ? '네팔 live 변화' : 'NEPAL LIVE CHANGE'}</span><strong>S1 3/4 · S2 4/4</strong><p>{ko ? '마지막 S1 기간 전에는 Δz를 만들지 않음. 모델 전체 실패가 아니라 한 live action의 입력 대기.' : 'No Δz before the final S1 period. This is one live action waiting for input—not failure of the representation.'}</p></article>
            </div>
            <div className="transfer-bars"><span>Frozen OLMo reuse</span><i style={{ width: '100%' }} /><b>{transfer?.reuse_region_macro.toFixed(3) ?? '—'}</b><span>Raw UNet3D</span><i style={{ width: transfer ? `${100 * transfer.raw_strong_region_macro / transfer.reuse_region_macro}%` : '0%' }} /><b>{transfer?.raw_strong_region_macro.toFixed(3) ?? '—'}</b></div>
            <p>{ko ? '왜 유의미한가: EO 모델이 EO task에서 좋은 것은 당연하지 않다. 같은 공개 지역·같은 decoder 조건에서 frozen 표현 재사용이 강한 raw 시계열 모델보다 8개 외부 지역 중 6곳에서 이겼고, 동시에 Indonesia/Itogon의 실패도 남겼다. 즉 “항상 좋다”가 아니라 어디까지 전이되는지를 계량했다. 단, 이것은 OLMo 재사용 대 raw baseline 결과이며, 동일 입력계약의 두 번째 GeoFM(Presto) 대조 전에는 OLMo만의 고유 우월성으로 쓰지 않는다.' : 'Why this matters: an EO model is not automatically better on every EO task. Under matched public regions and decoder contracts, frozen representation reuse beat a strong raw time-series model in six of eight external regions, while preserving the Indonesia and Itogon non-wins. It measures where transfer holds rather than claiming universal superiority. This is Olmo reuse versus raw baselines—not Olmo-specific superiority until a second GeoFM control such as Presto is run under the same input contract.'}</p>
          </section>

          <section className="story-section story-step story-wide">
            <p className="story-kicker">04 · {ko ? '결합 실험' : 'THE FUSION EXPERIMENT'} — <em>{ko ? '그림이 아니라 닫힌 검증 루프' : 'a falsifiable loop, not decoration'}</em></p>
            <h2>{ko ? 'OLMo가 후보를 만들고, 물리가 설명하고, 위성이 반증한다' : 'Olmo proposes. Physics explains. Satellites falsify.'}</h2>
            <div className="fusion-loop">
              <article><b>1</b><strong>SENTINEL + DEM</strong><span>{ko ? '실제 픽셀·지형·궤도 footprint' : 'pixels, terrain, orbit footprint'}</span></article><i>→</i>
              <article><b>2</b><strong>OLMoEarth</strong><span>{ko ? '발원·변화·유사사례 후보' : 'source, change, analogue proposals'}</span></article><i>→</i>
              <article><b>3</b><strong>r.avaflow ENSEMBLE</strong><span>{ko ? '부피·마찰·수분 범위별 runout' : 'runout across volume/friction/water ranges'}</span></article><i>→</i>
              <article><b>4</b><strong>SENSOR OPERATOR</strong><span>{ko ? '각 runout이 S1/S2에서 보여야 할 mask' : 'what each runout should look like to S1/S2'}</span></article><i>→</i>
              <article><b>5</b><strong>OBSERVED Δ / MASK</strong><span>{ko ? 'Bidur·Rasuwagadhi 실측과 일치도' : 'agreement with Bidur/Rasuwagadhi observation'}</span></article><i>↺</i>
              <article><b>6</b><strong>RANK + REVIEW</strong><span>{ko ? '앙상블 재순위·D-Claw 독립 확인' : 're-rank ensemble; D-Claw independent check'}</span></article>
            </div>
            <p>{ko ? '첫 버전의 “위성 시뮬레이션”은 포토리얼한 가짜 사진이 아니다. 물/토석/노출지 mask와 관측 가능 footprint를 만드는 observation operator가 더 검증 가능하다. 그 다음에만 학습된 surrogate로 r.avaflow 앙상블을 웹에서 빠르게 재생한다. OLMo 임베딩 값을 마찰계수로 바꾸지 않으며, 물리 파라미터는 범위로 샘플링한다.' : 'The first satellite simulation should not be a photorealistic fake image. A semantic observation operator—water/debris/exposed-ground masks plus sensor visibility—is more testable. Only then should a learned surrogate replay the r.avaflow ensemble interactively. Embedding values never become friction coefficients; physical parameters are sampled as ranges.'}</p>
            <div className="story-test-metrics"><span><b>CHANGE</b>AUPRC · false changed area</span><span><b>PHYSICS</b>runout IoU · max-runout error</span><span><b>UNCERTAINTY</b>interval coverage · rank calibration</span><span><b>OPS</b>minutes · invalid actions @ recall</span></div>
          </section>

          <section className="story-section story-step story-wide priority-section">
            <p className="story-kicker">05 · {ko ? 'AI 엔지니어 우선순위' : 'ENGINEERING PRIORITIES'} — <em>{ko ? '영향으로 이어지는 경로' : 'the path to impact'}</em></p>
            <h2>{ko ? '지금 가장 가치 있는 네 가지 빌드' : 'The four builds that matter next'}</h2>
            <div className="priority-stack">
              <article><b>P0 · NOW</b><strong>{ko ? '관측 사슬과 라벨 확보' : 'Close evidence + labels'}</strong><p>{ko ? 'Bidur/Rasuwagadhi 전후 mask를 CEMS·Charter·수동 판독으로 동결. S1 8/31 footprint 통과 시 Nepal live cube 봉인.' : 'Freeze Bidur/Rasuwagadhi pre/post masks with CEMS, Charter and blinded manual review. Seal Nepal live cube only if the 31 Aug S1 footprint passes.'}</p><em>VALUE · ground truth, reproducibility</em></article>
              <article><b>P1 · 1 WEEK</b><strong>{ko ? 'OLMo change + retrieval 본실험' : 'Olmo change + retrieval experiment'}</strong><p>{ko ? '고전 NDWI/SAR 변화탐지, OLMo Δz, gate-aware abstention을 동일 recall에서 비교. Nepal query로 SEN12 6,834 patch 유사사건 검색.' : 'Compare classical NDWI/SAR change, Olmo Δz and gate-aware abstention at matched recall; query 6,834 SEN12 patches with the Nepal representation.'}</p><em>VALUE · AI2 relevance, triage speed</em></article>
              <article><b>P2 · 2–3 WEEKS</b><strong>{ko ? '물리–관측 앙상블' : 'Physics–observation ensemble'}</strong><p>{ko ? 'Copernicus DEM/GLO-30에서 r.avaflow 파라미터 sweep, D-Claw 소수 독립 run, S1/S2 semantic operator로 실측 일치도 순위화.' : 'Sweep r.avaflow over Copernicus DEM/GLO-30, run a small independent D-Claw check and rank outputs through S1/S2 semantic observation operators.'}</p><em>VALUE · causal plausibility, uncertainty</em></article>
              <article><b>P3 · 4–6 WEEKS</b><strong>{ko ? '빠른 surrogate + 운영 UI' : 'Fast surrogate + operations UI'}</strong><p>{ko ? '물리 앙상블로 neural operator/emulator를 학습해 웹에서 scenario를 재생. 실제 OSM 경로·위성 pass·OLMo evidence를 EarthRanger식 incident ledger와 연결.' : 'Train a neural operator/emulator on the physics ensemble for interactive scenarios; join OSM routes, satellite passes and Olmo evidence into an EarthRanger-style incident ledger.'}</p><em>VALUE · scalable decision support</em></article>
            </div>
          </section>

          <section className="story-section story-step story-boundary">
            <p className="story-kicker">06 · {ko ? '다음 게이트와 출처' : 'NEXT GATE + SOURCES'} — <em>{ko ? '멈춤도 결과, 진전도 결과' : 'progress with boundaries'}</em></p>
            <h2>{ko ? '8월 31일 S1이 다음 live 판정을 연다' : 'The 31 August S1 pass opens the next live decision'}</h2>
            <p>{ko ? '8월 28일 S1은 게시 지연이 아니라 AOI를 빗나갔다. 다음 후보도 예정표가 아니라 실제 footprint containment로 통과시킨다. 그 전에도 baseline embedding, 8-region transfer, Bidur 전후관측, physics experiment graph는 이미 유효한 산출물이다.' : 'The 28 Aug S1 pass missed the AOI; it was not a publication delay. The next candidate must again pass actual footprint containment. Until then, the baseline embeddings, eight-region transfer result, Bidur before/after observation and physics experiment graph remain valid outputs.'}</p>
            <div className="story-schedule">{(scenario?.scheduled_scenes ?? []).map((scene) => <div key={scene.id ?? scene.acquired_at} className={scene.state === 'missed_coverage' ? 'missed' : ''}><b>{shortSensor(scene.sensor)}</b><span>{kstStamp(scene.acquired_at)} KST</span><em>{scene.state.replace(/_/g, ' ').toUpperCase()}</em></div>)}</div>
            <div className="story-sources"><a href="https://www.usgs.gov/programs/landslide-hazards/science/2026-nepal-debris-avalanche-and-flash-flood" target="_blank" rel="noreferrer">USGS event assessment ↗</a><a href="https://allenai.org/blog/olmoearth-embeddings" target="_blank" rel="noreferrer">Ai2 embedding workflow ↗</a><a href="https://doi.org/10.5194/gmd-18-9879-2025" target="_blank" rel="noreferrer">r.avaflow v4 ↗</a><a href="https://claw.code-pages.usgs.gov/dclaw/" target="_blank" rel="noreferrer">USGS D-Claw ↗</a><a href="https://planetarycomputer.microsoft.com/docs/quickstarts/using-the-data-api/" target="_blank" rel="noreferrer">Planetary Computer STAC ↗</a><a href="https://mapping.emergency.copernicus.eu/activations/EMSR927/" target="_blank" rel="noreferrer">CEMS EMSR927 ↗</a></div>
            <p className="story-outro">{scenario?.research.integration_disclaimer}</p>
          </section>
        </div>
        );
      })()}

      <div className="provenance-stamp">DATA SNAPSHOT {scenario?.generated_at.slice(0, 16).replace('T', ' ') ?? '—'} UTC · OSM ODbL · ESA COPERNICUS</div>
    </main>
  );
}
