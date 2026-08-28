'use client';

import { AttributionControl, LngLatBounds, Map as MapLibreMap, NavigationControl, Popup, setWorkerUrl } from 'maplibre-gl';
import type { MapLayerMouseEvent } from 'maplibre-gl';
import type { Feature, FeatureCollection } from 'geojson';
import Image from 'next/image';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
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
  name: string;
  coordinates: [number, number];
  role: string;
  place: string;
  source?: string;
  source_url?: string;
  evidence_level?: string;
  story?: string;
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
  const initialSceneFitRef = useRef(false);
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
  const [selectedPoint, setSelectedPoint] = useState('A');
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
      type: 'Feature', properties: { id: p.id, name: p.name },
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
        center: [85.3779, 28.276],
        zoom: 14.15,
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
          'circle-radius': ['case', ['==', ['get', 'id'], 'A'], 18, 12],
          'circle-color': ['case', ['==', ['get', 'id'], 'C'], '#ffb45f', '#5fffd7'],
          'circle-opacity': 0.16, 'circle-stroke-width': 1,
          'circle-stroke-color': ['case', ['==', ['get', 'id'], 'C'], '#ffb45f', '#5fffd7'],
        },
      });
      map.addLayer({
        id: 'point-core', type: 'circle', source: 'research-points',
        paint: {
          'circle-radius': ['case', ['==', ['get', 'id'], 'A'], 6, 4],
          'circle-color': ['case', ['==', ['get', 'id'], 'C'], '#ffb45f', '#5fffd7'],
          'circle-stroke-width': 2, 'circle-stroke-color': '#081411',
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
        const win = ({ A: 'rasuwagadhi', B: 'rasuwagadhi', D: 'source', E: 'source' } as Record<string, string>)[pt.id];
        const thumbs = win
          ? `<div class="pp-thumbs">`
            + `<figure><img src="/data/story/anchors/${win}_pre.png" alt="pre"/><figcaption>PRE 08-12</figcaption></figure>`
            + `<figure><img src="/data/story/anchors/${win}_post.png" alt="post"/><figcaption>POST 08-27</figcaption></figure>`
            + `</div>`
          : '';
        new Popup({ closeButton: true, maxWidth: '320px', className: 'story-popup' })
          .setLngLat(pt.coordinates)
          .setHTML(`<p class="pp-eyebrow">${pt.id} · ${pt.role.replace(/_/g, ' ').toUpperCase()}</p>`
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
    fitScene(scene, initialSceneFitRef.current ? 700 : 0);
    initialSceneFitRef.current = true;
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
    return () => { cancelled = true; cancelAnimationFrame(animationFrame); };
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

  const focusPoint = (id: string) => {
    setSelectedPoint(id);
    const card = points.find((item) => item.id === id);
    if (!card) return;
    mapRef.current?.flyTo({
      center: card.coordinates, zoom: id === 'C' ? 10.5 : 14, pitch: viewDimRef.current === '3d' ? TERRAIN_PITCH : 0, bearing: viewDimRef.current === '3d' ? -18 : 0,
      duration: prefersReducedMotion() ? 0 : 1100,
    });
  };

  const fitCorridor = () => {
    mapRef.current?.fitBounds(new LngLatBounds([85.302, 28.135], [85.386, 28.288]), {
      padding: scenePadding(), pitch: viewDimRef.current === '3d' ? TERRAIN_PITCH : 0, bearing: viewDimRef.current === '3d' ? -18 : 0, duration: prefersReducedMotion() ? 0 : 1100,
    });
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
          <button onClick={() => activeScene && fitScene(activeScene, 900)} disabled={!activeScene || mapStatus !== 'ready'}>SATELLITE FRAME</button>
          <button onClick={fitCorridor} disabled={mapStatus !== 'ready'}>RIVER CORRIDOR</button>
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
        <div className="panel-heading"><span>01</span><div><p>AREA OF INTEREST</p><strong>Coordinate audit</strong></div></div>
        <div className="coordinate-list">
          {points.map((point) => (
            <button key={point.id} className={selectedPoint === point.id ? 'coordinate active' : 'coordinate'} onClick={() => focusPoint(point.id)}>
              <span>{point.id}</span>
              <div>
                <strong>{point.name}</strong><small>{point.coordinates[1].toFixed(6)}, {point.coordinates[0].toFixed(6)}</small>
                {selectedPoint === point.id && point.story && <p className="point-story">{point.story}</p>}
              </div>
              <em>{point.role.split('_')[0].toUpperCase()}</em>
            </button>
          ))}
          {points.length === 0 && <p className="rail-empty">{dataStatus === 'loading' ? 'Loading points…' : 'No points in snapshot.'}</p>}
        </div>
        {selectedCard && (
          <div className="selected-place">
            <span>{selectedCard.distance_from_a_km.toFixed(2)} km FROM A</span>
            <strong>{selectedCard.place}</strong>
          </div>
        )}
        <p className="audit-note"><b>C is 113.79 km away.</b> It is not used as the event flow endpoint; it remains a separate transfer/reference AOI.</p>
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
        <div className="panel-heading"><span>02</span><div><p>EVIDENCE LENS</p><strong>Before → after contract</strong></div></div>
        {decision && (
          <div className={`decision-card ${decision.status}`} role="status">
            <span>CURRENT DECISION</span>
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
          <div className={`pipeline-row ${dataStatus === 'ready' ? 'ready' : dataStatus === 'failed' ? 'pending' : 'preview'}`}><span>DS</span><div><strong>Snapshot data</strong><small>scenario · hydrography · anchors</small></div><b>{dataStatus.toUpperCase()}</b></div>
          <div className="pipeline-row ready"><span>S1</span><div><strong>Radar baseline</strong><small>{scenario ? `${scenario.scene_records.filter((s) => shortSensor(s.sensor) === 'S1').length} acquisitions · local GeoTIFF` : '—'}</small></div><b>READY</b></div>
          <div className="pipeline-row ready"><span>S2</span><div><strong>Optical baseline</strong><small>{scenario ? `${scenario.scene_records.filter((s) => shortSensor(s.sensor) === 'S2').length} acquisitions · true color from 12 bands` : '—'}</small></div><b>READY</b></div>
          <div className={`pipeline-row ${liveObservation?.olmo_ready ? 'ready' : 'pending'}`}><span>OE</span><div><strong>OLMoEarth contract</strong><small>{scenario ? `${scenario.olmoearth.anchors} anchors · ${livePeriodText}` : '—'}</small></div><b>{liveObservation?.olmo_ready ? 'SEALED' : 'HOLD'}</b></div>
          <div className="pipeline-row pending"><span>Δ</span><div><strong>Embedding delta</strong><small>{liveObservation?.olmo_ready ? 'sealed cube ready; embedding not run' : 'catalogued pixels ≠ sealed OLMo cube'}</small></div><b>{(typeof scenario?.olmoearth?.post_event_delta === 'object' && (scenario.olmoearth.post_event_delta as Record<string, unknown>).live_mode) ? 'READY' : liveObservation?.olmo_ready ? 'EMBED WAIT' : 'BLOCKED'}</b></div>
          <div className={`pipeline-row ${wasmStatus === 'ready' ? 'ready' : wasmStatus === 'failed' ? 'pending' : 'preview'}`}><span>W</span><div><strong>Flow layer</strong><small>Rust/WASM · {scenario?.simulation.route_points ?? '—'} route nodes</small></div><b>{wasmStatus.toUpperCase()}</b></div>
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
          <button onClick={() => setFlowPlaying((value) => !value)} aria-label={flowPlaying ? 'Pause flow animation' : 'Play flow animation'}>{flowPlaying ? 'PAUSE' : 'PLAY'}</button>
          <div>
            <label htmlFor="flow-speed"><span>ILLUSTRATIVE FLOW</span><b>{(flowSpeed / 0.034).toFixed(1)}×</b></label>
            <input id="flow-speed" type="range" min="0.012" max="0.09" step="0.002" value={flowSpeed} onChange={(event) => setFlowSpeed(Number(event.target.value))} />
          </div>
        </div>
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
              onClick={() => setActiveSceneId(scene.id)}
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
            <p className="story-dateline">RASUWA, NEPAL · 26 AUG 2026 · {ko ? '증거 갱신' : 'EVIDENCE UPDATED'} {scenario ? kstStamp(scenario.generated_at) : '—'} KST</p>
            <h1>{ko ? '한 사건, 여러 시계.' : 'One event. Many clocks.'}</h1>
            <p className="story-lede">{ko
              ? '8월 26일 네팔 영내 Langtang Lirung 북사면의 암반–빙하 붕괴가 Gyirong–Rasuwagadhi를 잇는 초국경 토석류·홍수로 이어졌음. 이후 생긴 언색호는 28일 배수 중이었지만 2차 위험 감시는 계속됨. 이 페이지는 재해를 예측했다고 주장하지 않음. 공개 관측, OLMoEarth 표현, 물리모델, 공식 확인이 서로 다른 속도로 도착할 때 무엇을 말할 수 있는지 보여줌.'
              : 'On 26 August, a rock–ice collapse on the Nepal side of Langtang Lirung generated a cross-border debris flow and flood through Gyirong–Rasuwagadhi. A debris-dammed lake formed afterward and was draining on the 28th, while secondary-hazard monitoring continued. This page does not claim it predicted the disaster. It shows what can be said as observations, OlmoEarth representations, physics and official review arrive on different clocks.'}</p>
          </section>

          <section className="story-section story-step story-now">
            <p className="story-kicker">01 · {ko ? '현재' : 'EVIDENCE NOW'} — <em>{ko ? '결론보다 상태' : 'state before conclusion'}</em></p>
            <h2>{ko ? '지금 계산 가능한 것은 어디까지인가' : 'What is computable right now'}</h2>
            <div className="story-status-grid">
              <div className="ok"><b>O</b><strong>{ko ? '관측' : 'OBSERVATION'}</strong><span>{ko ? '8/27 S2 픽셀 확보' : '27 Aug S2 pixels acquired'}</span></div>
              <div className="hold"><b>E</b><strong>{ko ? 'OLMo 증거' : 'OLMo EVIDENCE'}</strong><span>{ko ? 'S1 3/4 · S2 4/4, 보류' : 'S1 3/4 · S2 4/4, held'}</span></div>
              <div className="design"><b>P</b><strong>{ko ? '물리' : 'PHYSICS'}</strong><span>{ko ? '실험 설계, 아직 미실행' : 'designed, not yet run'}</span></div>
              <div className="external"><b>H</b><strong>{ko ? '공식 확인' : 'HUMAN / OFFICIAL'}</strong><span>{ko ? '외부 검증 자료로 동결' : 'frozen external evidence'}</span></div>
            </div>
            <p>{ko
              ? '현재 올바른 출력은 DO NOT EMBED임. 사건 후 광학은 있지만 OLMoEarth의 S1+S2×4기간 계약이 덜 찼고, 네팔 placebo도 2개뿐이라 Δz 임계값을 만들 수 없음.'
              : 'The correct output is currently DO NOT EMBED. Post-event optical pixels exist, but the S1+S2×4-period OlmoEarth contract is incomplete and two Nepal placebo windows are not enough to define a Δz anomaly threshold.'}</p>
          </section>

          <section className="story-section story-step">
            <p className="story-kicker">02 · {ko ? '시간축' : 'THE EVENT CLOCK'} — <em>{ko ? '27일에 새 산사태가 있었나' : 'was there a new slide on the 27th?'}</em></p>
            <h2>{ko ? '아니오. 같은 사건의 여파가 이어졌다.' : 'No. The aftermath of the same event continued.'}</h2>
            <div className="story-event-clock">
              {(scenario?.incident_updates ?? []).map((item) => (
                <article key={`${item.occurred_at_utc}-${item.status}`}>
                  <time>{item.occurred_at_utc.slice(8, 10)} AUG · {item.occurred_at_utc.slice(11, 16)}Z</time>
                  <div><strong>{item.title}</strong><p>{item.summary}</p><a href={item.source_url} target="_blank" rel="noreferrer">{item.source} ↗</a></div>
                </article>
              ))}
            </div>
          </section>

          <section className="story-section story-step">
            <p className="story-kicker">03 · {ko ? '회랑' : 'THE CORRIDOR'} — <em>{ko ? '어디를 볼 것인가' : 'where to look'}</em></p>
            <h2>{ko ? '한 개의 선이 아니라, 검증할 지점들의 순서' : 'Not a flood polygon—a sequence of places to inspect'}</h2>
            <div className="story-corridor">
              {corridorSketch && (
                <svg viewBox={`0 0 ${corridorSketch.W} ${corridorSketch.H}`} role="img" aria-label="Mapped drainage corridor">
                  <path d={corridorSketch.path} fill="none" stroke="var(--blue)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
                  {corridorSketch.dots.map((d, i) => (
                    <g key={d.name}><circle cx={d.x} cy={d.y} r="3.4" fill={i === 0 ? 'var(--orange)' : 'var(--surface)'} stroke={i === 0 ? 'var(--orange)' : 'var(--blue)'} strokeWidth="1.6" /><text x={d.x + 7} y={d.y + 3.5} fontSize="8.5" fontFamily="var(--font-geist-mono)" fill="var(--muted)">{d.name}</text></g>
                  ))}
                </svg>
              )}
            </div>
            <p>{ko
              ? '이 선은 OSM에 매핑된 배수 회랑임. 보도와 사건 후 영상이 이 회랑을 따라 영향을 놓지만, 선 자체가 홍수 경계·수심·도달시간을 뜻하지는 않음. E는 네팔 영내 Langtang Lirung 발원 수색점이고, D는 정확 좌표가 공개되지 않은 언색호의 설명용 표식임.'
              : 'The line is a mapped OSM drainage corridor along which reports and post-event imagery place the flood. It is not a flood boundary, depth map or travel-time estimate. E is the Nepal-side Langtang Lirung source-search anchor; D is an illustrative marker for a reported lake whose exact public coordinates remain unresolved.'}</p>
          </section>

          <section className="story-section story-step">
            <p className="story-kicker">04 · {ko ? '광학' : 'THE OPTICAL VIEW'} — <em>{ko ? '픽셀은 무엇을 보여주는가' : 'what the pixels show'}</em></p>
            <h2>{ko ? '15일 간격의 두 관측' : 'Two observations, fifteen days apart'}</h2>
            <p>{ko ? '두 장면 모두 Sentinel-2 L2A이며, 핸들을 움직여 같은 Rasuwagadhi 창을 비교할 수 있음.' : 'Both frames are Sentinel-2 L2A observations of the same Rasuwagadhi window. Drag the handle to compare them.'}</p>
            {sceneById('s2-2026-08-12') && sceneById('s2-2026-08-27') && (
              <div className="story-swipe" style={{ ['--swipe' as string]: `${swipe}%` }}>
                <img src={sceneById('s2-2026-08-27')!.image} alt="Sentinel-2 27 August post-event observation" />
                <div className="swipe-clip"><img src={sceneById('s2-2026-08-12')!.image} alt="Sentinel-2 12 August pre-event observation" /></div>
                <div className="swipe-bar" /><span className="swipe-label pre">PRE · 08-12</span><span className="swipe-label post">POST · 08-27</span>
                <input type="range" min={0} max={100} value={swipe} aria-label="Compare before and after" onChange={(e) => setSwipe(Number(e.target.value))} />
              </div>
            )}
            <p className="story-caption">{ko
              ? 'Sentinel-2 L2A · 10m · 2.56km 창 · 우측 장면은 8/27 04:56 UTC. 타일 전체 구름률은 78.47%지만 AOI 구름분류(SCL)는 확보하지 못했음. B02 밝기 휴리스틱은 2.5%였으나 눈·구름을 구분하지 못하므로 “맑음” 판정이 아님. 회색/갈색 수로 폭과 반사도 변화는 후보 변화이며 피해 라벨이 아님. © Copernicus Sentinel data 2026.'
              : 'Sentinel-2 L2A · 10 m · 2.56 km window · right frame 27 Aug 04:56 UTC. The tile is 78.47% cloudy, but no AOI SCL classification is available. A B02-bright heuristic measured 2.5%, yet cannot separate snow from cloud and is not a “clear” label. Apparent channel widening and altered reflectance are candidate changes, not damage labels. © Copernicus Sentinel data 2026.'}</p>
          </section>

          <section className="story-section story-step">
            <p className="story-kicker">05 · {ko ? '공백' : 'THE GAPS'} — <em>{ko ? '보이지 않는 곳도 결과다' : 'not seeing is also a result'}</em></p>
            <h2>{ko ? '같은 영상, 서로 다른 관측성' : 'One scene, uneven observability'}</h2>
            <figure className="story-figure"><img src="/data/story/corridor_post_grid.png" alt="Four corridor windows from the 27 August Sentinel-2 scene" /><figcaption className="story-caption">{ko ? 'Rasuwagadhi · Timure · Syabrubesi · Dhunche. 밝은 구름·눈이 앵커마다 다르게 나타나며 Dhunche 판독은 제한됨.' : 'Rasuwagadhi · Timure · Syabrubesi · Dhunche. Bright cloud/snow varies by anchor and limits interpretation at Dhunche.'}</figcaption></figure>
            <figure className="story-figure story-figure-pair"><img src="/data/story/source_pre_0812.png" alt="Langtang Lirung source-search window on 12 August" /><img src="/data/story/source_post_0827.png" alt="Cloud-obscured source-search window on 27 August" /><figcaption className="story-caption">{ko ? 'Langtang Lirung 발원 수색 창: 8/12와 8/27. 사건 후 광학은 구름·눈 때문에 방출흔을 독립 확인하지 못함.' : 'Langtang Lirung source-search window: 12 Aug and 27 Aug. Cloud/snow prevents independent optical confirmation of the release scar.'}</figcaption></figure>
            <p>{ko
              ? '8월 24일의 보라색 화면은 광학 사진이 아니라 Sentinel-1 VV/VH/비율을 RGB로 배치한 레이더 false-colour임. 색은 지표의 실제 색이 아니며, 후방산란 채널의 조합을 읽기 위한 표현임.'
              : 'The purple 24 August frame is not an optical photograph. It is a Sentinel-1 false-colour composite of VV, VH and their contrast; the colours encode radar backscatter channels, not surface colour.'}</p>
          </section>

          <section className="story-section story-step">
            <p className="story-kicker">06 · OLMoEarth — <em>{ko ? '표현과 예측을 구분하기' : 'representation is not prediction'}</em></p>
            <h2>{ko ? '장소가 자기 과거와 얼마나 달라졌는가' : 'How far a place moved from its own past'}</h2>
            <p>{ko
              ? 'frozen OLMoEarth v1은 2.56km 창의 S1·S2 시계열을 공간 패치별 768차원 표현으로 바꿈. 이 표현의 전후 거리 Δz와 유사사례 검색은 가능하지만, 원인·피해·유속을 직접 출력하지 않음.'
              : 'Frozen OlmoEarth v1 turns each 2.56 km S1/S2 time-series window into 768-dimensional spatial patch representations. Their temporal distance Δz and nearest neighbours can support change triage, but do not directly output cause, damage, velocity or depth.'}</p>
            <div className="story-diagram" aria-hidden="true"><div className="sd-box">S1 + S2<br /><small>4 × 14-day periods</small></div><div className="sd-arrow">→</div><div className="sd-box sd-vec">768-d<br /><small>frozen OLMoEarth v1</small></div><div className="sd-arrow">→</div><div className="sd-box sd-delta">Δz + neighbours<br /><small>vs local placebo</small></div></div>
            <div className="story-research-grid">
              <article><span>{ko ? '네팔 현재' : 'NEPAL NOW'}</span><strong>BLOCKED</strong><p>{scenario?.research.nepal_embedding.claim}</p></article>
              <article><span>{ko ? '과거 사건 파일럿' : 'HISTORICAL PILOT'}</span><strong>S2-ONLY</strong><p>{scenario?.research.historical_event_delta_pilot.claim_boundary}</p></article>
              <article><span>{ko ? '사전 위험예측' : 'PRE-EVENT RISK'}</span><strong>NOT DETECTED</strong><p>{scenario?.research.pre_event_susceptibility_probe.claim_boundary}</p></article>
            </div>
            <div className="story-metric-table">
              {(scenario?.research.historical_event_delta_pilot.rows ?? []).map((row) => <div key={row.region}><span>{row.region}</span><i style={{ width: `${Math.max(0, Math.min(100, row.auroc * 100))}%` }} /><b>{row.auroc.toFixed(3)}</b></div>)}
            </div>
            <p className="story-caption">{ko ? 'M66은 관련 S2-only pre4/post4 선행 실험임. 홋카이도·히로시마에서는 강했지만 도미니카는 약하고 placebo가 12패치뿐임. 네팔 S1+S2 계약의 검증값으로 사용하지 않음.' : 'M66 is a related S2-only pre4/post4 pilot. It was strong in Hokkaido and Hiroshima, weak in Dominica with only 12 placebo patches, and is not a Nepal S1+S2 validation result.'}</p>
          </section>

          <section className="story-section story-step">
            <p className="story-kicker">07 · {ko ? '물리 결합' : 'PHYSICS COUPLING'} — <em>{ko ? '표현에서 이동경로까지' : 'from evidence to runout'}</em></p>
            <h2>{ko ? 'OLMo가 제안하고, 물리가 이동시킨다' : 'OLMo proposes. Physics propagates.'}</h2>
            <div className="story-pipeline"><div><b>O</b><strong>Sentinel / DEM</strong><small>{ko ? '관측과 지형' : 'observation + terrain'}</small></div><i>→</i><div><b>E</b><strong>OLMoEarth</strong><small>{ko ? '발원·변화·유사사례 후보' : 'source/change/analogue candidates'}</small></div><i>→</i><div><b>P</b><strong>r.avaflow</strong><small>{ko ? '앙상블 runout' : 'ensemble runout'}</small></div><i>→</i><div><b>H</b><strong>Review</strong><small>{ko ? '공식·현장 검증' : 'official/field check'}</small></div></div>
            <p>{ko
              ? '현재 WASM 입자는 검증된 배수 중심선을 따라 움직이는 인터페이스 설명용 애니메이션임. 과학 계산은 상류 암반–빙하–토석–물 연쇄를 r.avaflow v4 앙상블로 풀고 D-Claw로 독립 확인한 뒤, 정의된 단면 수문곡선이 있을 때만 LISFLOOD-FP 또는 BASEMENT로 하류 수리단계를 잇는 구조가 적절함.'
              : 'The current WASM particles are an interface illustration along a mapped drainage centreline. A scientific upgrade would run an r.avaflow v4 ensemble for the upper rock–ice–debris–water cascade, check it independently with D-Claw, and couple to LISFLOOD-FP or BASEMENT downstream only after a defensible cross-section hydrograph exists.'}</p>
            <p className="story-rule">{scenario?.research.physics.coupling_rule}</p>
          </section>

          <section className="story-section story-step">
            <p className="story-kicker">08 · {ko ? '평가' : 'THE TEST'} — <em>{ko ? '멋진 그림을 연구로 바꾸기' : 'turning a demo into evidence'}</em></p>
            <h2>{ko ? '한 단계씩 무엇이 추가되는지 측정한다' : 'Measure what each layer actually adds'}</h2>
            <div className="story-arms">{(scenario?.research.evaluation_arms ?? []).map((arm) => <div key={arm.id}><b>{arm.id}</b><span>{arm.label}</span></div>)}</div>
            <p>{ko
              ? '주 비교는 A1 고전 변화탐지 대 A3 gate-aware OLMoEarth임. 같은 recall에서 잘못된 action과 분석시간이 줄어드는지가 운영 헤드라인이고, 변화 영역 AUPRC·발원 위치오차·runout IoU·최대도달 오차·불확실성 coverage를 함께 봄. CEMS·Charter·USGS 산출물은 입력이 아니라 untouched 외부 판정자료로 동결함.'
              : 'The primary comparison is classical change detection (A1) versus gate-aware OlmoEarth with abstention (A3). The operational headline is fewer invalid actions and analyst minutes at matched recall; event AUPRC, source-localisation error, runout IoU, maximum-runout error and interval coverage remain scientific metrics. CEMS, Charter and USGS products are frozen as untouched external adjudication—not model inputs.'}</p>
          </section>

          <section className="story-section story-step story-boundary">
            <p className="story-kicker">09 · {ko ? '주장 경계' : 'CLAIM BOUNDARY'} — <em>{ko ? '말하지 않는 것이 기능이다' : 'abstention is a feature'}</em></p>
            <h2>{ko ? '후보 변화까지만' : 'Candidate change—and no further'}</h2>
            <p>{ko
              ? '이 시스템은 현재 피해 확률, 원인 귀속, 붕괴 부피, 홍수 수심, 도달시간을 주장하지 않음. OLMo 임베딩을 마찰계수나 속도로 변환하지 않음. 입력계약·코드 스냅샷·placebo·외부 확인 중 하나라도 빠지면 결과 대신 보류를 남김.'
              : 'The system currently makes no claim about damage probability, causal attribution, release volume, flood depth or arrival time. It never converts embedding values into friction or velocity. If the input contract, code snapshot, placebo distribution or independent corroboration is missing, it records abstention instead of a result.'}</p>
          </section>

          <section className="story-section story-step">
            <p className="story-kicker">10 · {ko ? '다음 관측' : 'THE NEXT CLOCK'} — <em>{ko ? '놓친 패스도 증거다' : 'a missed pass is evidence too'}</em></p>
            <h2>{ko ? '8월 28일 레이더는 늦은 것이 아니라 빗나갔다' : 'The 28 August radar pass was not late. It missed.'}</h2>
            <p>{ko
              ? 'Copernicus에 인접 S1D 제품 두 개가 게시됐지만, 남쪽 제품의 북단은 위도 28.008°, 북쪽 제품의 남단은 29.113°였고 발원 AOI 28.277°는 둘 사이에 놓였음. 따라서 0개 footprint가 AOI를 포함함. 다음 레이더 후보는 8월 31일 09:07 KST이며, 예정표가 아니라 실제 footprint로 다시 통과시켜야 함.'
              : 'Copernicus published two nearby S1D products, but the southern footprint ended at 28.008°N and the northern footprint began at 29.113°N; the 28.277°N source AOI sat between them. Zero footprints contained the AOI. The next radar candidate is 31 August 09:07 KST and must again pass actual footprint containment—not a schedule assumption.'}</p>
            <div className="story-schedule">{(scenario?.scheduled_scenes ?? []).map((scene) => <div key={scene.id ?? scene.acquired_at} className={scene.state === 'missed_coverage' ? 'missed' : ''}><b>{shortSensor(scene.sensor)}</b><span>{kstStamp(scene.acquired_at)} KST</span><em>{scene.state.replace(/_/g, ' ').toUpperCase()}</em></div>)}</div>
          </section>

          <section className="story-section story-step">
            <p className="story-kicker">11 · {ko ? '아카이브' : 'THE LEDGER'} — <em>{ko ? '관측과 출처' : 'observations and sources'}</em></p>
            <h2>{ko ? '보여준 모든 픽셀과 판단의 계보' : 'A provenance trail for every pixel and decision'}</h2>
            <div className="story-archive">{(scenario?.scene_records ?? []).map((s) => <figure key={s.id}><img src={s.image} alt={`${s.sensor} ${s.acquired_at.slice(0, 10)}`} loading="lazy" /><figcaption>{shortSensor(s.sensor)} · {s.acquired_at.slice(5, 10)}</figcaption></figure>)}</div>
            <div className="story-sources"><a href="https://www.usgs.gov/programs/landslide-hazards/science/2026-nepal-debris-avalanche-and-flash-flood" target="_blank" rel="noreferrer">USGS assessment ↗</a><a href="https://www.icimod.org/press-release/major-flash-flood-sweeps-through-nepals-rasuwa-district-raising-fears-of-further-downstream-flooding/" target="_blank" rel="noreferrer">ICIMOD assessment ↗</a><a href="https://mapping.emergency.copernicus.eu/activations/EMSR927/" target="_blank" rel="noreferrer">Copernicus EMSR927 ↗</a><a href="https://disasterscharter.org/activations/flood-in-nepal-activation-1052-" target="_blank" rel="noreferrer">International Charter 1052 ↗</a></div>
            <p className="story-outro">{scenario?.research.integration_disclaimer}</p>
          </section>
        </div>
        );
      })()}

      <div className="provenance-stamp">DATA SNAPSHOT {scenario?.generated_at.slice(0, 16).replace('T', ' ') ?? '—'} UTC · OSM ODbL · ESA COPERNICUS</div>
    </main>
  );
}
