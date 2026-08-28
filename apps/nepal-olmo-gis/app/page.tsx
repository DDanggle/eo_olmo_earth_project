'use client';

import { AttributionControl, LngLatBounds, Map as MapLibreMap, NavigationControl, Popup, setWorkerUrl } from 'maplibre-gl';
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
  story?: string;
  distance_from_a_km: number;
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
  scheduled_scenes: { sensor: string; acquired_at: string; state: string }[];
  live_observation: LiveObservation | null;
  olmoearth: { input_contract: string; anchors: number; embedding_status: string; post_event_delta: string | Record<string, unknown> };
  decision: CurrentDecision;
  ops_log?: { event_id?: string; time_utc: string; source: string; type: string; priority: 'green' | 'orange' | 'blue'; summary: string }[];
  simulation: { route_points: number; claim: string };
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
  const [storyOpen, setStoryOpen] = useState(false);
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
      state: s.state === 'planned' ? 'PLANNED' : s.state === 'catalog_published_cloudy' ? 'CATALOG' : 'PENDING',
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
    if (!mapReady || !map || points.length === 0 || map.getSource('research-points')) return;
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
    map.on('click', 'point-core', (event) => {
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
            + `<p class="pp-src">${pt.source ?? ''}</p>`)
          .addTo(map);
      }
    });
    map.on('mouseenter', 'point-core', () => { map.getCanvas().style.cursor = 'pointer'; });
    map.on('mouseleave', 'point-core', () => { map.getCanvas().style.cursor = ''; });
  }, [mapReady, points, researchPoints]);

  useEffect(() => {
    const map = mapRef.current;
    if (!mapReady || !map || !hydrography || map.getSource('hydrography')) return;
    const before = map.getLayer('point-halo') ? 'point-halo' : undefined;
    map.addSource('hydrography', { type: 'geojson', data: hydrography as FeatureCollection });
    map.addLayer({ id: 'river-casing', type: 'line', source: 'hydrography', paint: { 'line-color': '#06100e', 'line-width': 8, 'line-opacity': 0.82 } }, before);
    map.addLayer({ id: 'river-route', type: 'line', source: 'hydrography', paint: { 'line-color': '#5fffd7', 'line-width': 2.2, 'line-opacity': 0.8, 'line-dasharray': [1.2, 1.6] } }, before);
  }, [hydrography, mapReady]);

  useEffect(() => {
    const map = mapRef.current;
    if (!mapReady || !map) return;
    fetch('/data/olmo-input-anchors.geojson').then((r) => r.json() as Promise<FeatureCollection>).then((anchors) => {
      if (map.getSource('olmo-anchors')) return;
      const before = map.getLayer('point-halo') ? 'point-halo' : undefined;
      map.addSource('olmo-anchors', { type: 'geojson', data: anchors });
      map.addLayer({ id: 'olmo-anchor-fill', type: 'fill', source: 'olmo-anchors', paint: { 'fill-color': '#5fffd7', 'fill-opacity': 0.045 } }, before);
      map.addLayer({ id: 'olmo-anchor-line', type: 'line', source: 'olmo-anchors', paint: { 'line-color': '#b7ffe9', 'line-width': 1, 'line-opacity': 0.52, 'line-dasharray': [3, 2] } }, before);
    }).catch(() => undefined);
  }, [mapReady]);

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
    if (!mapReady || !map || !scenario || !activeSceneId) return;
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
  }, [activeSceneId, mapReady, scenario, fitScene]);

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
  const nextScheduled = scenario?.scheduled_scenes[0] ?? null;
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
  // #story 딥링크 — 공유 시 스토리부터 열림.
  useEffect(() => { if (window.location.hash === '#story') setStoryOpen(true); }, []);
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
            <strong>S2B 27 AUG · {liveObservation.catalog_status.toUpperCase()}</strong>
            <small>{kstStamp(liveObservation.publication_utc)} KST · TILE CLOUD {liveObservation.cloud_cover_tile_pct?.toFixed(2) ?? '—'}%</small>
            <em>{liveReadinessLabel} · {livePeriodText}</em>
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
          <div className="layer-contract-row off"><b>P</b><span>Physics — r.avaflow runout · SFINCS envelope</span><em>NOT YET</em></div>
          <div className="layer-contract-row off"><b>H</b><span>Human/official — Charter · ICIMOD polygons</span><em>NOT YET</em></div>
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
            <p className="story-dateline">RASUWA, NEPAL · 26 AUG 2026 · {ko ? '조사 진행 중' : 'INVESTIGATION ONGOING'}</p>
            <h1>{ko ? '계곡이 자기 모습이기를 멈췄다.' : 'The valley stopped looking like itself.'}</h1>
            <p className="story-lede">{ko
              ? '2026년 8월 26일 새벽, 네팔–중국 국경의 Bhote Koshi 계곡을 돌발 홍수가 휩쓸었음. 원인은 rock–ice avalanche로 추정되며 조사 중임. 급류는 첫 22km를 시속 약 193km로 내려갔음. 이 페이지는 공개 데이터 파이프라인이 그 계곡을 어떻게 지켜보는지를 설명함 — 강이 말해주는 것, 위성이 보는 것, 그리고 frozen EO 모델이 말할 수 있는 것과 말하기를 거부하는 것.'
              : 'Before dawn on 26 August, a flash flood tore down the Bhote Koshi valley on the Nepal–China border — a suspected rock–ice avalanche, still under investigation. The surge covered its first 22 kilometres at roughly 193 km/h. This page explains how an open-data pipeline watches that valley: what the river tells us, what the satellites see, and what a frozen Earth-observation model can — and refuses to — say.'}</p>
          </section>

          <section className="story-section story-step">
            <p className="story-kicker">01 · {ko ? '강' : 'THE RIVER'} — <em>{ko ? '어디를 볼 것인가' : 'where to look'}</em></p>
            <h2>{ko ? '물이 그린 하나의 회랑' : 'One corridor, drawn by water'}</h2>
            <div className="story-corridor">
              {corridorSketch && (
                <svg viewBox={`0 0 ${corridorSketch.W} ${corridorSketch.H}`} role="img" aria-label="River corridor">
                  <path d={corridorSketch.path} fill="none" stroke="var(--blue)" strokeWidth="1.8"
                        strokeLinecap="round" strokeLinejoin="round" />
                  {corridorSketch.dots.map((d, i) => (
                    <g key={d.name}>
                      <circle cx={d.x} cy={d.y} r="3.4" fill={i === 0 ? 'var(--orange)' : 'var(--surface)'}
                              stroke={i === 0 ? 'var(--orange)' : 'var(--blue)'} strokeWidth="1.6" />
                      <text x={d.x + 7} y={d.y + 3.5} fontSize="8.5" fontFamily="var(--font-geist-mono)" fill="var(--muted)">{d.name}</text>
                    </g>
                  ))}
                </svg>
              )}
            </div>
            <p>{ko
              ? '수계 레이어는 장식이 아님 — 급류가 실제로 사용한 OSM 강 중심선임. Rasuwagadhi 상류는 전부 티베트의 Lhende 유역이고, 그곳에 이탈 추정 발원(E, 약 5,200m)과 8월 27일 생긴 언색호(D, 약 0.11km²)가 있음 — 중국 당국 발표로는 28일에 붕괴해 점진 배수됐고, 계곡은 2차 급류를 가까스로 피했음. 강은 재해 파이프라인의 첫 질문에 답함: 어디를 볼 것인가.'
              : 'The hydrography layer is not decoration: it is the OSM river centreline the flood actually used. Everything upstream of Rasuwagadhi is the Lhende basin in Tibet — the suspected detachment source (E, ~5,200 m) and, a barrier lake (D, ~0.11 km²) that formed on 27 August and, per Chinese authorities, breached and drained gradually on the 28th — a second surge the valley narrowly avoided. The river answers the first question of any disaster pipeline: '}<strong>{ko ? '' : 'where to look.'}</strong></p>
          </section>

          <section className="story-section story-step">
            <p className="story-kicker">02 · {ko ? '위성' : 'THE SATELLITES'} — <em>{ko ? '무엇이 보이는가' : 'what is seen'}</em></p>
            <h2>{ko ? '15일 간격의 두 장' : 'Two pictures, fifteen days apart'}</h2>
            <p>{ko ? '아래는 사건 전 마지막으로 구름이 걷힌 Sentinel-2 관측과 사건 후 첫 관측임 — 핸들을 끌어 비교.' : 'Below are the last cloud-usable Sentinel-2 view before the event and the first one after it — drag the handle to compare.'}</p>
            {sceneById('s2-2026-08-12') && sceneById('s2-2026-08-27') && (
              <div className="story-swipe" style={{ ['--swipe' as string]: `${swipe}%` }}>
                <img src={sceneById('s2-2026-08-27')!.image} alt="Sentinel-2 2026-08-27, one day after the event" />
                <div className="swipe-clip">
                  <img src={sceneById('s2-2026-08-12')!.image} alt="Sentinel-2 2026-08-12, before the event" />
                </div>
                <div className="swipe-bar" />
                <span className="swipe-label pre">PRE · 08-12</span>
                <span className="swipe-label post">POST · 08-27</span>
                <input type="range" min={0} max={100} value={swipe} aria-label="Compare before and after"
                       onChange={(e) => setSwipe(Number(e.target.value))} />
              </div>
            )}
            <p className="story-caption">{ko
              ? 'Sentinel-2 L2A 트루컬러 · 10m · 2.56km 창(Rasuwagadhi 앵커) · 좌 2026-08-12 (S2C) · 우 2026-08-27 04:56 UTC (S2B, 사건 +1일; 타일 구름 78% 중 이 구역은 맑음). 계곡 바닥을 따라 회색 debris 폭이 넓어짐. © Copernicus Sentinel data 2026.'
              : 'Sentinel-2 L2A true color · 10 m · 2.56 km window (Rasuwagadhi anchor) · left 2026-08-12 (S2C) · right 2026-08-27 04:56 UTC (S2B, event +1 day; this window clear inside a 78%-cloud tile). Grey debris widens along the valley floor. © Copernicus Sentinel data 2026.'}</p>
          </section>

          <section className="story-section story-step">
            <p className="story-kicker">03 · {ko ? '흐름의 추적' : 'THE FLOW TRACE'} — <em>{ko ? '급류는 어디로 갔는가' : 'where the surge went'}</em></p>
            <h2>{ko ? '국경에서 하류까지, 네 개의 창' : 'Four windows, border to downstream'}</h2>
            <p>{ko
              ? '같은 8월 27일 장면을 회랑의 네 앵커에서 잘라 보면 급류의 경로가 그대로 읽힘. Rasuwagadhi(국경)에서는 두 물줄기가 만나는 Y자 합류부 전체가 넓은 회색 debris 판이 됐고, Timure는 구름 사이로 갈색으로 변한 물길이 보이며, Syabrubesi에서는 하폭이 크게 넓어져 강변 구조물 일부가 사라졌음. Dhunche는 구름에 덮여 판독 불가 — 그 정직한 공백은 오늘 밤 레이더가 채움.'
              : 'Cut the same 27 August scene at four corridor anchors and the surge’s path reads directly. At Rasuwagadhi (border) the whole Y-junction where two channels meet became one broad grey debris sheet; at Timure a browned channel shows through cloud gaps; at Syabrubesi the channel widened dramatically and riverside structures are gone. Dhunche is cloud-blocked — an honest gap that tonight’s radar fills.'}</p>
            <figure className="story-figure">
              <img src="/data/story/corridor_post_grid.png" alt="Four anchor windows on 27 August showing the debris corridor" />
              <figcaption className="story-caption">{ko
                ? '2026-08-27 Sentinel-2 · 네 앵커 창 (각 2.56km). 좌상 Rasuwagadhi → 우상 Timure → 좌하 Syabrubesi → 우하 Dhunche(구름). © Copernicus Sentinel data 2026.'
                : 'Sentinel-2, 27 Aug 2026 · four anchor windows (2.56 km each). Top-left Rasuwagadhi → top-right Timure → bottom-left Syabrubesi → bottom-right Dhunche (cloud). © Copernicus Sentinel data 2026.'}</figcaption>
            </figure>
            <p>{ko
              ? '발원 추정 지점(E)의 8월 27일 광학은 구름에 막혔음. 즉 광학만으로는 빙하 이탈 흔적을 아직 확인할 수 없음 — 이것이 레이더(Sentinel-1)가 이 파이프라인의 절반인 이유임.'
              : 'The optical view of the suspected source (E) on 27 August is blocked by cloud — so the detachment scar cannot yet be confirmed optically. This is exactly why radar (Sentinel-1) is half of this pipeline.'}</p>
            <figure className="story-figure story-figure-pair">
              <img src="/data/story/source_pre_0812.png" alt="Source window before, 12 August" />
              <img src="/data/story/source_post_0827.png" alt="Source window after, 27 August — cloud covered" />
              <figcaption className="story-caption">{ko
                ? '발원 수색 창(E, 2.56km) · 좌 08-12 맑음 — 빙하·암릉이 보임 · 우 08-27 구름 — 판독 불가로 기록함.'
                : 'Source search window (E, 2.56 km) · left 08-12, clear — glacier and ridgelines visible · right 08-27, cloud — recorded as unreadable.'}</figcaption>
            </figure>
          </section>

          <section className="story-section story-step">
            <p className="story-kicker">04 · {ko ? '모델' : 'THE MODEL'} — <em>{ko ? '장소가 자기 과거처럼 보이기를 멈췄는가' : 'whether the place stopped looking like its own past'}</em></p>
            <h2>{ko ? '기계가 보는 것' : 'What a machine sees'}</h2>
            <p>{ko
              ? 'frozen OlmoEarth v1 모델이 8주치 레이더(Sentinel-1)와 광학(Sentinel-2) 이력을 읽어 40m 패치 하나를 768개의 숫자로 압축함 — 그 장소가 어떻게 보이고 어떻게 행동하는지의 상태 서명임. 라벨도 없고, 이 사건으로 학습하지도 않았음.'
              : 'A frozen OlmoEarth v1 model reads eight weeks of radar (Sentinel-1) and optical (Sentinel-2) history and compresses every 40 m patch into 768 numbers — a state signature of how that place looks and behaves. No labels, no training on this event.'}</p>
            <div className="story-diagram" aria-hidden="true">
              <div className="sd-box">{ko ? '40m 패치' : '40 m patch'}<br /><small>{ko ? '8주 · S1+S2' : '8 weeks · S1+S2'}</small></div>
              <div className="sd-arrow">→</div>
              <div className="sd-box sd-vec">{ko ? '768차원 서명' : '768-d signature'}<br /><small>frozen OlmoEarth v1</small></div>
              <div className="sd-arrow">→</div>
              <div className="sd-box sd-delta">{ko ? '자기 과거와의 Δz' : 'Δz vs its own past'}<br /><small>{ko ? 'placebo 주간과 대조 판정' : 'judged against placebo weeks'}</small></div>
            </div>
            <p>{ko
              ? '변화는 사건 없는 평범한 주간(placebo 창)이 만들어내는 Δz를 초과할 때만 선언됨. 같은 프로토콜을 산사태 지도가 있는 과거 재해 세 곳 — 2018 홋카이도, 2018 히로시마, 2017 도미니카 — 에 재생하면 AUROC 0.85 / 0.95 / 0.61 로 피해를 지역화했음. 네팔의 앵커 하나가 일화가 아닌 이유임: 레시피가 이동함.'
              : 'Change is declared only when the pre/post distance Δz exceeds what ordinary, uneventful weeks produce (the placebo windows). The same protocol, replayed on three past disasters with mapped landslides — Hokkaido 2018, Hiroshima 2018, Dominica 2017 — localized damage with AUROC 0.85 / 0.95 / 0.61. That is why a single anchor in Nepal is not an anecdote: the recipe travels.'}</p>
          </section>

          <section className="story-section story-step story-boundary">
            <p className="story-kicker">05 · {ko ? '경계' : 'THE BOUNDARY'} — <em>{ko ? '말하기를 거부하는 것' : 'what we refuse to say'}</em></p>
            <h2>{ko ? '주장의 경계' : 'Claim boundary'}</h2>
            <p>{ko
              ? '이 비교가 지지하는 것은 “후보 변화(candidate change)”까지임 — 그 이상이 아님. 피해 확률·사상자·원인·깊이를 주장하지 않음. 눈사태 귀속은 “추정, 조사 중”으로 유지함. 임베딩 판정은 모든 게이트를 통과한 뒤에만 계산됨: 장면 선택 봉인, 센서당 정확히 4개의 14일 기간, 코드 스냅샷 해시, placebo 분포 확보. 하늘이 관측 불가능할 때 정직한 출력은 추측이 아니라 보류(abstention)임.'
              : 'This comparison supports “candidate change” — nothing more. It does not claim damage probability, casualty figures, cause, or depth. The avalanche attribution stays “suspected, under investigation.” The embedding verdict is computed only after every gate passes: scene selection sealed, exactly four 14-day periods per sensor, code snapshot hashed, placebo distribution on file. Where the sky is not observable, the honest output is abstention, not a guess.'}</p>
          </section>

          <section className="story-section story-step">
            <p className="story-kicker">06 · {ko ? '오늘 밤' : 'TONIGHT'} — <em>{ko ? '다음 관측' : 'the next observation'}</em></p>
            <h2>{ko ? '오늘 밤 레이더가 결정하는 것' : 'What tonight’s radar pass settles'}</h2>
            <p>{ko
              ? 'Sentinel-1D가 오늘 밤 21:19 KST에 이 계곡을 지남 — 구름과 무관함. 마지막 남은 14일 레이더 기간이 채워지면 사건 후 큐브 봉인이 완성되고 첫 라이브 Δz 판정이 나옴. 또한 보도된 언색호가 낮은 후방산란의 어두운 패치로 찍혀 실제 위치가 고정될 것임. 모든 단계는 메인 화면의 operations log에 현장 레인저들이 쓰는 이벤트-레코드 문법 그대로 기록됨.'
              : 'Sentinel-1D crosses this valley at 21:19 KST tonight, cloud-blind. It fills the last missing 14-day radar period — unlocking the sealed post-event cube and the first live Δz verdict — and it should image the reported barrier lake as a dark, low-backscatter patch, fixing its true position. Every step lands in the operations log on the main screen, in the same event-record grammar rangers use in the field.'}</p>
          </section>

          <section className="story-section story-step">
            <p className="story-kicker">07 · {ko ? '아카이브' : 'THE ARCHIVE'} — <em>{ko ? '모든 관측' : 'every acquisition'}</em></p>
            <h2>{ko ? '이 계곡이 관측된 모든 순간' : 'Every time this valley was seen'}</h2>
            <p>{ko
              ? '아래는 이 서비스가 물질화한 관측 전부임 — 광학(S2)과 레이더(S1)가 번갈아 계곡을 지나감. 어떤 프레임도 합성이 아니고, 각 파일의 SHA-256이 봉인돼 있음.'
              : 'Below is every acquisition this service has materialized — optical (S2) and radar (S1) alternating over the valley. No frame is synthetic; each file’s SHA-256 is sealed.'}</p>
            <div className="story-archive">
              {(scenario?.scene_records ?? []).map((s) => (
                <figure key={s.id}>
                  <img src={s.image} alt={`${s.sensor} ${s.acquired_at.slice(0, 10)}`} loading="lazy" />
                  <figcaption>{shortSensor(s.sensor)} · {s.acquired_at.slice(5, 10)}</figcaption>
                </figure>
              ))}
            </div>
            <p className="story-outro">{ko ? '스토리를 닫고 로그를 지켜보라. 계곡은 아직 움직이고 있음.' : 'Close this story and watch the log. The valley is still moving.'}</p>
          </section>
        </div>
        );
      })()}

      <div className="provenance-stamp">DATA SNAPSHOT {scenario?.generated_at.slice(0, 16).replace('T', ' ') ?? '—'} UTC · OSM ODbL · ESA COPERNICUS</div>
    </main>
  );
}
