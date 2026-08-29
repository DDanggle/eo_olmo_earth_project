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
  nearest_window?: string | null;
  nearest_window_km?: number;
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
type AiRunRecord = {
  id: string;
  state: 'EXECUTED' | 'MEASURED' | 'MEASURED_PILOT' | 'NEGATIVE_RESULT' | 'WAITING_INPUT' | 'NOT_RUN' | 'SUPERSEDED';
  model: string;
  input: string;
  output: string;
  allows: string;
  forbids: string;
  artifact_sha256: string | Record<string, string> | null;
};
type ResearchBlock = {
  integration_disclaimer: string;
  nepal_embedding: { status: string; baseline: string; placebo_count: number; claim: string };
  ai_run_ledger: AiRunRecord[];
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
  coverage_status?: string | null;
  operational_anchor_count?: number | null;
  operational_anchor_covering_product_count?: number | null;
  selection_preflight_valid: boolean;
  materialization_seal_valid: boolean;
  period_readiness: { sentinel1?: number; sentinel2_l2a?: number };
  olmo_ready: boolean;
  claim_boundary: string;
};

type CurrentDecision = {
  status: 'candidate_ready' | 'not_detected' | 'embed_ready' | 'hold' | 'wait_observation';
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
  simulation: { route_points: number; mapped_route_km_from_border?: number; reported_total_travel_km?: number;
    reported_reach_source?: string; trace_endpoint?: { name: string; coordinates: [number, number] };
    trace_endpoint_boundary?: string; claim: string; scientific_upgrade?: string };
  corridor_contract?: {
    expected_windows: number; expected_layers_per_window: number; contract: string; stage: string; next_step: string;
    baseline: { complete_windows: number; partial_windows: string[]; missing_windows: string[]; completed_layers: number; total_layers: number; materialization_sealed: boolean; embedded_windows: number; embedding_sealed: boolean; updated_at_utc: string | null };
    s1_live: { complete_windows: number; partial_windows: string[]; missing_windows: string[]; completed_layers: number; total_layers: number; materialization_sealed: boolean; embedded_windows: number; embedding_sealed: boolean; updated_at_utc: string | null };
    placebo_b?: { complete_windows: number; partial_windows: string[]; missing_windows: string[]; completed_layers: number; total_layers: number; materialization_sealed: boolean; embedded_windows: number; embedding_sealed: boolean; updated_at_utc: string | null };
    claim_boundary: string;
  };
  input_contract_audit?: { status: string; defect: string; official_contract: string; official_source: string; superseded_results: string[]; claim_boundary: string } | null;
  corridor_sealed?: {
    schema: string; model: string; status: string; windows: number; max_exceedance: number; windows_with_any_exceedance: number;
    comparison: { event: string; ordinary: string; threshold: string; ordinary_transition_count: number };
    input_contract: Record<string, string>; claim: string; limitations: string[]; report_sha256: string; visual_legend: string;
    top: { id: string; rank: number; name: string; kind: string; center_lonlat: [number, number]; coordinates: [number, number][];
      event_mean: number; placebo_mean: number; placebo_p99: number; frac_above_local_placebo_p99: number;
      mean_ratio_event_to_placebo: number; s2_only_rank?: number | null; pre_image: string; post_image: string; delta_image: string }[];
    geojson: FeatureCollection;
  } | null;
  headline?: { sealed_candidates: number | null; sealed_total: number | null; sealed_not_detected: string[]; live_mode?: string; placebo_n?: number; corridor_ranked: number | null; corridor_windows?: number; corridor_top: string[]; matched?: { n_pairs: number; candidates: string[]; ranks: Record<string, string>; token?: Record<string, { event_frac: number | null; placebo_max: number; rank: number | null; candidate: boolean }>; token_candidates?: string[] } };
  ai_vs_classical?: { rows: { region: string; patches: number; classical_best: number; ai: number | null; gain: number | null }[]; regions: number; ahead: number; wins_at_005: number; pre_registered_margin: number; corridor?: { spearman: number; top10_overlap: number; reported_hits: { ai: number; classical: number } } | null } | null;
  candidates?: { schema: string; claim: string; threshold_placebo_p99: number | null; placebo_tokens: number; windows: number;
    top10: { id: string; rank: number; center_lonlat: [number, number]; candidate_token_frac: number; valid_event_frac: number; place?: string; distance_from_a_km?: number; kind?: string }[];
    hillslope_top?: { id: string; rank: number; center_lonlat: [number, number]; candidate_token_frac: number; valid_event_frac: number; place?: string; distance_from_a_km?: number; kind?: string }[];
    judged_by_kind?: Record<string, number>; unobservable_by_kind?: Record<string, number>;
    report_sha256: string; geojson: FeatureCollection;
    retrieval?: { query_windows: string[]; threshold: number; top10: { id: string; rank: number; similar_token_frac: number; place?: string; center_lonlat?: [number, number]; delta_rank?: number | null }[] } | null } | null;
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
  glyphs: 'https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf',
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
  const initialCorridorFitRef = useRef(false);
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
  // SSR과 첫 client render는 반드시 같은 값이어야 한다. window.hash를 state initializer에서
  // 읽으면 /#story 직링크에서 hydration mismatch가 난다(2026-08-29 브라우저 QA 실측).
  const [storyOpen, setStoryOpen] = useState(false);
  const [storyLang, setStoryLang] = useState<'en' | 'ko'>('en');
  // 큰 비교 뷰어(라이트박스): 어떤 작은 사진이든 클릭하면 전·후 슬라이더로 크게 봄.
  type Lightbox = { title: string; sub?: string; before: string; after: string; beforeLabel: string; afterLabel: string; extra?: { src: string; label: string }[] };
  const [lightbox, setLightbox] = useState<Lightbox | null>(null);
  const [lbSwipe, setLbSwipe] = useState(50);
  const [lbExtra, setLbExtra] = useState<number | null>(null);
  const openLightbox = useCallback((lb: Lightbox) => { setLbSwipe(50); setLbExtra(null); setLightbox(lb); }, []);
  useEffect(() => {
    if (!lightbox) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setLightbox(null); };
    window.addEventListener('keydown', onKey); return () => window.removeEventListener('keydown', onKey);
  }, [lightbox]);
  // 팝업(HTML 문자열) 안의 썸네일 클릭 → 라이트박스 (이벤트 위임)
  useEffect(() => {
    const h = (e: MouseEvent) => {
      const el = (e.target as HTMLElement).closest('.pp-thumbs') as HTMLElement | null;
      if (!el) return;
      const win = el.dataset.win; const name = el.dataset.name ?? ''; const place = el.dataset.place ?? '';
      if (el.dataset.ptc) {
        openLightbox({ title: name, sub: `${place} · negative-control window, 114 km from Rasuwagadhi`, before: '/data/candidates/ptC_pre.png', after: '/data/candidates/ptC_post.png', beforeLabel: 'PRE · 08-12', afterLabel: 'POST · 08-27 (cloud)' });
        return;
      }
      const cand = el.dataset.cand;
      if (cand) {
        openLightbox({ title: name, sub: `${place} · scan window ${cand}`, before: `/data/candidates/${cand}_pre.png`, after: `/data/candidates/${cand}_post.png`, beforeLabel: 'PRE · 08-12', afterLabel: 'POST · 08-27',
                       extra: [{ src: `/data/candidates/${cand}_delta.png`, label: 'AI change tokens (orange) on 08-27' }, ...(win === 'rasuwagadhi' ? [{ src: '/data/story/planet/ps_rasuwagadhi_0828.png', label: 'PlanetScope 3.8 m · 08-28 · © Planet Labs PBC CC-BY-NC-4.0' }] : [])] });
        return;
      }
      if (!win) return;
      const extra = win === 'rasuwagadhi' ? [{ src: '/data/story/planet/ps_rasuwagadhi_0828.png', label: 'PlanetScope 3.8 m · 08-28' }] : [];
      openLightbox({ title: name, sub: place, before: `/data/story/anchors/${win}_pre.png`, after: `/data/story/anchors/${win}_post.png`, beforeLabel: 'PRE · 08-12', afterLabel: 'POST · 08-27', extra });
    };
    document.addEventListener('click', h); return () => document.removeEventListener('click', h);
  }, [openLightbox]);
  const [swipe, setSwipe] = useState(52);
  const viewDimRef = useRef<'2d' | '3d'>('2d');
  const [selectedPoint, setSelectedPoint] = useState('E');
  const [overlayOpacity, setOverlayOpacity] = useState(0.78);
  const [showAnchors, setShowAnchors] = useState(true);
  const [flowPlaying, setFlowPlaying] = useState(true);
  const [visibleParticles, setVisibleParticles] = useState<number | null>(null);
  const visibleLogRef = useRef(0);
  const [flowSpeed, setFlowSpeed] = useState(0.034);
  const [candidateScope, setCandidateScope] = useState<'all' | 'river' | 'hillslope'>('all');
  const [satTiles, setSatTiles] = useState(false);
  const [candView, setCandView] = useState<{ id: string; rank?: number; place?: string; mode: 'pre' | 'post' | 'delta' } | null>(null);
  const [leftOpen, setLeftOpen] = useState(true);
  const [rightOpen, setRightOpen] = useState(true);
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    const syncStoryHash = () => setStoryOpen(window.location.hash === '#story');
    // Defer the client-only hash read so the first hydrated tree remains identical to SSR.
    const frame = window.requestAnimationFrame(syncStoryHash);
    window.addEventListener('hashchange', syncStoryHash);
    return () => {
      window.cancelAnimationFrame(frame);
      window.removeEventListener('hashchange', syncStoryHash);
    };
  }, []);

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
        center: [85.27, 28.06],
        zoom: 8.95,
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
      // styledata는 style 객체가 붙기 전에도 한 번 발생할 수 있다. 그 시점의 getStyle()은
      // 화면에는 무해하지만 MapLibre 경고를 남기므로 완성된 style에서만 진단한다.
      map.on('styledata', () => {
        if (!map.isStyleLoaded()) return;
        console.log('[diag] styledata — 레이어', map.getStyle()?.layers?.length ?? 0, '개');
      });
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
    if (!mapReady || !map || points.length === 0) return;
    // 벡터 스타일은 'load' 직후에도 isStyleLoaded()가 false일 수 있음 → idle에 한 번 더 시도 (2026-08-29 실측: 강·점·장면이 영영 안 붙던 원인)
    if (!map.isStyleLoaded()) { map.once('idle', () => setStyleRevision((r) => r + 1)); return; }
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
          'text-field': ['get', 'map_label'], 'text-size': 11,
          // E/D와 A/B는 수백 m 이내라 같은 anchor를 쓰면 모바일에서 한 덩어리로 겹친다.
          // 점 자체는 유지하되 서로 반대 방향으로 라벨을 밀어 사건 순서를 읽을 수 있게 한다.
          'text-offset': ['match', ['get', 'id'],
            'E', ['literal', [1.15, 0]], 'D', ['literal', [-1.15, 0]],
            'A', ['literal', [0, 1.8]], 'B', ['literal', [0, -1.8]],
            'G', ['literal', [0, 1.8]], ['literal', [0, 1.55]]],
          'text-anchor': ['match', ['get', 'id'],
            'E', 'left', 'D', 'right', 'A', 'top', 'B', 'bottom', 'top'],
          'text-font': ['Noto Sans Regular'], 'text-allow-overlap': true,
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
        const cw = pt.nearest_window ?? null;
        const thumbs = cw
          ? `<div class="pp-thumbs" data-cand="${cw}" data-name="${pt.name}" data-place="${pt.place}" data-win="${win ?? ''}" title="Click to compare large">`
            + `<figure><img src="/data/candidates/${cw}_pre.png" alt="pre"/><figcaption>PRE 08-12</figcaption></figure>`
            + `<figure><img src="/data/candidates/${cw}_post.png" alt="post"/><figcaption>POST 08-27</figcaption></figure>`
            + `<figure><img src="/data/candidates/${cw}_delta.png" alt="AI change"/><figcaption>AI Δ · win ${cw}</figcaption></figure>`
            + (win === 'rasuwagadhi' ? `<figure><img src="/data/story/planet/ps_rasuwagadhi_0828.png" alt="PlanetScope 28 Aug"/><figcaption>PLANETSCOPE 3.8 m · 08-28<br/><a href="https://source.coop/planet/disasterdata/nepal-flash-flood-2026-08-26" target="_blank" rel="noopener">© Planet Labs PBC · CC-BY-NC-4.0</a></figcaption></figure>` : '')
            + `</div><p class="pp-hint">▲ nearest scan window ${cw} (${pt.nearest_window_km} km) · click to open the large slider</p>`
          : pt.id === 'C'
          ? `<div class="pp-thumbs" data-ptc="1" data-name="${pt.name}" data-place="${pt.place}" title="Click to compare large">`
            + `<figure><img src="/data/candidates/ptC_pre.png" alt="pre"/><figcaption>PRE 08-12 (41% bright)</figcaption></figure>`
            + `<figure><img src="/data/candidates/ptC_post.png" alt="post"/><figcaption>POST 08-27 (100% cloud)</figcaption></figure>`
            + `</div><p class="pp-hint">▲ control window · 27 Aug is fully cloud-covered here — that is why C is a placebo/reference, not a judged site</p>`
          : win
          ? `<div class="pp-thumbs" data-win="${win}" data-name="${pt.name}" data-place="${pt.place}" title="Click to compare large">`
            + `<figure><img src="/data/story/anchors/${win}_pre.png" alt="pre"/><figcaption>PRE 08-12</figcaption></figure>`
            + `<figure><img src="/data/story/anchors/${win}_post.png" alt="post"/><figcaption>POST 08-27</figcaption></figure>`
            + (win === 'rasuwagadhi' ? `<figure><img src="/data/story/planet/ps_rasuwagadhi_0828.png" alt="PlanetScope 28 Aug"/><figcaption>PLANETSCOPE 3.8 m · 08-28<br/><a href="https://source.coop/planet/disasterdata/nepal-flash-flood-2026-08-26" target="_blank" rel="noopener">© Planet Labs PBC · CC-BY-NC-4.0 · source.coop</a></figcaption></figure>` : '')
            + `</div><p class="pp-hint">▲ click any frame to open the large before/after slider</p>`
          : '';
        new Popup({ closeButton: true, maxWidth: '400px', className: 'story-popup' })
          .setLngLat(pt.coordinates)
          .setHTML(`<p class="pp-eyebrow" style="color:${pt.marker_color}">${pt.display_label}${pt.id === 'C' ? ' · OUTSIDE EVENT CHAIN' : ''}</p>`
            + `<h3>${pt.name}</h3><p class="pp-place">${pt.place}</p>`
            + thumbs
            + (pt.story ? `<p class="pp-story">${pt.story}</p>` : '')
            + `<p class="pp-src">coordinate source: ${pt.source_url ? `<a href="${pt.source_url}" target="_blank" rel="noreferrer">${pt.source ?? 'source'} ↗</a>` : (pt.source ?? '').replace('user coordinate + OSM Nominatim reverse lookup', 'user-specified point · place name from OSM reverse geocoding')}</p>`)
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
    if (!mapReady || !map || !hydrography || map.getSource('hydrography')) return;
    // 벡터 스타일은 'load' 직후에도 isStyleLoaded()가 false일 수 있음 → idle에 한 번 더 시도 (2026-08-29 실측: 강·점·장면이 영영 안 붙던 원인)
    if (!map.isStyleLoaded()) { map.once('idle', () => setStyleRevision((r) => r + 1)); return; }
    const before = map.getLayer('point-halo') ? 'point-halo' : undefined;
    map.addSource('hydrography', { type: 'geojson', data: hydrography as FeatureCollection });
    map.addLayer({ id: 'river-casing', type: 'line', source: 'hydrography', paint: { 'line-color': '#06100e', 'line-width': 8, 'line-opacity': 0.82 } }, before);
    map.addLayer({ id: 'river-route', type: 'line', source: 'hydrography', paint: { 'line-color': '#0f5fd7', 'line-width': 2.4, 'line-opacity': 0.9 } }, before);
    // 파란 실선 = OSM 하천, 빨간 점선 = USGS 잠정 이동 보고를 따라 검사 중인 회랑.
    // 빨간 선은 침수 폭이나 최종 퇴적 경계를 뜻하지 않는다.
    map.addLayer({ id: 'reported-reach', type: 'line', source: 'hydrography', paint: {
      'line-color': '#d9363e', 'line-width': 2.6, 'line-opacity': 0.82,
      'line-dasharray': [2.4, 1.4], 'line-offset': 4,
    } }, before);
  }, [hydrography, mapReady, styleRevision]);

  useEffect(() => {
    const map = mapRef.current;
    if (!mapReady || !map || !map.isStyleLoaded()) return;
    fetch('/data/olmo-input-anchors.geojson').then((r) => r.json() as Promise<FeatureCollection>).then((anchors) => {
      // 벡터 스타일은 'load' 직후에도 isStyleLoaded()가 false일 수 있음 → idle에 한 번 더 시도
      if (!map.isStyleLoaded()) { map.once('idle', () => setStyleRevision((r) => r + 1)); return; }
      const before = map.getLayer('point-halo') ? 'point-halo' : undefined;
      // 2026-08-30 결함 수정: 예전엔 olmo-anchors 가 이미 있으면 여기서 return 해서, scenario 가 늦게 도착한
      // 뒤의 재실행에서 후보 사각형·청록 점·검색 윤곽이 영영 추가되지 않았음 (사용자 "청록색이 안 보여").
      if (!map.getSource('olmo-anchors')) {
        map.addSource('olmo-anchors', { type: 'geojson', data: anchors });
        map.addLayer({ id: 'olmo-anchor-fill', type: 'fill', source: 'olmo-anchors', paint: { 'fill-color': '#5fffd7', 'fill-opacity': 0.045 } }, before);
      }
      // Contract-correct canonical OLMo result: vivid orange and O-ranks.
      // This is separate from the amber S2-only discovery scan below.
      if (scenario?.corridor_sealed?.geojson && !map.getSource('olmo-canonical')) {
        map.addSource('olmo-canonical', { type: 'geojson', data: scenario.corridor_sealed.geojson });
        map.addLayer({ id: 'olmo-canonical-fill', type: 'fill', source: 'olmo-canonical',
          paint: { 'fill-color': '#ff6a21', 'fill-opacity': ['interpolate', ['linear'], ['coalesce', ['get', 'exceedance'], 0], 0, 0.02, 0.001, 0.15, 0.0042, 0.48] } }, before);
        map.addLayer({ id: 'olmo-canonical-line', type: 'line', source: 'olmo-canonical',
          paint: { 'line-color': '#ff5a1f', 'line-width': ['case', ['<=', ['get', 'rank'], 6], 3.2, 1.1], 'line-opacity': ['case', ['<=', ['get', 'rank'], 6], 0.98, 0.38] } }, before);
        try { map.addLayer({ id: 'olmo-canonical-rank', type: 'symbol', source: 'olmo-canonical', filter: ['<=', ['get', 'rank'], 6],
          layout: { 'text-field': ['concat', 'O', ['to-string', ['get', 'rank']]], 'text-size': 15,
                    'text-font': ['Open Sans Bold', 'Arial Unicode MS Bold'], 'text-allow-overlap': true },
          paint: { 'text-color': '#ff5a1f', 'text-halo-color': '#fffaf3', 'text-halo-width': 2.5 } }); }
        catch (e) { console.warn('[diag] canonical OLMo labels skipped', e); }
        map.on('click', 'olmo-canonical-fill', (e) => {
          const pr = e.features?.[0]?.properties as Record<string, unknown> | undefined; if (!pr) return;
          const id = String(pr.id); const row = scenario.corridor_sealed?.top.find((item) => item.id === id);
          if (row) {
            openLightbox({ title: `O${row.rank} · ${row.name}`, sub: `${(100 * row.frac_above_local_placebo_p99).toFixed(2)}% tokens above this location's single ordinary-transition p99 · screening, not damage`, before: row.pre_image, after: row.post_image, beforeLabel: 'PRE · 08-12', afterLabel: 'POST · 08-27', extra: [{ src: row.delta_image, label: 'OLMo Δ intensity; yellow-white = above local placebo p99' }] });
          }
        });
        map.on('mouseenter', 'olmo-canonical-fill', () => { map.getCanvas().style.cursor = 'pointer'; });
        map.on('mouseleave', 'olmo-canonical-fill', () => { map.getCanvas().style.cursor = ''; });
      }
      // AI 후보 창 (S2-only, 미봉인) — 후보 토큰 비율로 채움 농도.
      if (scenario?.candidates?.geojson && !map.getSource('ai-candidates')) {
        map.addSource('ai-candidates', { type: 'geojson', data: scenario.candidates.geojson });
        // 모든 스캔 창 중심: 작은 청록 점 (위성이 찍힌 모든 자리)
        map.addSource('scan-centers', { type: 'geojson', data: { type: 'FeatureCollection', features: scenario.candidates.geojson.features.map((f) => ({ type: 'Feature', properties: f.properties, geometry: { type: 'Point', coordinates: (f.properties?.center_lonlat as [number, number]) } })) } });
        map.addLayer({ id: 'scan-center-dot', type: 'circle', source: 'scan-centers',
          paint: { 'circle-radius': ['interpolate', ['linear'], ['zoom'], 8, 3.5, 12, 6, 15, 9], 'circle-color': '#19d3b0', 'circle-stroke-color': '#fffefb', 'circle-stroke-width': 2, 'circle-opacity': 1 } });
        map.addLayer({ id: 'ai-candidate-fill', type: 'fill', source: 'ai-candidates',
          paint: { 'fill-color': ['case', ['==', ['get', 'kind'], 'hillslope'], '#7b3fbf', '#d99a24'],
                   'fill-opacity': ['interpolate', ['linear'], ['coalesce', ['get', 'candidate_token_frac'], 0], 0, 0.02, 0.05, 0.18, 0.2, 0.42, 0.5, 0.6] } }, before);
        try { map.addLayer({ id: 'ai-candidate-rank', type: 'symbol', source: 'ai-candidates',
          filter: ['has', 'rank'],
          layout: { 'text-field': ['concat', '#', ['to-string', ['get', 'rank']]], 'text-size': ['case', ['<=', ['get', 'rank'], 6], 15, 11],
                    'text-font': ['Open Sans Bold', 'Arial Unicode MS Bold'], 'text-allow-overlap': true, 'text-anchor': 'center' },
          paint: { 'text-color': ['case', ['<=', ['get', 'rank'], 6], '#b77708', '#7a4a2e'], 'text-halo-color': '#fffefb', 'text-halo-width': 2 } }); }
        catch (e) { console.warn('[diag] candidate rank labels skipped', e); }
        const simIds = (scenario?.candidates?.retrieval?.top10 ?? []).map((r) => r.id);
        if (simIds.length) {
          map.addLayer({ id: 'ai-similar-line', type: 'line', source: 'ai-candidates', filter: ['in', ['get', 'id'], ['literal', simIds]],
            paint: { 'line-color': '#2a78d6', 'line-width': 2.2, 'line-dasharray': [1.5, 1.2], 'line-opacity': 0.9 } }, before);
        }
        map.on('click', 'ai-candidate-fill', (e) => {
          const pr = e.features?.[0]?.properties as Record<string, unknown> | undefined; if (!pr) return;
          const id = String(pr.id); const rank = pr.rank ? `#${pr.rank}` : 'not judged (cloud/snow)';
          if (satTiles) {  // 위성 타일 모드: 클릭 즉시 큰 전·후 슬라이더
            openLightbox({ title: `Scan window ${id} · ${rank}`, sub: `${pr.kind === 'hillslope' ? 'off-river hillslope' : String(pr.kind ?? 'river')} · ${typeof pr.candidate_token_frac === 'number' ? (100 * (pr.candidate_token_frac as number)).toFixed(0) + '% changed tokens' : 'not judged'}`, before: `/data/candidates/${id}_pre.png`, after: `/data/candidates/${id}_post.png`, beforeLabel: 'PRE · 08-12', afterLabel: 'POST · 08-27', extra: [{ src: `/data/candidates/${id}_delta.png`, label: 'AI change tokens (orange) on 08-27' }] });
            return;
          }
          const kindLabel = pr.kind === 'hillslope' ? 'OFF-RIVER HILLSLOPE' : pr.kind === 'lhende' ? 'LHENDE UPSTREAM' : 'RIVER';
          const frac = typeof pr.candidate_token_frac === 'number' ? `${(100 * (pr.candidate_token_frac as number)).toFixed(0)}% changed tokens` : 'not judged';
          const vis = typeof pr.valid_event_frac === 'number' ? `${(100 * (pr.valid_event_frac as number)).toFixed(0)}% observable` : '';
          new Popup({ closeButton: true, maxWidth: '420px', className: 'story-popup' }).setLngLat(e.lngLat)
            .setHTML(`<p class="pp-eyebrow">${kindLabel} · ${rank}</p><h3>Scan window ${id}</h3><p class="pp-place">${frac} · ${vis}</p>`
              + `<div class="pp-thumbs" data-cand="${id}" data-name="Scan window ${id}" data-place="${kindLabel} · ${rank}" title="Click to compare large">`
              + `<figure><img src="/data/candidates/${id}_pre.png" alt="pre"/><figcaption>PRE 08-12</figcaption></figure>`
              + `<figure><img src="/data/candidates/${id}_post.png" alt="post"/><figcaption>POST 08-27</figcaption></figure>`
              + `<figure><img src="/data/candidates/${id}_delta.png" alt="AI change"/><figcaption>AI Δ</figcaption></figure></div>`
              + `<p class="pp-hint">▲ click to open the large slider · orange = changed more than any ordinary fortnight · grey = cloud/snow</p>`).addTo(map);
        });
        map.on('click', 'scan-center-dot', (e) => {
          const pr = e.features?.[0]?.properties as Record<string, unknown> | undefined; if (!pr) return;
          const id = String(pr.id);
          if (satTiles) {
            openLightbox({ title: `Scan window ${id}`, sub: pr.rank ? `rank #${pr.rank}` : 'not judged (cloud/snow)', before: `/data/candidates/${id}_pre.png`, after: `/data/candidates/${id}_post.png`, beforeLabel: 'PRE · 08-12', afterLabel: 'POST · 08-27', extra: [{ src: `/data/candidates/${id}_delta.png`, label: 'AI change tokens (orange) on 08-27' }] });
            return;
          }
          new Popup({ closeButton: true, maxWidth: '420px', className: 'story-popup' }).setLngLat(e.lngLat)
            .setHTML(`<p class="pp-eyebrow">SCAN WINDOW · ${pr.rank ? '#' + pr.rank : 'not judged'}</p><h3>${id}</h3>`
              + `<div class="pp-thumbs" data-cand="${id}" data-name="Scan window ${id}" data-place="" title="Click to compare large">`
              + `<figure><img src="/data/candidates/${id}_pre.png" alt="pre"/><figcaption>PRE 08-12</figcaption></figure>`
              + `<figure><img src="/data/candidates/${id}_post.png" alt="post"/><figcaption>POST 08-27</figcaption></figure>`
              + `<figure><img src="/data/candidates/${id}_delta.png" alt="AI change"/><figcaption>AI Δ</figcaption></figure></div>`
              + `<p class="pp-hint">▲ click to open the large slider</p>`).addTo(map);
        });
        map.on('mouseenter', 'scan-center-dot', () => { map.getCanvas().style.cursor = 'pointer'; });
        map.on('mouseleave', 'scan-center-dot', () => { map.getCanvas().style.cursor = ''; });
        map.on('mouseenter', 'ai-candidate-fill', () => { map.getCanvas().style.cursor = 'pointer'; });
        map.on('mouseleave', 'ai-candidate-fill', () => { map.getCanvas().style.cursor = ''; });
        if (map.getLayer('scan-center-dot')) map.moveLayer('scan-center-dot');
        map.addLayer({ id: 'ai-candidate-line', type: 'line', source: 'ai-candidates',
          paint: { 'line-color': ['case', ['==', ['get', 'kind'], 'hillslope'], '#7b3fbf', '#d99a24'], 'line-width': ['case', ['<=', ['coalesce', ['get', 'rank'], 99], 5], 2, 0.6], 'line-opacity': ['case', ['==', ['get', 'status'], 'ranked'], 0.8, 0.25] } }, before);
        if (map.getLayer('scan-center-dot')) map.moveLayer('scan-center-dot');
      }
      map.addLayer({ id: 'olmo-anchor-line', type: 'line', source: 'olmo-anchors', paint: { 'line-color': '#b7ffe9', 'line-width': 1, 'line-opacity': 0.52, 'line-dasharray': [3, 2] } }, before);
    }).catch(() => undefined);
  }, [mapReady, styleRevision, scenario?.candidates?.geojson, scenario?.candidates?.retrieval?.top10, scenario?.corridor_sealed, satTiles, openLightbox]);

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
    // 벡터 스타일은 'load' 직후에도 isStyleLoaded()가 false일 수 있음 → idle에 한 번 더 시도 (2026-08-29 실측: 강·점·장면이 영영 안 붙던 원인)
    if (!map.isStyleLoaded()) { map.once('idle', () => setStyleRevision((r) => r + 1)); return; }
    const scene = scenario.scene_records.find((item) => item.id === activeSceneId);
    if (!scene) return;
    if (map.getLayer('satellite-scene')) map.removeLayer('satellite-scene');
    if (map.getSource('satellite-scene')) map.removeSource('satellite-scene');
    const before = map.getLayer('point-halo') ? 'point-halo' : undefined;
    map.addSource('satellite-scene', { type: 'image', url: scene.image, coordinates: scene.coordinates });
    map.addLayer({ id: 'satellite-scene', type: 'raster', source: 'satellite-scene', paint: { 'raster-opacity': overlayOpacity, 'raster-fade-duration': 120, 'raster-saturation': 0.12, 'raster-contrast': 0.08, 'raster-resampling': 'nearest' } }, before);
    if (!userSelectedSceneRef.current) {
      // 첫 화면은 단일 A/B 위성창이 아니라 SOURCE→DOWNSTREAM 사건 전체를 보여준다.
      // 단 **최초 1회만**: 이 효과는 styleRevision 등으로 재실행되는데, 그때마다 fitBounds 하면
      // GO/확대 뒤 화면이 원위치로 튀는 결함이 생김 (2026-08-29 사용자 보고).
      if (!initialCorridorFitRef.current) {
        initialCorridorFitRef.current = true;
        map.fitBounds(new LngLatBounds([84.96, 27.77], [85.55, 28.36]), {
          padding: scenePadding(), maxZoom: 10.8, duration: 0,
        });
      }
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
          // 2026-08-29: 'lighter'(가산) + 민트색은 어두운 지도 전제였음. 밝은 종이 톤·MapTiler
          // 배경에서는 흰색으로 사라져 "애니메이션이 없다"로 보였음. 밝은 배경에서 보이는
          // 진한 색 + 일반 합성으로 바꾸고, 흰 테두리로 위성 장면 위에서도 분리되게 함.
          context.globalCompositeOperation = 'source-over';
          context.shadowColor = 'rgba(255, 255, 255, 0.9)';
          context.shadowBlur = 3;
          let onScreen = 0;
          for (let index = 0; index < count; index += 1) {
            const screen = map.project([values[index * 3], values[index * 3 + 1]]);
            if (screen.x < 0 || screen.y < 0 || screen.x > width || screen.y > height) continue;
            onScreen += 1;
            context.globalAlpha = 0.35 + values[index * 3 + 2] * 0.65;
            context.fillStyle = index % 7 === 0 ? '#eb6834' : '#0f5fd7';
            context.beginPath();
            context.arc(screen.x, screen.y, index % 7 === 0 ? 2.6 : 1.9, 0, Math.PI * 2);
            context.fill();
          }
          context.globalAlpha = 1;
          context.shadowBlur = 0;
          if ((visibleLogRef.current += 1) % 60 === 1) {
            setVisibleParticles(onScreen);
            if (visibleLogRef.current === 1) console.log('[diag] flow first frame | particles =', count, '| on-screen =', onScreen, '| canvas =', width + 'x' + height);
          }
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

  const liveDelta = useMemo(() => {
    const ped = scenario?.olmoearth?.post_event_delta;
    return typeof ped === 'object' && ped && (ped as Record<string, unknown>).live_mode ? (ped as Record<string, unknown>) : null;
  }, [scenario]);
  // GO TO MAP: 후보 창의 위성 사진(전/후/AI Δ)을 지도 위에 실제 좌표로 깔아 보여줌.
  const showCandidate = useCallback((id: string, mode: 'pre' | 'post' | 'delta', meta?: { rank?: number; place?: string; center?: [number, number] }) => {
    const map = mapRef.current; const fc = scenario?.candidates?.geojson;
    if (!map || !fc) return;
    const f = fc.features.find((x) => x.properties?.id === id);
    if (!f || f.geometry.type !== 'Polygon') return;
    const ring = f.geometry.coordinates[0] as [number, number][];  // SW, SE, NE, NW, SW
    const coords: [[number, number], [number, number], [number, number], [number, number]] = [ring[3], ring[2], ring[1], ring[0]];
    if (map.getLayer('cand-scene')) map.removeLayer('cand-scene');
    if (map.getSource('cand-scene')) map.removeSource('cand-scene');
    map.addSource('cand-scene', { type: 'image', url: `/data/candidates/${id}_${mode}.png`, coordinates: coords });
    const before = map.getLayer('ai-candidate-fill') ? 'ai-candidate-fill' : (map.getLayer('point-halo') ? 'point-halo' : undefined);
    map.addLayer({ id: 'cand-scene', type: 'raster', source: 'cand-scene', paint: { 'raster-opacity': 1, 'raster-fade-duration': 120 } }, before);
    const center = meta?.center ?? (f.properties?.center_lonlat as [number, number] | undefined);
    if (center) map.flyTo({ center, zoom: 14.2, pitch: 0, bearing: 0, duration: 900 });
    setCandView({ id, mode, rank: meta?.rank, place: meta?.place });
  }, [scenario]);
  const clearCandidate = useCallback(() => {
    const map = mapRef.current; if (map?.getLayer('cand-scene')) map.removeLayer('cand-scene'); if (map?.getSource('cand-scene')) map.removeSource('cand-scene'); setCandView(null);
  }, []);

  // 위성 타일 토글: 모든 스캔 창의 08-27 128px 썸네일을 실제 좌표에 드레이프
  useEffect(() => {
    const map = mapRef.current; const fc = scenario?.candidates?.geojson;
    if (!mapReady || !map || !fc) return;
    const ids = fc.features.map((f) => String(f.properties?.id));
    if (!satTiles) {
      ids.forEach((id) => { if (map.getLayer(`tile-${id}`)) map.removeLayer(`tile-${id}`); if (map.getSource(`tile-${id}`)) map.removeSource(`tile-${id}`); });
      return;
    }
    const before = map.getLayer('ai-candidate-fill') ? 'ai-candidate-fill' : undefined;
    fc.features.forEach((f) => {
      const id = String(f.properties?.id); if (map.getSource(`tile-${id}`) || f.geometry.type !== 'Polygon') return;
      const ring = f.geometry.coordinates[0] as [number, number][];
      map.addSource(`tile-${id}`, { type: 'image', url: `/data/candidates/thumbs/${id}_post128.png`, coordinates: [ring[3], ring[2], ring[1], ring[0]] });
      map.addLayer({ id: `tile-${id}`, type: 'raster', source: `tile-${id}`, paint: { 'raster-opacity': 0.92, 'raster-fade-duration': 0 } }, before);
    });
  }, [satTiles, mapReady, scenario]);

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
        : liveObservation.materialization_status === 'blocked_provider_selection'
          ? 'OFFICIAL 5/5 · PROVIDER INDEX WAIT'
        : liveObservation.materialization_status === 'selected_not_materialized'
          ? 'SCENE SELECTED · MATERIALIZE WAIT'
          : 'INPUT CONTRACT BLOCKED';
  const selectedCard = points.find((item) => item.id === selectedPoint) ?? points[0] ?? null;
  const eventPoints = points.filter((point) => point.in_event_chain);
  const controlPoints = points.filter((point) => !point.in_event_chain);
  const bidurPre = scenario?.downstream_visual.records.find((record) => record.label === 'pre') ?? null;
  const bidurPost = scenario?.downstream_visual.records.find((record) => record.label === 'post') ?? null;
  const providerSyncBlocked = liveObservation?.materialization_status === 'blocked_provider_selection';
  const corridorContract = scenario?.corridor_contract ?? null;
  const canonicalTop = scenario?.corridor_sealed?.top[0] ?? null;
  const candidateRows = !scenario?.candidates ? []
    : candidateScope === 'hillslope'
      ? (scenario.candidates.hillslope_top ?? [])
      : candidateScope === 'river'
        ? scenario.candidates.top10.filter((candidate) => candidate.kind !== 'hillslope')
        : scenario.candidates.top10;

  const focusPoint = (id: string) => {
    setSelectedPoint(id);
    const card = points.find((item) => item.id === id);
    if (!card) return;
    mapRef.current?.flyTo({
      center: card.coordinates, zoom: id === 'C' ? 10.5 : id === 'F' ? 13.2 : id === 'G' ? 12.2 : 14,
      pitch: viewDimRef.current === '3d' ? TERRAIN_PITCH : 0, bearing: viewDimRef.current === '3d' ? -18 : 0,
      duration: prefersReducedMotion() ? 0 : 1100,
    });
  };

  const fitCorridor = () => {
    userSelectedSceneRef.current = false;
    mapRef.current?.fitBounds(new LngLatBounds([84.96, 27.77], [85.55, 28.36]), {
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
      { name: 'Galchhi · trace end', lon: 84.9883, lat: 27.8055 },
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
      {scenario?.candidates && mapStatus === 'ready' && (
        <div className="map-legend" aria-label="Map legend">
          <span><i className="sw orange" />contract-correct sealed S1+S2 OLMo screening</span>
          <span><i className="sw amber" />S2-only optical discovery scan</span>
          <span><i className="sw purple" />off-river hillslope window</span>
          <span><i className="sw blue" />same kind of change as the top candidates (embedding search)</span>
          <span><i className="sw grey" />cloud/snow · not judged</span>
          <span><i className="sw teal" />A–G inspection points · click any box or point for before/after</span>
        </div>
      )}
      {candView && (
        <div className="cand-chip" role="status">
          <b>{candView.rank ? `#${candView.rank}` : candView.id}</b><span>{candView.place ?? ''}</span>
          <div className="cand-chip-modes">
            {(['pre', 'post', 'delta'] as const).map((m) => <button key={m} className={candView.mode === m ? 'is-active' : ''} onClick={() => showCandidate(candView.id, m, candView)}>{m === 'pre' ? 'PRE 08-12' : m === 'post' ? 'POST 08-27' : 'AI Δ'}</button>)}
          </div>
          <button className="cand-chip-close" onClick={clearCandidate} aria-label="Remove overlay">×</button>
        </div>
      )}
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
        <button className={`sat-tiles-toggle ${satTiles ? 'is-active' : ''}`} onClick={() => setSatTiles((v) => !v)} title="Drape every scan window's 27 Aug Sentinel-2 thumbnail on the map">{satTiles ? 'SATELLITE TILES ON' : 'SATELLITE TILES'}</button>
        <div className="map-mode-switch dim-switch" role="group" aria-label="View dimension">
          <button className={viewDim === '2d' ? 'is-active' : ''} onClick={() => setDimension('2d')} disabled={mapStatus !== 'ready'}>2D</button>
          <button className={viewDim === '3d' ? 'is-active' : ''} onClick={() => setDimension('3d')} disabled={mapStatus !== 'ready'}>3D</button>
        </div>
        <button className="story-launch" onClick={() => setStoryOpen(true)}>STORY</button>
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
            <span className="ops-title">RIVER CORRIDOR · Bhote Koshi → Trishuli → Galchhi</span>
            <svg viewBox={`0 0 ${corridorSketch.W} ${corridorSketch.H}`} role="img"
                 aria-label="Bhote Koshi to Trishuli corridor from the source area through Rasuwagadhi to the current Galchhi trace endpoint">
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
            <div className="reach-facts">
              <span><b>{scenario?.simulation.mapped_route_km_from_border?.toFixed(1) ?? '73.7'} km</b>mapped river trace below Rasuwagadhi</span>
              <span><b>≈{scenario?.simulation.reported_total_travel_km ?? 100} km</b>USGS preliminary total travel from source</span>
              <em>G · Galchhi is this map&apos;s trace end, not a confirmed terminal deposit.</em>
            </div>
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
          <span><i className="blue" />Mapped river centerline</span>
          <span><i className="red" />Preliminary reported reach corridor</span>
          <span><i className="white" />OLMo input 2.56 km</span>
          <span><i className="amber" />Unverified / pending</span>
        </div>
      </aside>
      )}

      {rightOpen && (
      <aside className="right-rail glass-panel">
        {scenario?.headline && (
          <div className="headline-card">
            <p className="eyebrow">AT A GLANCE · {scenario.generated_at.slice(0, 10)}</p>
            <strong>{scenario.corridor_sealed
              ? `Contract-correct OLMo screening complete: ${scenario.corridor_sealed.windows_with_any_exceedance}/${scenario.corridor_sealed.windows} windows contain any token above their own single ordinary-transition p99; maximum ${(100 * scenario.corridor_sealed.max_exceedance).toFixed(2)}%`
              : scenario.headline.matched?.token_candidates?.length
              ? `Token-level (matched): ${scenario.headline.matched.token_candidates.join(', ')} shows candidate change — ${(100 * (scenario.headline.matched.token![scenario.headline.matched.token_candidates[0]].event_frac ?? 0)).toFixed(1)}% of tokens vs ≤${(100 * scenario.headline.matched.token![scenario.headline.matched.token_candidates[0]].placebo_max).toFixed(1)}% in any ordinary fortnight`
              : scenario.headline.sealed_candidates != null
              ? (scenario.headline.sealed_candidates === 0
                  ? `Anchor-scale Δz: not detected above pre-event variability (placebo n=${scenario.headline.placebo_n})`
                  : `${scenario.headline.sealed_candidates} of ${scenario.headline.sealed_total} sealed anchors show candidate change`)
              : 'Sealed verdict not yet computed'}</strong>
            {!scenario.corridor_sealed && scenario.headline.matched?.token && <small>Token-level ranks (event vs 9 matched pairs): {Object.entries(scenario.headline.matched.token).map(([a, v]) => `${a.replace('_provisional', '')} ${v.rank ?? '—'}/10 (${((v.event_frac ?? 0) * 100).toFixed(1)}%)`).join(' · ')} · anchor-mean Δz alone: not detected</small>}
            {!scenario.corridor_sealed && scenario.headline.matched && <small>Matched 1-period pairs (n={scenario.headline.matched.n_pairs}): {scenario.headline.matched.candidates.length ? `${scenario.headline.matched.candidates.join(', ')} rank 1 (by a hair)` : 'no anchor ranks first'} · ranks {Object.entries(scenario.headline.matched.ranks).map(([a, r]) => `${a.replace('_provisional', '')} ${r}`).join(' · ')}</small>}
            {scenario.headline.sealed_candidates !== 0 && <small>{scenario.headline.sealed_not_detected.length ? `not detected (cloud/snow): ${scenario.headline.sealed_not_detected.join(', ')} · ` : ''}{scenario.headline.placebo_n != null ? `placebo n=${scenario.headline.placebo_n}` : ''}</small>}
            {scenario.headline.corridor_ranked != null && <small>Corridor scan: {scenario.headline.corridor_ranked}/{scenario.headline.corridor_windows} windows judged · top: {scenario.headline.corridor_top.join(' · ')}</small>}
            <em>{scenario.corridor_sealed ? 'Single-placebo screening only — not a calibrated detection, damage, cause, extent, or probability.' : 'Candidate change only — not damage, not cause, not probability.'}</em>
          </div>
        )}
        <div className="panel-heading"><span>02</span><div><p>AI EVIDENCE</p><strong>What works now</strong></div></div>
        {scenario?.input_contract_audit && (
          <div className="contract-audit-card">
            <p className="eyebrow">INPUT CONTRACT CORRECTION · SELF-AUDITED</p>
            <strong>Earlier five-anchor S1+S2 claims are superseded</strong>
            <p>{scenario.input_contract_audit.defect}</p>
            <small>The old files remain provenance records. Active evidence below was recomputed with Sentinel1ToDecibels, 27 matched windows and the same-location placebo.</small>
            <a href={scenario.input_contract_audit.official_source} target="_blank" rel="noreferrer">Official rslearn OLMoEarth contract ↗</a>
          </div>
        )}
        {scenario?.corridor_sealed && (
          <div className="canonical-olmo-card">
            <header><span>WHAT OLMO FOUND · CONTRACT-CORRECT</span><b>27/27 SCREENED</b></header>
            <p>OLMoEarth v1 converted each S1+S2 time cube into 4,096 spatial tokens × 768 dimensions. We measured cosine change from pre→post, then compared every token with the same location&apos;s ordinary placebo transition.</p>
            <div className="canonical-list">
              {scenario.corridor_sealed.top.map((row) => (
                <article key={row.id}>
                  <header><b>O{row.rank}</b><strong>{row.name}</strong><em>{(100 * row.frac_above_local_placebo_p99).toFixed(2)}%</em></header>
                  <div className="cand-strip zoomable" role="button" tabIndex={0}
                       onClick={() => openLightbox({ title: `O${row.rank} · ${row.name}`, sub: `${(100 * row.frac_above_local_placebo_p99).toFixed(2)}% above this location's one ordinary-transition p99 · event mean Δ ${row.event_mean.toFixed(4)} vs ordinary ${row.placebo_mean.toFixed(4)}`, before: row.pre_image, after: row.post_image, beforeLabel: 'PRE · 08-12', afterLabel: 'POST · 08-27', extra: [{ src: row.delta_image, label: 'OLMo Δ intensity; yellow-white = above local placebo p99' }] })}
                       onKeyDown={(event) => { if (event.key === 'Enter') (event.currentTarget as HTMLElement).click(); }}>
                    <figure><img src={row.pre_image} alt="pre-event Sentinel-2" loading="lazy" /><figcaption>PRE</figcaption></figure>
                    <figure><img src={row.post_image} alt="post-event Sentinel-2" loading="lazy" /><figcaption>POST</figcaption></figure>
                    <figure><img src={row.delta_image} alt="OLMoEarth embedding delta" loading="lazy" /><figcaption>OLMo Δ</figcaption></figure>
                  </div>
                  <div className="rarity-bar"><i style={{ width: `${Math.max(1.5, 100 * row.frac_above_local_placebo_p99 / Math.max(scenario.corridor_sealed!.max_exceedance, 1e-9))}%` }} /></div>
                  <footer><small>event mean is {(100 * row.mean_ratio_event_to_placebo).toFixed(0)}% of ordinary mean</small><button onClick={() => mapRef.current?.flyTo({ center: row.center_lonlat, zoom: 13.2, duration: 900 })}>GO TO MAP</button></footer>
                </article>
              ))}
            </div>
            <em>{scenario.corridor_sealed.visual_legend}. Ranking is a review queue, not a hazard map.</em>
          </div>
        )}
        {scenario?.ai_vs_classical && (
          <div className="ai-vs-card">
            <p className="eyebrow">AI vs NO-AI · same data, same labels, same metric</p>
            <strong>OLMoEarth Δz beats classical band-change in {scenario.ai_vs_classical.ahead}/{scenario.ai_vs_classical.regions} past disasters · {scenario.ai_vs_classical.wins_at_005}/{scenario.ai_vs_classical.regions} above the pre-registered +{scenario.ai_vs_classical.pre_registered_margin} AUROC margin</strong>
            <table className="ai-vs-table"><thead><tr><th>region</th><th>no-AI</th><th>AI</th><th>Δ</th></tr></thead><tbody>
              {scenario.ai_vs_classical.rows.map((r) => <tr key={r.region} className={(r.gain ?? 0) >= 0.05 ? 'win' : ''}><td>{r.region}</td><td>{r.classical_best.toFixed(2)}</td><td>{r.ai?.toFixed(2) ?? '—'}</td><td>{r.gain != null ? (r.gain >= 0 ? '+' : '') + r.gain.toFixed(2) : '—'}</td></tr>)}
            </tbody></table>
            <small>AUROC = probability a landslide token outranks a non-landslide token. no-AI = best of normalized band difference and |ΔNDVI|+|ΔNBR|, identical patches and pre/post scene choice (label-blind). Labels used for scoring only.{scenario.ai_vs_classical.corridor ? ` Nepal corridor (no labels): top-10 reported-place hits AI ${scenario.ai_vs_classical.corridor.reported_hits.ai} vs no-AI ${scenario.ai_vs_classical.corridor.reported_hits.classical}.` : ''}</small>
          </div>
        )}
        <div className="olmo-outcomes">
          <article className="ready"><span>OLMo CANONICAL CORRIDOR</span><strong>{scenario?.corridor_sealed ? '81 RASTERS SEALED' : 'PENDING'}</strong><small>placebo + baseline + live · 27 windows each · 768×64×64</small></article>
          <article className="win"><span>TRANSFER EVIDENCE</span><strong>{transfer ? `${transfer.wins_reuse_vs_raw_strong}/${transfer.regions} REGIONS WON` : 'LOADING'}</strong><small>{transfer ? `region-macro ${transfer.reuse_region_macro.toFixed(3)} vs ${transfer.raw_strong_region_macro.toFixed(3)} · +${transfer.absolute_gap.toFixed(3)}` : 'confirmatory summary'}</small></article>
          <article className={scenario?.corridor_sealed ? 'ready' : 'wait'}><span>NEPAL LIVE CHANGE</span><strong>{scenario?.corridor_sealed ? 'SCREENING COMPLETE · NO CALIBRATED DETECTION' : 'WAITING'}</strong><small>{scenario?.corridor_sealed ? `top local-p99 exceedance ${(100 * scenario.corridor_sealed.max_exceedance).toFixed(2)}% · one ordinary transition only` : `${livePeriodText} · baseline value remains usable`}</small></article>
        </div>
        {corridorContract && (
          <div className="corridor-progress" role="status">
            <header><span>SEALED CORRIDOR · 27 WINDOWS</span><b>{corridorContract.stage.replace(/_/g, ' ').toUpperCase()}</b></header>
            {([['PLACEBO', corridorContract.placebo_b], ['BASELINE', corridorContract.baseline], ['LIVE', corridorContract.s1_live]] as const).filter((entry) => entry[1]).map(([label, mode]) => {
              if (!mode) return null;
              const pct = Math.round(100 * mode.completed_layers / mode.total_layers);
              return <div className="corridor-progress-row" key={label}>
                <span>{label}</span><i><u style={{ width: `${pct}%` }} /></i>
                <strong>{mode.complete_windows}/{corridorContract.expected_windows}</strong><small>{pct}% · {mode.partial_windows.length} partial</small>
              </div>;
            })}
            <p>{corridorContract.next_step}</p><em>{corridorContract.claim_boundary}</em>
          </div>
        )}
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
          {canonicalTop ? (
            <div className="scene-preview delta-preview zoomable" role="button" tabIndex={0} title="Click: large view"
                 onClick={() => openLightbox({ title: `O${canonicalTop.rank} · ${canonicalTop.name}`, sub: `${(100 * canonicalTop.frac_above_local_placebo_p99).toFixed(2)}% above local placebo p99 · screening only`, before: canonicalTop.pre_image, after: canonicalTop.post_image, beforeLabel: 'PRE · 08-12', afterLabel: 'POST · 08-27', extra: [{ src: canonicalTop.delta_image, label: 'OLMo Δ intensity' }] })}>
              <img src={canonicalTop.post_image} alt="" className="delta-base" />
              <img src={canonicalTop.delta_image} alt="Contract-correct OLMoEarth delta heatmap" className="delta-heat" />
              <span>O{canonicalTop.rank} · SEALED · SCREENING</span>
            </div>
          ) : (
            <div className="scene-preview pending-preview">
              <span className="waiting-cross" />
              <span>POST · {liveObservation?.catalog_status === 'published' ? 'CATALOG / CUBE WAIT' : nextScheduled ? `${shortSensor(nextScheduled.sensor)} ${nextScheduled.acquired_at.slice(5, 10)}` : 'PENDING'}</span>
            </div>
          )}
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
          <div className={`pipeline-row ${scenario?.corridor_sealed ? 'ready' : 'pending'}`}><span>OE</span><div><strong>Canonical S1+S2 corridor screening</strong><small>{scenario?.corridor_sealed ? `27 windows · dB-corrected S1 · local placebo · max ${(100 * scenario.corridor_sealed.max_exceedance).toFixed(2)}%` : 'not computed'}</small></div><b>{scenario?.corridor_sealed ? 'SCREENED' : 'PENDING'}</b></div>
          <div className={`pipeline-row ${scenario?.candidates ? 'preview' : 'pending'}`}><span>AI</span><div><strong>Optical discovery queue · S2-only</strong><small>{scenario?.candidates ? `${scenario.candidates.windows} auto windows · placebo p99 ${scenario.candidates.threshold_placebo_p99?.toFixed(3)} · unsealed` : 'not computed'}</small></div><b>{scenario?.candidates ? 'LEADS' : 'PENDING'}</b></div>
          {scenario?.candidates && (
            <div className="candidate-cards">
              <p className="cand-help">{scenario.candidates.windows} auto windows{scenario.candidates.judged_by_kind ? ` · judged: river ${scenario.candidates.judged_by_kind.river ?? 0}, hillslope ${scenario.candidates.judged_by_kind.hillslope ?? 0}, lhende ${scenario.candidates.judged_by_kind.lhende ?? 0}` : ''}{scenario.candidates.unobservable_by_kind ? ` · cloud/snow (not judged): ${Object.values(scenario.candidates.unobservable_by_kind).reduce((a, b) => a + b, 0)}` : ''} · orange = changed more than any ordinary fortnight (placebo p99) · purple = off-river hillslope window</p>
              <div className="candidate-scopes" role="group" aria-label="Filter AI candidate windows">
                {(['all', 'river', 'hillslope'] as const).map((scope) => <button key={scope} className={candidateScope === scope ? 'is-active' : ''} onClick={() => setCandidateScope(scope)}>{scope === 'all' ? 'TOP ALL' : scope === 'river' ? 'RIVER' : 'OFF-RIVER'}</button>)}
              </div>
              {candidateRows.slice(0, 6).map((c) => (
                <article key={c.id} className="cand-card">
                  <header><b>#{c.rank}</b><strong>{c.place || `${c.center_lonlat[1].toFixed(3)}, ${c.center_lonlat[0].toFixed(3)}`}</strong><small>{c.kind === 'hillslope' ? 'OFF-RIVER HILLSLOPE · ' : c.kind === 'lhende' ? 'LHENDE UPSTREAM · ' : ''}{c.distance_from_a_km != null ? `${c.distance_from_a_km.toFixed(1)} km from border` : ''}</small></header>
                  <div className="cand-strip" role="button" tabIndex={0}
                       onClick={() => openLightbox({ title: `#${c.rank} · ${c.place || c.id}`, sub: `${(c.candidate_token_frac * 100).toFixed(0)}% of judged tokens above placebo p99 · ${(c.valid_event_frac * 100).toFixed(0)}% observable`, before: `/data/candidates/${c.id}_pre.png`, after: `/data/candidates/${c.id}_post.png`, beforeLabel: 'PRE · 08-12', afterLabel: 'POST · 08-27', extra: [{ src: `/data/candidates/${c.id}_delta.png`, label: 'AI change tokens (orange) on 08-27' }] })}
                       onKeyDown={(e) => { if (e.key === 'Enter') (e.currentTarget as HTMLElement).click(); }}>
                    <figure><img src={`/data/candidates/${c.id}_pre.png`} alt="before" loading="lazy" /><figcaption>PRE 08-12</figcaption></figure>
                    <figure><img src={`/data/candidates/${c.id}_post.png`} alt="after" loading="lazy" /><figcaption>POST 08-27</figcaption></figure>
                    <figure><img src={`/data/candidates/${c.id}_delta.png`} alt="AI change tokens" loading="lazy" /><figcaption>AI Δ</figcaption></figure>
                  </div>
                  <footer><span>{(c.candidate_token_frac * 100).toFixed(0)}% changed · {(c.valid_event_frac * 100).toFixed(0)}% visible</span>
                    <button onClick={() => showCandidate(c.id, 'post', { rank: c.rank, place: c.place, center: c.center_lonlat })}>GO TO MAP</button></footer>
                </article>
              ))}
              {scenario.candidates.hillslope_top && scenario.candidates.hillslope_top.length > 0 && (
                <div className="retrieval-box hillslope-box">
                  <p className="cand-help"><b>OFF-RIVER · hillslope grid around the source</b> — 49 windows ±7.7 km around Langtang Lirung; most are cloud/snow and not judged. Ranked ones below (low observability — treat as leads only).</p>
                  <ol>
                    {scenario.candidates.hillslope_top.map((r) => (
                      <li key={r.id}><b>{r.rank}</b><span>{r.place || r.id}</span><em>{(r.candidate_token_frac * 100).toFixed(0)}% changed · {(r.valid_event_frac * 100).toFixed(0)}% visible</em>
                        <button onClick={() => showCandidate(r.id, 'post', { rank: r.rank, place: r.place, center: r.center_lonlat })}>GO</button></li>
                    ))}
                  </ol>
                </div>
              )}
              {scenario.candidates.retrieval && (
                <div className="retrieval-box">
                  <p className="cand-help"><b>SEARCH · same kind of change</b> — query = change vectors of #{scenario.candidates.retrieval.query_windows.join(', #')} candidate tokens; every window&apos;s tokens scored by cosine to that query, threshold = placebo p99.</p>
                  <ol>
                    {scenario.candidates.retrieval.top10.slice(0, 8).map((r) => (
                      <li key={r.id}><b>{r.rank}</b><span>{r.place || r.id}</span><em>{(r.similar_token_frac * 100).toFixed(0)}% similar{r.delta_rank ? ` · Δ rank #${r.delta_rank}` : ''}</em>
                        {r.center_lonlat && <button onClick={() => showCandidate(r.id, 'post', { rank: r.rank, place: r.place, center: r.center_lonlat! })}>GO</button>}</li>
                    ))}
                  </ol>
                </div>
              )}
            </div>
          )}
          <div className="pipeline-row ready"><span>FM</span><div><strong>Frozen representation</strong><small>sealed baseline · transfer · downstream probes</small></div><b>READY</b></div>
          <div className="pipeline-row ready"><span>DV</span><div><strong>Bidur downstream pair</strong><small>{bidurPost ? `S2 ${bidurPost.acquired_at.slice(0, 10)} · tile ${bidurPost.mgrs_tile}` : 'visual audit'}</small></div><b>{bidurPost ? 'READY' : 'AUDIT'}</b></div>
          <div className="pipeline-row ready"><span>8R</span><div><strong>Cross-region transfer</strong><small>{transfer ? `${transfer.strong_wins} strong wins · ${transfer.non_win_regions.length} non-wins` : 'confirmatory'}</small></div><b>MEASURED</b></div>
          <div className={`pipeline-row ${scenario?.input_contract_audit ? 'pending' : liveDelta ? 'ready' : 'pending'}`}><span>5A</span><div><strong>Legacy five-anchor S1+S2 delta</strong><small>{scenario?.input_contract_audit ? 'superseded: missing Sentinel1ToDecibels · full placebo rerun required' : liveDelta ? 'executed' : 'not run'}</small></div><b>{scenario?.input_contract_audit ? 'SUPERSEDED' : liveDelta ? 'EXECUTED' : 'WAIT'}</b></div>
          <div className={`pipeline-row ${wasmStatus === 'ready' ? 'preview' : 'pending'}`}><span>Φ</span><div><strong>Physics ensemble</strong><small>r.avaflow primary · D-Claw check · satellite likelihood</small></div><b>NEXT BUILD</b></div>
        </div>
        <div className="risk-queues">
          <span>RISK SEARCH QUEUES · WHO ACTUALLY OWNS THE ANSWER</span>
          <article className="screened"><b>01 · CHANNEL CHANGE</b><strong>OLMoEarth S1+S2</strong><p>Devighat·Bidur·Rasuwagadhi review order. Corrected screening found only sparse local-p99 exceedances, not a calibrated detection.</p><em>SCREENED</em></article>
          <article className="lead"><b>02 · OFF-RIVER SLOPES</b><strong>S2 leads → S1 + DEM next</strong><p>Only 6/49 optical hillslope windows were observable. Salê/Gosaikunda are reacquisition leads—not landslide findings.</p><em>PARTIAL</em></article>
          <article className="planned"><b>03 · BARRIER LAKE / BLOCKAGE</b><strong>SAR + water extent + official footprint</strong><p>Search for new water/backscatter and channel blockage; OLMo can rank change after the footprint contract is sealed.</p><em>PLANNED</em></article>
          <article className="planned"><b>04 · RUNOUT / ARRIVAL</b><strong>r.avaflow · D-Claw</strong><p>DEM, release geometry, volume and rheology own depth, speed and arrival-time estimates. No physical run has executed.</p><em>NOT RUN</em></article>
          <article className="planned"><b>05 · PEOPLE / HEALTH ACCESS</b><strong>GIS network + official exposure data</strong><p>Road, bridge, settlement, clinic and WASH intersections are consequence analysis—not OLMo predictions.</p><em>NOT RUN</em></article>
        </div>
        <div className="field-review-links">
          <span>FIELD / OFFICIAL REVIEW · OPENS SEPARATELY</span>
          <a href="https://www.usgs.gov/media/images/2026-nepal-debris-avalanche-and-flash-flood-map" target="_blank" rel="noreferrer">USGS extent map ↗</a>
          <a href="https://www.unosat.org/products/" target="_blank" rel="noreferrer">UNOSAT Rasuwa / Nuwakot products ↗</a>
          <a href="https://source.coop/planet/disasterdata/nepal-flash-flood-2026-08-26" target="_blank" rel="noreferrer">Planet crisis imagery ↗</a>
        </div>
        {/* O/E/P/H 4-layer 계약 — 설계 문서의 관측/증거/물리/공식 분리를 UI에 명시함.
            P·H는 아직 산출물이 없으므로 회색 placeholder로 정직하게 표시함. */}
        <div className="layer-contract">
          <span>LAYER CONTRACT</span>
          <div className="layer-contract-row on"><b>O</b><span>Observation — S1 VV/VH · S2 12-band · masks</span><em>ACTIVE</em></div>
          <div className={`layer-contract-row ${scenario?.corridor_sealed ? 'on' : 'off'}`}><b>E</b><span>OLMo evidence — 768-d embedding · matched-location Δz screening</span><em>{scenario?.corridor_sealed ? 'ACTIVE' : 'PENDING'}</em></div>
          <div className="layer-contract-row off"><b>P</b><span>Physics — r.avaflow ensemble · D-Claw check</span><em>DESIGNED</em></div>
          <div className="layer-contract-row off"><b>H</b><span>Human/official — Charter · CEMS · USGS review</span><em>EXTERNAL</em></div>
        </div>
        <div className="flow-control">
          <button onClick={replayEventChain} aria-label="Replay the event-chain corridor animation">REPLAY CHAIN</button>
          <div>
            <label htmlFor="flow-speed"><span>{flowPlaying ? 'ROUTE PLAYING' : 'ROUTE PAUSED'}{visibleParticles != null ? ` · ${visibleParticles} ON SCREEN` : wasmStatus === 'ready' ? ' · 0 ON SCREEN' : ` · ${wasmStatus.toUpperCase()}`}</span><b>{(flowSpeed / 0.034).toFixed(1)}×</b></label>
            <input id="flow-speed" type="range" min="0.012" max="0.09" step="0.002" value={flowSpeed} onChange={(event) => setFlowSpeed(Number(event.target.value))} />
          </div>
        </div>
        <button className="flow-pause" onClick={() => setFlowPlaying((value) => !value)}>{flowPlaying ? 'PAUSE PARTICLES' : 'RESUME PARTICLES'}</button>
        <div className="truth-box"><span>CLAIM BOUNDARY</span><p>Particles follow the mapped OSM Bhote Koshi→Trishuli→Galchhi centerline. Blue is river geometry; the offset red dash is a preliminary reach-inspection corridor informed by USGS&apos;s ≈100 km report. Neither shows flood width, depth, arrival time, nor a confirmed terminal deposit.</p></div>
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
            <h1>{ko ? '산이 무너진 지 사흘, 위성은 무엇을 봤고 무엇을 보지 못했나' : 'It began on a mountain. It appeared in a river.'}</h1>
            <p className="story-lede">{ko
              ? '8월 26일 오전 8시 40분, 네팔 라수와 군의 랑탕 리룽 북사면에서 바위와 얼음이 함께 무너졌다. 토사와 물은 렌데 계곡을 타고 국경 마을 라수와가디를 덮친 뒤 트리슐리 강을 따라 72km를 내려가 비두르까지 닿았다. 사망자는 사흘 만에 600명을 넘었다. 이 페이지는 그 사흘 동안 유럽우주국의 무료 위성 두 대가 이 계곡을 어떻게 지켜봤는지, 그리고 인공지능이 그 관측으로 무엇을 계산했고 무엇을 계산하지 않았는지를 기록한다. 결론부터 말하면 인공지능은 아직 이 사건을 판정하지 않았다. 판정에 필요한 레이더 관측 한 장이 파이프라인에 도착하지 않았기 때문이다. 이 시스템은 그 공백을 메우지 않고 기다린다. 그것이 설계다.'
              : 'This system links the Langtang Lirung rock–ice collapse, the Rasuwagadhi border impact and a measured Sentinel-2 change about 47 km downstream at Bidur. The question is not whether AI foretold the disaster. It is whether OlmoEarth can place different sensors and locations in one representation space—and help select which physical explanations agree with observation.'}</p>
            <div className="hero-answer"><span>{ko ? '현재 답' : 'CURRENT ANSWER'}</span><strong>{liveDelta
              ? (ko ? 'S1+S2 라이브 임베딩은 실행됐다. 5개 앵커 평균에서는 탐지되지 않았고, Rasuwagadhi의 40 m 토큰만 대조기간을 넘는 검토 후보로 남았다. 이제 27창 동일계약 회랑 검증이 진행 중이다.' : 'SEALED S1+S2 EMBEDDING EXECUTED · ANCHOR MEAN NOT DETECTED · RASUWAGADHI TOKEN REVIEW OPEN · 27-WINDOW CORRIDOR RUN IN PROGRESS')
              : ko ? (providerSyncBlocked ? '광학 관측은 사건 전후 모두 확보됐다. 레이더 8월 28일 제품은 공식 카탈로그에 올라왔으나 지형보정본이 아직 없어 판정은 보류 중이다.' : '광학 관측은 사건 전후 모두 확보됐다. 레이더 마지막 한 장이 도착하면 판정이 시작된다.') : (providerSyncBlocked ? 'OBSERVATION + OFFICIAL S1 5/5 READY · EMBEDDING WAITS FOR PROVIDER SYNC' : 'OBSERVATION CHAIN CLOSED · NEPAL LIVE EMBEDDING WAITS FOR S1')}</strong></div>
          </section>

          <section className="story-section story-step story-wide">
            <p className="story-kicker">01 · {ko ? '사건 구조' : 'EVENT ANATOMY'} — <em>{ko ? '여섯 개의 점이 뜻하는 것' : 'roles, not letters'}</em></p>
            <h2>{ko ? '무너진 곳과 덮친 곳은 20km 떨어져 있다' : 'The collapse source is not the impact window'}</h2>
            <div className="event-chain-cards">{eventPoints.map((point) => <article key={point.id} style={{ '--point-color': point.marker_color } as CSSProperties}><b>{point.stage}</b><span>{point.display_label}</span><strong>{point.name}</strong><small>{point.id} · {point.distance_from_a_km.toFixed(1)} km from impact A</small><p>{ko ? point.story_ko : point.story}</p></article>)}</div>
            <div className="control-explainer"><b>Ø · C · NEGATIVE CONTROL</b><p>{ko ? controlPoints[0]?.story_ko : controlPoints[0]?.story}</p></div>
            <p className="story-caption">{ko ? 'E(빨강)는 발원 수색점, D(보라)는 위치 미공개 언색호 수색구역, A·B는 국경 충격창, F는 비두르 실측창이다. G는 Rasuwagadhi 아래 73.7 km를 이은 현재 지도 추적 종점이며 재해의 확정 종점이 아니다. C는 사건 밖 대조군이다. 이 점들은 사람이 근거로 지정했고, AI가 새로 제안한 후보는 주황/보라 2.56 km 격자로 따로 표시된다.' : 'E is the source-search estimate; D is the unresolved lake search zone; A/B are border impact windows; F is the Bidur observation. G is the current map-trace endpoint 73.7 river-km below Rasuwagadhi—not a confirmed disaster terminus. C is outside the chain. AI-proposed candidates are the separate orange/purple 2.56 km grids.'}</p>
          </section>

          <section className="story-section story-step story-wide">
            <p className="story-kicker">02 · {ko ? '위성 증거' : 'SATELLITE EVIDENCE'} — <em>{ko ? '사흘 동안 위성이 본 것' : 'time × distance'}</em></p>
            <h2>{ko ? '국경에서 시작된 회색 띠가 47km 아래에서도 확인됐다' : 'A border change now has a downstream counterpart'}</h2>
            <div className="evidence-pairs">
              <article><header><span>A · IMPACT</span><strong>Rasuwagadhi</strong></header>{sceneById('s2-2026-08-12') && sceneById('s2-2026-08-27') && <div className="story-swipe compact" style={{ ['--swipe' as string]: `${swipe}%` }}><img src={sceneById('s2-2026-08-27')!.image} alt="Rasuwagadhi Sentinel-2 post-event" /><div className="swipe-clip"><img src={sceneById('s2-2026-08-12')!.image} alt="Rasuwagadhi Sentinel-2 pre-event" /></div><div className="swipe-bar" /><span className="swipe-label pre">08-12</span><span className="swipe-label post">08-27</span><input type="range" min={0} max={100} value={swipe} aria-label="Compare Rasuwagadhi before and after" onChange={(e) => setSwipe(Number(e.target.value))} /></div>}<p>{ko ? '27창 계약교정 OLMo 스크린의 w00. 기존 5-anchor 주장은 입력계약 위반으로 폐기됐다.' : 'Window w00 in the contract-correct 27-window OLMo screen. The legacy five-anchor claim is superseded by an input-contract failure.'}</p></article>
              <article><header><span>F · DOWNSTREAM</span><strong>Bidur / Trishuli</strong></header><div className="fixed-pair zoomable" role="button" tabIndex={0} onClick={() => bidurPre && bidurPost && openLightbox({ title: 'F · Bidur / Trishuli', sub: 'Sentinel-2 · 2.56 km · tile 45RUL', before: bidurPre.image, after: bidurPost.image, beforeLabel: 'PRE · 08-12', afterLabel: 'POST · 08-27' })}>{bidurPre && <figure><img src={bidurPre.image} alt="Bidur Sentinel-2 before event" /><figcaption>PRE · 08-12</figcaption></figure>}{bidurPost && <figure><img src={bidurPost.image} alt="Bidur Sentinel-2 after event" /><figcaption>POST · 08-27</figcaption></figure>}<span className="zoom-hint">⤢ enlarge</span></div><p>{ko ? '기존 Rasuwagadhi 타일 밖, 인접 45RUL에서 새로 회수한 실제 2.56 km 창.' : 'A real 2.56 km pair recovered from adjacent MGRS tile 45RUL, missed by the original Rasuwagadhi-only catalog.'}</p></article>
            </div>
            <div className="distance-matrix">
              {['source', 'rasuwagadhi', 'timure', 'syabrubesi', 'dhunche', 'bidur'].map((name, i) => <div key={name}><b>{i + 1}</b><span>{name === 'source' ? 'SOURCE · Langtang Lirung' : name.toUpperCase()}</span><figure><img src={`/data/story/anchors/${name}_pre.png`} alt={`${name} before`} /></figure><i>→</i><figure><img src={`/data/story/anchors/${name}_post.png`} alt={`${name} after`} /></figure></div>)}
            </div>
            <figure className="story-figure zoomable" role="button" tabIndex={0} onClick={() => openLightbox({ title: 'Rasuwagadhi · PlanetScope 3.8 m · 28 Aug', sub: '© Planet Labs PBC · CC-BY-NC-4.0 · Planet Disaster Data on source.coop (planet/disasterdata/nepal-flash-flood-2026-08-26) · reference only, not AI input', before: '/data/story/anchors/rasuwagadhi_post.png', after: '/data/story/planet/ps_rasuwagadhi_0828.png', beforeLabel: 'SENTINEL-2 10 m · 08-27', afterLabel: 'PLANETSCOPE 3.8 m · 08-28' })}>
              <img src="/data/story/planet/ps_rasuwagadhi_0828.png" alt="PlanetScope 3.8 m view of Rasuwagadhi on 28 August 2026" /><span className="zoom-hint">⤢ compare with Sentinel-2</span>
              <figcaption className="story-caption">{ko
                ? '같은 국경 합류부를 상업위성 플래닛스코프가 8월 28일 오전에 찍은 3.8m 영상. 센티넬(10m)보다 2.6배 세밀해 두 물줄기가 만나는 지점의 토사 판과 그 안의 물길, 끊긴 도로가 그대로 보인다. 플래닛은 이번 재난에 한해 영상을 비상업 조건으로 공개했다(CC-BY-NC-4.0). 이 영상은 참고용이며 인공지능 입력에는 쓰지 않았다 — 입력 계약(밴드·해상도)이 다르기 때문이다. © Planet Labs PBC · CC-BY-NC-4.0 · Planet Disaster Data(source.coop/planet/disasterdata/nepal-flash-flood-2026-08-26, 아이템 20260828_045744_48_2544 visual).'
                : 'The same border confluence seen by a commercial PlanetScope satellite on the morning of 28 August at 3.8 m — 2.6× finer than Sentinel-2. The debris sheet at the junction, the channel threading through it and the severed road are visible directly. Planet released this imagery for the disaster under a non-commercial licence (CC-BY-NC-4.0). It is reference only: it is not fed to the AI, whose input contract (bands, resolution) differs. © Planet Labs PBC · CC-BY-NC-4.0 · Planet Disaster Data, source.coop/planet/disasterdata/nepal-flash-flood-2026-08-26 (item 20260828_045744_48_2544, visual asset).'}</figcaption>
            </figure>
            <div className="story-spectra zoomable" role="button" tabIndex={0} onClick={() => openLightbox({ title: 'Rasuwagadhi · 27 Aug · true colour vs SWIR', sub: 'SWIR B12·B8A·B04: vegetation green, wet sediment pink-brown, water deep blue', before: '/data/story/spec_true_post0827.png', after: '/data/story/spec_swir_post0827.png', beforeLabel: 'TRUE COLOUR', afterLabel: 'SWIR', extra: [{ src: '/data/story/spec_ndwi_post0827.png', label: 'NDWI water index' }, { src: '/data/story/spec_swir_pre0812.png', label: 'SWIR · pre-event 08-12' }] })}>
              <figure><img src="/data/story/spec_true_post0827.png" alt="True colour, Rasuwagadhi, 27 August" /><figcaption>{ko ? '트루컬러 · 08-27' : 'TRUE COLOUR · 08-27'}</figcaption></figure>
              <figure><img src="/data/story/spec_swir_post0827.png" alt="SWIR composite B12/B8A/B04, 27 August" /><figcaption>{ko ? 'SWIR 합성 B12·B8A·B04 · 08-27' : 'SWIR B12·B8A·B04 · 08-27'}</figcaption></figure>
              <figure><img src="/data/story/spec_ndwi_post0827.png" alt="NDWI water index, 27 August" /><figcaption>{ko ? 'NDWI 물 지수 · 08-27' : 'NDWI WATER INDEX · 08-27'}</figcaption></figure>
              <figure><img src="/data/story/spec_swir_pre0812.png" alt="SWIR composite before the event, 12 August" /><figcaption>{ko ? 'SWIR 합성 · 사건 전 08-12' : 'SWIR · PRE-EVENT 08-12'}</figcaption></figure>
            </div>
            <p>{ko
              ? '같은 Rasuwagadhi 창을 세 가지 눈으로 다시 그림. 트루컬러는 사람 눈(B04·B03·B02), SWIR 합성은 식생을 초록·젖은 퇴적물을 분홍-갈색·물을 짙은 청색으로 분리하고, NDWI는 물만 밝은 청색으로 뽑음. debris 판 안에 아직 흐르는 물길이 어디인지는 SWIR·NDWI에서만 분명함. 사건 전 8/12 SWIR과 비교하면 분홍-갈색 회랑의 폭 차이가 곧 후보 변화임. OLMoEarth가 12밴드를 모두 입력받는 이유가 이것임 — 사람 눈에 같아 보이는 픽셀이 스펙트럼에서는 다름.'
              : 'The same Rasuwagadhi window re-rendered three ways. True colour is the human eye (B04·B03·B02); the SWIR composite separates vegetation (green), wet sediment (pink-brown) and water (deep blue); NDWI isolates water as bright blue. Where water still flows inside the debris sheet is only clear in SWIR and NDWI. Against the pre-event 12 Aug SWIR, the width change of the pink-brown corridor is the candidate change itself. This is why OLMoEarth ingests all twelve bands: pixels that look alike to the eye differ in spectrum.'}</p>
            <p className="story-caption">{ko ? '위 비교는 모두 유럽우주국 센티넬-2호가 8월 12일과 27일에 찍은 것이다. 계곡 바닥을 따라 넓어진 회색 띠는 토사가 지나간 자리로 보이지만, 이 화면은 그것을 “피해”로 단정하지 않는다. 사람이 현장에서 확인하기 전까지는 “변화 후보”라고만 부른다. 붕괴 지점과 둔체 마을은 구름과 눈에 가려 판독이 어렵다. 8월 24일의 보랏빛 화면은 사진이 아니라 레이더 신호를 색으로 바꾼 것이다.' : 'Every row is Sentinel-2 from 12→27 Aug. Differences are candidate observations, not damage labels. Source and Dhunche remain cloud/snow limited. The purple 24 Aug frame is Sentinel-1 VV/VH/contrast false colour—not surface colour.'}</p>
          </section>

          <section className="story-section story-step story-wide olmo-proof">
            <p className="story-kicker">03 · OLMoEarth — <em>{ko ? '인공지능은 실제로 무엇을 계산했나' : 'the executed AI, not the interface'}</em></p>
            <h2>{ko ? '계산한 것과 계산하지 않은 것을 장부로 나눴다' : 'Separate what the AI computed from what the product merely proposes'}</h2>
            <div className="ai-chain" role="img" aria-label="Satellite tensors pass through a frozen OlmoEarth encoder into embeddings and then task-specific evidence">
              <article><b>INPUT</b><strong>S1 + S2 × TIME</strong><span>{ko ? '밴드·날짜·공간창 봉인' : 'sealed bands, dates and windows'}</span></article><i>→</i>
              <article><b>FROZEN AI</b><strong>OLMoEarth v1</strong><span>{ko ? '학습된 지구 표현 인코더' : 'pretrained Earth encoder'}</span></article><i>→</i>
              <article><b>OUTPUT</b><strong>768 × 64 × 64</strong><span>{ko ? '40 m 공간 토큰 격자' : '40 m spatial-token grid'}</span></article><i>→</i>
              <article><b>USE</b><strong>PROBE · Δ · RETRIEVE</strong><span>{ko ? '분할·변화·유사사건' : 'segment, compare, retrieve'}</span></article>
            </div>
            <div className="ai-run-ledger">
              {(scenario?.research.ai_run_ledger ?? []).map((run) => (
                <article key={run.id} className={`ai-run ${run.state.toLowerCase()}`}>
                  <header><span>{run.state.replace(/_/g, ' ')}</span><strong>{run.model}</strong></header>
                  <dl>
                    <div><dt>{ko ? '입력' : 'INPUT'}</dt><dd>{run.input}</dd></div>
                    <div><dt>{ko ? '실제 출력' : 'ACTUAL OUTPUT'}</dt><dd>{run.output}</dd></div>
                    <div><dt>{ko ? '허용 주장' : 'ALLOWS'}</dt><dd>{run.allows}</dd></div>
                    <div><dt>{ko ? '금지 주장' : 'FORBIDS'}</dt><dd>{run.forbids}</dd></div>
                  </dl>
                </article>
              ))}
            </div>
            <div className="olmo-proof-grid">
              <article className="ready"><b>01</b><span>{ko ? '네팔 기준 표현' : 'NEPAL BASELINE'}</span><strong>15 × 768×64×64 READY</strong><p>{ko ? 'baseline·placebo 2개 × 5앵커의 S1+S2×4기간 임베딩 GeoTIFF를 봉인했다. 지금도 유사지역 검색·선형 probe·사전/사후 비교 기준으로 사용 가능.' : 'Sealed embedding GeoTIFFs for baseline plus two placebos across five anchors—usable now for retrieval, linear probes and a pre/post reference.'}</p></article>
              <article className="win"><b>02</b><span>{ko ? '확증 전이' : 'CONFIRMATORY TRANSFER'}</span><strong>{transfer ? `${transfer.wins_reuse_vs_raw_strong}/${transfer.regions} WINS · +${transfer.absolute_gap.toFixed(3)}` : 'LOADING'}</strong><p>{transfer ? `Frozen reuse region-macro ${transfer.reuse_region_macro.toFixed(3)} vs raw UNet3D ${transfer.raw_strong_region_macro.toFixed(3)} (${transfer.relative_gain_pct.toFixed(1)}% relative).` : ''}</p></article>
              <article className="pilot"><b>03</b><span>{ko ? '사건 변화 파일럿' : 'EVENT-DELTA PILOT'}</span><strong>2 / 3 STRONG</strong><p>{ko ? '관련 S2-only M66에서 Hokkaido·Hiroshima 분리, Dominica 약함. 가능성 증거이지 네팔 검증값은 아님.' : 'Related S2-only M66 separated Hokkaido and Hiroshima; Dominica was weak. Feasibility evidence, not Nepal validation.'}</p></article>
              <article className="wait"><b>04</b><span>{ko ? '네팔 live 변화' : 'NEPAL LIVE CHANGE'}</span><strong>{providerSyncBlocked ? 'S1 5/5 · SYNC WAIT' : 'S1 3/4 · S2 4/4'}</strong><p>{ko ? (providerSyncBlocked ? '공식 footprint는 5개 anchor를 모두 덮지만 provider가 8/28 장면을 아직 선택하지 않는다. 픽셀 전에는 임베딩도 없다.' : '마지막 S1 기간 전에는 Δz를 만들지 않음. 모델 전체 실패가 아니라 한 live action의 입력 대기.') : (providerSyncBlocked ? 'The official footprint covers all five anchors, but the provider has not selected the 28 Aug scene. No pixels means no embedding yet.' : 'No Δz before the final S1 period. This is one live action waiting for input—not failure of the representation.')}</p></article>
            </div>
            <div className="transfer-bars"><span>Frozen OLMo reuse</span><i style={{ width: '100%' }} /><b>{transfer?.reuse_region_macro.toFixed(3) ?? '—'}</b><span>Raw UNet3D</span><i style={{ width: transfer ? `${100 * transfer.raw_strong_region_macro / transfer.reuse_region_macro}%` : '0%' }} /><b>{transfer?.raw_strong_region_macro.toFixed(3) ?? '—'}</b></div>
            <div className="ai-not-code"><b>{ko ? '웹 코드의 역할' : 'WHAT THE WEB CODE DOES'}</b><p>{ko ? '이 화면 자체는 인공지능이 아니다. 지도와 흐르는 입자, 타임라인은 사람이 만든 화면이고, 인공지능이 계산한 것은 서버에서 봉인된 파일들이다. 화면은 그 파일을 읽어 허용된 문장만 보여준다. 이 구분을 흐리면 “AI가 재해를 예측했다”는 오해가 생긴다.' : 'The interface does not imitate a model. It reads manifests, GeoTIFFs, evaluation JSON and SHA-256 records produced by the Python/GPU pipeline, then displays only allowed claims. The map and WASM particles are UI—not AI output.'}</p></div>
            <p>{ko ? '이 모델을 쓰는 이유는 실험 결과 때문이다. 앨런인공지능연구소가 공개한 올모어스(OLMoEarth)는 위성 시계열을 읽어 장소마다 768개의 숫자를 내놓는다. 산사태 지도가 있는 세계 8개 지역에서 이 숫자를 그대로 재사용한 방법이, 원본 영상으로 처음부터 학습한 강한 비교 모델을 6곳에서 이겼다. 인도네시아와 필리핀 이토곤에서는 졌다. 연구팀은 이긴 곳만 보고하지 않고 진 곳을 함께 남겼다. “항상 좋다”가 아니라 “어디까지 통하는가”를 잰 것이다. 다만 이 결과는 원본 대비 우위일 뿐, 같은 조건의 다른 지구관측 모델(프레스토)과 견주기 전까지는 올모어스만의 고유한 우위라고 말하지 않는다.' : 'Why this matters: an EO model is not automatically better on every EO task. Under matched public regions and decoder contracts, frozen representation reuse beat a strong raw time-series model in six of eight external regions, while preserving the Indonesia and Itogon non-wins. It measures where transfer holds rather than claiming universal superiority. This is Olmo reuse versus raw baselines—not Olmo-specific superiority until a second GeoFM control such as Presto is run under the same input contract.'}</p>
          </section>

          <section className="story-section story-step story-wide">
            <p className="story-kicker">04 · {ko ? '결합 실험' : 'THE FUSION EXPERIMENT'} — <em>{ko ? '다음 단계, 물리와의 결합' : 'a falsifiable loop, not decoration'}</em></p>
            <h2>{ko ? '인공지능이 후보를 내고, 물리 계산이 설명하고, 다음 위성이 검증한다' : 'Olmo proposes. Physics explains. Satellites falsify.'}</h2>
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

          <section className="story-section story-step story-wide response-section">
            <p className="story-kicker">05 · {ko ? '행성 대응 스택' : 'PLANETARY RESPONSE STACK'} — <em>{ko ? '모델 하나로 지구를 설명하지 않는다' : 'models with distinct jobs'}</em></p>
            <h2>{ko ? '올모어스는 중심이되, 혼자가 아니다' : 'Keep Olmo at the centre—without asking it to explain the whole planet'}</h2>
            <div className="response-stack" role="img" aria-label="Observe, represent, explain, and impact layers in the planetary response stack">
              <article className="observe"><b>01 · OBSERVE</b><strong>{ko ? '무엇이 실제로 도착했나' : 'What actually arrived'}</strong><span>S1/S2 · DEM · weather · USGS/GDACS · field reports</span><em>{ko ? '픽셀·시간·footprint 봉인' : 'seal pixels, time, footprint'}</em></article>
              <i>→</i>
              <article className="represent"><b>02 · REPRESENT</b><strong>{ko ? '어디가 평소와 다른가' : 'Where is different'}</strong><span>OLMoEarth core · Presto control · Prithvi/Clay/TerraMind</span><em>{ko ? '변화·유사사건 후보' : 'change + analogue candidates'}</em></article>
              <i>→</i>
              <article className="explain"><b>03 · EXPLAIN</b><strong>{ko ? '물리적으로 가능한가' : 'Is it physically plausible'}</strong><span>r.avaflow · D-Claw · hydrology · sensor operator</span><em>{ko ? 'runout 앙상블·불확실성' : 'runout ensemble + uncertainty'}</em></article>
              <i>→</i>
              <article className="impact"><b>04 · IMPACT</b><strong>{ko ? '무엇을 먼저 확인할까' : 'What should be checked first'}</strong><span>roads · settlements · clinics · WASH · population</span><em>{ko ? '검토 우선순위, 피해 확정 아님' : 'review priority, not a verdict'}</em></article>
            </div>
            <div className="model-contracts">
              <article><b>CORE</b><strong>OLMoEarth</strong><p>{ko ? 'S1+S2 다기간 공통 표현, Nepal Δ·검색·frozen transfer의 중심.' : 'The shared S1+S2 temporal representation for Nepal delta, retrieval and frozen transfer.'}</p></article>
              <article><b>CONTROL</b><strong>Presto / classical</strong><p>{ko ? '같은 cube에서 OLMo 고유효과와 저비용 baseline을 판정.' : 'Same-cube controls for Olmo-specific value and low-cost baselines.'}</p></article>
              <article><b>EXTEND</b><strong>Prithvi · Clay · TerraMind</strong><p>{ko ? '광학·센서 shift·missing modality의 독립 보강. 생성 영상은 관측이 아님.' : 'Independent optical, sensor-shift and missing-modality evidence. Generated pixels are never observations.'}</p></article>
              <article><b>FORECAST</b><strong>Aurora · Flood Hub</strong><p>{ko ? '강수·대기·유량 forcing을 residual로 결합. OLMo 입력밴드로 가장하지 않음.' : 'Weather and discharge forcing joins as residual context—not disguised as Olmo input bands.'}</p></article>
            </div>
            <p className="story-caption">{ko ? '권장 결합은 embedding late fusion과 candidate cascade다. 새 센서나 생성 픽셀을 OLMo 입력처럼 넣는 것은 band·GSD·시간 계약을 다시 검증하기 전에는 금지한다.' : 'The recommended joins are embedding late fusion and a candidate cascade. New sensors or generated pixels do not enter Olmo as if they were canonical bands without a new band/GSD/time contract.'}</p>
          </section>

          <section className="story-section story-step story-wide candidate-section">
            <p className="story-kicker">06 · {ko ? '후보 공장' : 'CANDIDATE FACTORY'} — <em>{ko ? '탐지가 사건이 되기까지' : 'from detection to incident'}</em></p>
            <h2>{ko ? '위험 지도를 칠하는 대신, 확인할 후보 목록을 만든다' : 'Turn map pixels into candidates with a next action'}</h2>
            <div className="candidate-funnel">
              <article><b>C0</b><strong>TRIGGER</strong><span>USGS · GDACS · weather · report</span><em>AOI / acquisition queue</em></article>
              <article><b>C1</b><strong>OBSERVE</strong><span>footprint · pixels · masks</span><em>{ko ? '없으면 거부' : 'reject if absent'}</em></article>
              <article><b>C2</b><strong>REPRESENT</strong><span>OLMo Δ · retrieval · indices</span><em>{ko ? '변화 후보' : 'change candidates'}</em></article>
              <article><b>C3</b><strong>CONSENSUS</strong><span>second GeoFM · conflict</span><em>{ko ? '합의/기권' : 'agree / abstain'}</em></article>
              <article><b>C4</b><strong>PHYSICS</strong><span>runout feasible fraction</span><em>{ko ? '원인 경로 검증' : 'test causal route'}</em></article>
              <article><b>C5</b><strong>EXPOSURE</strong><span>road · clinic · WASH · people</span><em>{ko ? '검토 순위' : 'review priority'}</em></article>
              <article><b>C6</b><strong>REVIEW</strong><span>official · analyst · field</span><em>incident / rejected</em></article>
            </div>
            <div className="candidate-scorecard">
              <span><b>CHANGE</b>OLMo · classical · second GeoFM</span>
              <span><b>PHYSICS</b>ensemble agreement</span>
              <span><b>EXPOSURE</b>people · roads · facilities</span>
              <span><b>FRESHNESS</b>acquisition · provider · report latency</span>
              <span><b>EVIDENCE</b>sources · conflicts · review</span>
            </div>
            <p>{ko ? '이 구조는 야생동물 보호 현장의 어스레인저와 해상 감시 서비스 스카이라이트가 쓰는 방식을 재해 대응에 옮긴 것이다. 두 시스템 모두 기계가 “탐지”를 내고 사람이 “사건”으로 확정한다. 여기서도 인공지능 점수가 높아도 물리적으로 불가능하거나 관측이 없으면 사건이 되지 않는다. 반대로 데이터가 늦게 오면 모델이 실패한 것이 아니라 첫 관문에서 정직하게 멈춘 것이다.' : 'This adapts EarthRanger’s event-to-incident and Skylight’s detection-to-analyst-review grammar. A high Olmo score cannot become an incident without observation and physical plausibility. Provider latency stops the record honestly at C1; it is not mislabeled as model failure.'}</p>
          </section>

          <section className="story-section story-step story-wide human-section">
            <p className="story-kicker">07 · {ko ? '인간 영향' : 'HUMAN IMPACT'} — <em>{ko ? '사람의 피해를 위성으로 읽지 않는다' : 'verification priority, not diagnosis'}</em></p>
            <h2>{ko ? '위성이 계산하는 것은 고통이 아니라 끊긴 길이다' : 'Satellites do not read suffering. They can expose broken access.'}</h2>
            <div className="verified-impact">
              <header><span>{ko ? 'WHO · 현재 확인 범위' : 'WHO · CURRENT VERIFIED SCOPE'}</span><strong>{ko ? '이번 사건은 이미 보건 운영 문제다' : 'This is already a health-operations problem'}</strong></header>
              <div><article><b>3</b><span>{ko ? '보건소 전파' : 'health posts fully damaged'}</span></article><article><b>1</b><span>{ko ? '병원 부분 손상' : 'hospital partially damaged'}</span></article><article><b>2</b><span>{ko ? '병원 접근 영향' : 'hospital access affected'}</span></article><article><b>~10K</b><span>{ko ? '즉시 구호 필요 가구' : 'households needing relief'}</span></article></div>
              <a href="https://www.who.int/nepal/emergencies/2026-rasuwa-flash-floods" target="_blank" rel="noreferrer">WHO Nepal · 2026 Rasuwa flash floods ↗</a>
            </div>
            <div className="impact-lenses">
              <article><b>ACCESS</b><strong>{ko ? '병원까지 갈 수 있는가' : 'Can care still be reached'}</strong><p>{ko ? '도로·교량 변화 후보를 network graph에 반영해 facility travel-time 전후 범위를 계산.' : 'Propagate road and bridge candidates through a network graph to estimate before/after facility travel-time ranges.'}</p></article>
              <article><b>WASH</b><strong>{ko ? '어디를 먼저 검사할까' : 'Where should teams sample first'}</strong><p>{ko ? '침수/토석 후보와 정착지·급수 자산의 교차를 EWARS 현장 확인 queue로 전달.' : 'Intersect water/debris candidates with settlements and water assets, then hand a field-verification queue to EWARS workflows.'}</p></article>
              <article><b>EXPOSURE</b><strong>{ko ? '몇 명이 범위 안에 있는가' : 'Who may be in the footprint'}</strong><p>{ko ? 'WorldPop과 runout envelope를 겹치되 현재 체류 인구가 아닌 model-based range와 불확실성으로 표시.' : 'Intersect WorldPop with the runout envelope, while reporting a model-based range and uncertainty—not an exact live population.'}</p></article>
              <article><b>CONTINUITY</b><strong>{ko ? '대체진료 경로는 무엇인가' : 'What is the alternate-care route'}</strong><p>{ko ? 'HeRAMS 시설 상태·Alternate Care Site·구호창고를 최신 도로 후보와 함께 우선순위화.' : 'Rank HeRAMS facility status, alternate-care sites and relief depots against the freshest road evidence.'}</p></article>
            </div>
            <div className="safety-line"><b>{ko ? '금지선' : 'SAFETY BOUNDARY'}</b><span>{ko ? '이 시스템은 사망자 수나 질병 발생을 위성이나 소셜미디어에서 추론하지 않는다. 그것은 세계보건기구와 현장 조사의 몫이다. 위성이 할 수 있는 것은 도로와 다리가 끊겨 보건소까지 가는 시간이 얼마나 늘었는지를 계산해 확인 순서를 정하는 일이다.' : 'Never infer individual health, death or disease from satellite or social media. Public posts enter only through official embed/API or user-curated URLs, and never become ground-truth loss counts.'}</span></div>
            <div className="evidence-stream"><span>OFFICIAL</span><i>→</i><span>RELIEFWEB / GDACS</span><i>→</i><span>CURATED PUBLIC POST</span><i>→</i><span>FIELD REVIEW</span></div>
          </section>

          <section className="story-section story-step story-wide priority-section">
            <p className="story-kicker">08 · {ko ? 'AI 엔지니어 우선순위' : 'ENGINEERING PRIORITIES'} — <em>{ko ? '영향으로 이어지는 경로' : 'the path to impact'}</em></p>
            <h2>{ko ? '지금 만들어야 할 네 가지' : 'The four builds that matter next'}</h2>
            <div className="priority-stack">
              <article><b>P0 · NOW</b><strong>{ko ? '27창 회랑과 공식 범위 닫기' : 'Close 27-window corridor + official extent'}</strong><p>{liveDelta ? (ko ? '동일 27개 창의 baseline/live S1+S2를 모두 봉인해 토큰 Δ를 비교하고, USGS·UNOSAT 침수/토석 범위와 블라인드 대조한다. 100창 S2 후보는 수색 큐로만 유지한다.' : 'Seal matched baseline/live S1+S2 across the same 27 windows, compare token deltas, then blind-check against USGS/UNOSAT extent. Keep the 100-window S2 scan as a search queue only.') : ko ? 'Bidur/Rasuwagadhi mask를 동결하고 5/5 live cube를 봉인한다.' : 'Freeze Bidur/Rasuwagadhi masks and seal the 5/5 live cube.'}</p><em>VALUE · ground truth, reproducibility</em></article>
              <article><b>P1 · 1 WEEK</b><strong>{ko ? 'OLMo change + retrieval 본실험' : 'Olmo change + retrieval experiment'}</strong><p>{ko ? '고전 NDWI/SAR 변화탐지, OLMo Δz, gate-aware abstention을 동일 recall에서 비교. Nepal query로 SEN12 6,834 patch 유사사건 검색.' : 'Compare classical NDWI/SAR change, Olmo Δz and gate-aware abstention at matched recall; query 6,834 SEN12 patches with the Nepal representation.'}</p><em>VALUE · AI2 relevance, triage speed</em></article>
              <article><b>P2 · 2–3 WEEKS</b><strong>{ko ? '물리–관측 앙상블' : 'Physics–observation ensemble'}</strong><p>{ko ? 'Copernicus DEM/GLO-30에서 r.avaflow 파라미터 sweep, D-Claw 소수 독립 run, S1/S2 semantic operator로 실측 일치도 순위화.' : 'Sweep r.avaflow over Copernicus DEM/GLO-30, run a small independent D-Claw check and rank outputs through S1/S2 semantic observation operators.'}</p><em>VALUE · causal plausibility, uncertainty</em></article>
              <article><b>P3 · 4–6 WEEKS</b><strong>{ko ? '빠른 surrogate + 운영 UI' : 'Fast surrogate + operations UI'}</strong><p>{ko ? '물리 앙상블로 neural operator/emulator를 학습해 웹에서 scenario를 재생. 실제 OSM 경로·위성 pass·OLMo evidence를 EarthRanger식 incident ledger와 연결.' : 'Train a neural operator/emulator on the physics ensemble for interactive scenarios; join OSM routes, satellite passes and Olmo evidence into an EarthRanger-style incident ledger.'}</p><em>VALUE · scalable decision support</em></article>
            </div>
          </section>

          <section className="story-section story-step story-boundary">
            <p className="story-kicker">09 · {ko ? '다음 게이트와 출처' : 'NEXT GATE + SOURCES'} — <em>{ko ? '기다림의 기록' : 'progress with boundaries'}</em></p>
            <h2>{scenario?.corridor_sealed ? (ko ? '다음 관문은 더 넓은 평시 기준과 독립 피해경계다' : 'The next gate is a deeper ordinary baseline plus independent event extent') : liveDelta ? (ko ? '다음 관문은 27개 창의 계약교정 재계산이다' : 'The next gate is a contract-correct 27-window rerun') : ko ? (providerSyncBlocked ? '다음 판정을 막고 있는 것은 위성이 아니라 지형보정본 한 장이다' : '다음 레이더가 판정을 연다') : (providerSyncBlocked ? 'The next gate is provider sync—not another satellite' : 'The next S1 pass opens the live decision')}</h2>
            <p>{scenario?.corridor_sealed ? (ko ? 'Sentinel-1 dB 전처리를 포함해 27개 창 × placebo·baseline·live, 총 81개 OLMoEarth 임베딩을 다시 계산했다. Devighat·Bidur가 검토 큐 상단이지만 최대 초과 토큰은 0.415%이고 모든 창의 사건 평균은 단일 평시 전이 평균보다 작다. 따라서 현재 결과는 음성에 가까운 screening이며, 더 많은 평시 전이와 독립 피해경계 없이는 탐지·피해·확률로 승격하지 않는다.' : 'We recomputed 81 OLMoEarth embeddings—27 windows across placebo, baseline and live—with the required Sentinel-1 dB transform. Devighat and Bidur top the review queue, but the maximum exceedance is only 0.415%, and event mean change is below the single ordinary-transition mean in every window. This is a mostly negative screen, not a detection, until a deeper ordinary baseline and independent event extent are available.') : liveDelta ? (ko ? '이전 5-anchor S1+S2 결과는 Sentinel-1 dB 변환 누락으로 폐기했다. 동일 27개 창을 계약에 맞춰 재계산해야 한다.' : 'The legacy five-anchor S1+S2 result is superseded because Sentinel-1 dB conversion was missing. The same 27 windows must be recomputed under the correct contract.') : ko ? '다음 후보도 실제 footprint와 봉인 계약을 통과해야 한다.' : 'The next candidate must pass actual footprint containment and the sealed input contract.'}</p>
            <div className="story-schedule">{(scenario?.scheduled_scenes ?? []).map((scene) => <div key={scene.id ?? scene.acquired_at} className={scene.state === 'missed_coverage' ? 'missed' : ''}><b>{shortSensor(scene.sensor)}</b><span>{kstStamp(scene.acquired_at)} KST</span><em>{scene.state.replace(/_/g, ' ').toUpperCase()}</em></div>)}</div>
            <div className="story-sources"><a href="https://www.usgs.gov/programs/landslide-hazards/science/2026-nepal-debris-avalanche-and-flash-flood" target="_blank" rel="noreferrer">USGS event assessment ↗</a><a href="https://www.who.int/nepal/emergencies/2026-rasuwa-flash-floods" target="_blank" rel="noreferrer">WHO health response ↗</a><a href="https://allenai.org/blog/olmoearth-embeddings" target="_blank" rel="noreferrer">Ai2 embedding workflow ↗</a><a href="https://research.google/blog/planetary-prediction-engine-automating-global-models-via-earth-ai/" target="_blank" rel="noreferrer">Planetary Prediction Engine ↗</a><a href="https://doi.org/10.5194/gmd-18-9879-2025" target="_blank" rel="noreferrer">r.avaflow v4 ↗</a><a href="https://claw.code-pages.usgs.gov/dclaw/" target="_blank" rel="noreferrer">USGS D-Claw ↗</a><a href="https://planetarycomputer.microsoft.com/docs/quickstarts/using-the-data-api/" target="_blank" rel="noreferrer">Planetary Computer STAC ↗</a><a href="https://mapping.emergency.copernicus.eu/activations/EMSR927/" target="_blank" rel="noreferrer">CEMS EMSR927 ↗</a></div>
            <p className="story-outro">{scenario?.research.integration_disclaimer}</p>
          </section>
        </div>
        );
      })()}

      {lightbox && (
        <div className="lightbox" role="dialog" aria-modal="true" aria-label={lightbox.title} onClick={(e) => { if (e.target === e.currentTarget) setLightbox(null); }}>
          <div className="lb-panel">
            <header><div><strong>{lightbox.title}</strong>{lightbox.sub && <small>{lightbox.sub}</small>}</div><button onClick={() => setLightbox(null)} aria-label="Close">×</button></header>
            {lbExtra === null ? (
              <div className="story-swipe lb-swipe" style={{ ['--swipe' as string]: `${lbSwipe}%` }}>
                <img src={lightbox.after} alt={lightbox.afterLabel} />
                <div className="swipe-clip"><img src={lightbox.before} alt={lightbox.beforeLabel} /></div>
                <div className="swipe-bar" /><span className="swipe-label pre">{lightbox.beforeLabel}</span><span className="swipe-label post">{lightbox.afterLabel}</span>
                <input type="range" min={0} max={100} value={lbSwipe} aria-label="Compare" onChange={(e) => setLbSwipe(Number(e.target.value))} />
              </div>
            ) : (
              <div className="lb-single"><img src={lightbox.extra![lbExtra].src} alt={lightbox.extra![lbExtra].label} /><span className="swipe-label post">{lightbox.extra![lbExtra].label}</span></div>
            )}
            <footer>
              <button className={lbExtra === null ? 'is-active' : ''} onClick={() => setLbExtra(null)}>BEFORE ⇄ AFTER</button>
              {(lightbox.extra ?? []).map((x, i) => <button key={x.src} className={lbExtra === i ? 'is-active' : ''} onClick={() => setLbExtra(i)}>{x.label}</button>)}
              <span className="lb-tip">drag the handle · ESC closes</span>
            </footer>
          </div>
        </div>
      )}

      <div className="provenance-stamp">DATA SNAPSHOT {scenario?.generated_at.slice(0, 16).replace('T', ' ') ?? '—'} UTC · OSM ODbL · ESA COPERNICUS · PLANET DISASTER DATA (CC-BY-NC-4.0)</div>
    </main>
  );
}
