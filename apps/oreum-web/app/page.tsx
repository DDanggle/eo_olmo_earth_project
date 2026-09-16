'use client';
// 오름 변화 추적 지도 — 한 화면, 라우트 분할 없음. 클릭 → 즉시 전후 프레임.
// 모든 숫자는 public/data/summary.json 에서만 온다 (scripts/verify-assets.mjs 가 대조).
import { Map as MapLibreMap, NavigationControl, ScaleControl, AttributionControl, setWorkerUrl } from 'maplibre-gl';
import type { MapLayerMouseEvent } from 'maplibre-gl';
import type { Feature, FeatureCollection, Point } from 'geojson';
import { useEffect, useMemo, useRef, useState } from 'react';
import 'maplibre-gl/dist/maplibre-gl.css';

type Props = {
  oreum_id: string; name: string; verdict: 'scored' | 'abstain'; abstain_reason: string | null;
  rank: number | null; event_flag_frac: number | null; event_n_flag: number | null; null_flag_frac: number | null;
  persistent_tokens: number | null; event_only_tokens: number | null;
  valid_token_fraction: Record<string, number> | null; observable_by_year: Record<string, boolean> | null;
  tile: string | null; secondary_20m: { verdict: string; event_flag_frac: number | null; rank: number | null } | null;
  korea_tags: Record<string, string[]>;
};
type Summary = {
  frame_a_total: number; scored: number; abstain: number; sites_with_any_event_flag: number;
  threshold_p99: number; flag_rate_pooled: Record<string, number>; pairs: Record<string, [string, string]>;
  token_ground_m: number; class_breaks: number[]; class_colors: string[];
  anchor_dates: Record<string, Record<string, string>>; contract_sha256: string | null;
  secondary_20m: { scored: number; flag_rate_pooled: Record<string, number> } | null;
  korea_layers: Record<string, { name: string; n: number }> | null;
};

// Next 번들러가 MapLibre 의 인라인 워커를 깨뜨려 GeoJSON 소스가 영원히 로드되지 않는다(실측: 소스에 243점,
// isSourceLoaded false). 네팔과 같이 워커를 정적 파일로 둔다.
setWorkerUrl('/maplibre-gl-worker.mjs');

const YEARS = ['2023', '2024', '2025', '2026'];
const pct = (x: number | null | undefined, d = 1) => (x == null ? '—' : `${(x * 100).toFixed(d)}%`);

export default function Page() {
  const mapRef = useRef<HTMLDivElement>(null);
  const map = useRef<MapLibreMap | null>(null);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [fc, setFc] = useState<FeatureCollection<Point, Props> | null>(null);
  const [sel, setSel] = useState<Props | null>(null);
  const [showAbstain, setShowAbstain] = useState(true);
  const [showKorea, setShowKorea] = useState(false);
  const [frameYear, setFrameYear] = useState<'2025' | '2026'>('2026');

  useEffect(() => {
    fetch('/data/summary.json').then(r => r.json()).then(setSummary);
    fetch('/data/oreum.geojson').then(r => r.json()).then(setFc);
  }, []);

  // 색계급 표현식: summary.class_breaks 하나만 읽는다 → 범례와 절대 어긋나지 않는다.
  const colorExpr = useMemo(() => {
    if (!summary) return '#999';
    const { class_breaks: b, class_colors: c } = summary;
    return ['case', ['==', ['get', 'verdict'], 'abstain'], 'rgba(0,0,0,0)',
      ['step', ['coalesce', ['get', 'event_flag_frac'], 0], c[0], b[0], c[1], b[1], c[2], b[2], c[3], b[3], c[4]]] as unknown as string;
  }, [summary]);

  useEffect(() => {
    if (!mapRef.current || map.current || !fc || !summary) return;
    const m = new MapLibreMap({
      container: mapRef.current, center: [126.55, 33.38], zoom: 9.6, attributionControl: false,
      style: {
        version: 8, sources: {
          esri: { type: 'raster', tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'], tileSize: 256, attribution: 'Esri World Imagery' },
        },
        layers: [{ id: 'esri', type: 'raster', source: 'esri', paint: { 'raster-saturation': -0.35, 'raster-brightness-max': 0.85 } }],
      },
    });
    m.addControl(new NavigationControl({ showCompass: false }), 'bottom-right');
    m.addControl(new ScaleControl({ unit: 'metric' }), 'bottom-right');
    m.addControl(new AttributionControl({ compact: true }), 'bottom-right');
    m.on('load', () => {
      m.addSource('oreum', { type: 'geojson', data: fc });
      // abstain: 색이 아니라 패턴(점선 링) — "점수 없음 ≠ 변화 없음"
      m.addLayer({ id: 'abstain', type: 'circle', source: 'oreum', filter: ['==', ['get', 'verdict'], 'abstain'],
        paint: { 'circle-radius': 7, 'circle-color': 'rgba(0,0,0,0)', 'circle-stroke-color': '#c6cfcb', 'circle-stroke-width': 1.6, 'circle-stroke-opacity': 0.9 } });
      m.addLayer({ id: 'scored', type: 'circle', source: 'oreum', filter: ['==', ['get', 'verdict'], 'scored'],
        paint: { 'circle-radius': ['interpolate', ['linear'], ['zoom'], 9, 6, 13, 11], 'circle-color': colorExpr,
          'circle-stroke-color': '#14201d', 'circle-stroke-width': 0.8, 'circle-opacity': 0.92 } });
      const pick = (e: MapLayerMouseEvent) => { const f = e.features?.[0] as Feature<Point, Props> | undefined; if (f) setSel(f.properties); };
      for (const id of ['scored', 'abstain']) {
        m.on('click', id, pick);
        m.on('mouseenter', id, () => (m.getCanvas().style.cursor = 'pointer'));
        m.on('mouseleave', id, () => (m.getCanvas().style.cursor = ''));
      }
    });
    map.current = m;
    if (process.env.NODE_ENV !== 'production') (window as unknown as { __m: MapLibreMap }).__m = m;
  }, [fc, summary, colorExpr]);

  useEffect(() => { if (map.current?.getLayer('abstain')) map.current.setLayoutProperty('abstain', 'visibility', showAbstain ? 'visible' : 'none'); }, [showAbstain]);

  const ranking = useMemo(() => (fc?.features ?? []).map(f => f.properties).filter(p => p.verdict === 'scored').sort((a, b) => (a.rank ?? 1e9) - (b.rank ?? 1e9)).slice(0, 12), [fc]);
  const pair = summary?.pairs.event ?? ['2025', '2026'];

  return (
    <div className="app-shell">
      <div ref={mapRef} className="map-stage" />

      <header className="topbar">
        <div className="toggles">
          <button className="toggle" aria-pressed={showAbstain} onClick={() => setShowAbstain(v => !v)}>관측 불가 {summary ? summary.abstain : ''}</button>
          <button className="toggle" aria-pressed={showKorea} onClick={() => setShowKorea(v => !v)}>한국 지정</button>
        </div>
        <div className="brand">
          <div className="brand-mark" />
          <div>
            <p className="eyebrow">Jeju · OlmoEarth v1 Base · frozen</p>
            <h1>오름 변화 <span>추적</span></h1>
          </div>
        </div>
      </header>

      <aside className="rail rail-left">
        <p className="eyebrow">무엇을 보고 있나</p>
        <h2>오름 {summary?.frame_a_total ?? '…'}곳, 연간 변화의 읽는 순서</h2>
        <p>{pair[0]}년과 {pair[1]}년 같은 계절의 Sentinel-2 장면을 같은 임베딩 모델에 넣고, 40 m 토큰마다 두 해 사이의 거리를 잽니다.
          문턱은 <strong>전년도 쌍(2024→2025)의 평시 변화 p99</strong> 하나입니다.</p>
        <div className="stat-row">
          <div className="stat"><b>{summary?.scored ?? '…'}</b><small>채점됨</small></div>
          <div className="stat abstain"><b>{summary?.abstain ?? '…'}</b><small>관측 불가</small></div>
          <div className="stat flag"><b>{summary?.sites_with_any_event_flag ?? '…'}</b><small>문턱 넘은 곳</small></div>
        </div>
        <div className="claim">이건 <strong>읽는 순서</strong>입니다. 훼손을 탐지한 것이 아니고, 원인이나 시설 종류를 말하지 않습니다.
          점선 원은 두 해 중 하나가 구름이라 <strong>보지 못한</strong> 오름입니다 — 변화가 없다는 뜻이 아닙니다.</div>
        {summary && (
          <div className="legend">
            <p className="eyebrow">사건 쌍에서 문턱을 넘은 토큰 비율</p>
            {summary.class_colors.map((c, i) => (
              <div className="legend-row" key={c}><span className="swatch" style={{ background: c }} />
                {i === 0 ? `< ${pct(summary.class_breaks[0], 1)}` : i === 4 ? `≥ ${pct(summary.class_breaks[3], 1)}` : `${pct(summary.class_breaks[i - 1], 1)} – ${pct(summary.class_breaks[i], 1)}`}</div>
            ))}
            <div className="legend-row"><span className="swatch abstain" /> 관측 불가 (abstain)</div>
            <p style={{ fontSize: 11 }}>귀무 깃발율 {pct(summary.flag_rate_pooled.null_temporal_primary, 2)} (구성상 ≈1%) · 부귀무 {pct(summary.flag_rate_pooled.null_temporal_secondary, 2)} · 사건 {pct(summary.flag_rate_pooled.event, 2)}</p>
          </div>
        )}
      </aside>

      <aside className="rail rail-right">
        <p className="eyebrow">상위 12 — 먼저 볼 곳</p>
        {ranking.map(p => (
          <button key={p.oreum_id} className="toggle" style={{ display: 'flex', width: '100%', justifyContent: 'space-between', marginBottom: 6 }} onClick={() => setSel(p)}>
            <span>#{p.rank} {p.name}{(p.persistent_tokens ?? 0) >= 20 && <span className="badge abstain" style={{ marginLeft: 6 }} title="사건·귀무 양쪽에서 깃발 — 연간 변화보다 지속 인공물일 가능성">지속 {p.persistent_tokens}</span>}</span><span className="mono">{pct(p.event_flag_frac)}</span>
          </button>
        ))}
        <p style={{ fontSize: 11 }}>비율은 그 오름 창(2.56 km)의 유효 토큰 중 문턱 초과분. 양쪽 해에 다 깃발이 선 토큰(지속 인공물 후보)은 상세에서 따로 보입니다.</p>
      </aside>

      {sel && (
        <section className="detail">
          <header>
            <div>
              <h3>{sel.name} <span className="mono" style={{ color: 'var(--muted)', fontSize: 11 }}>{sel.oreum_id}</span></h3>
              <span className={`badge ${sel.verdict}`}>{sel.verdict === 'scored' ? `채점 · 순위 #${sel.rank}` : '관측 불가 · 점수 없음'}</span>
              {sel.verdict === 'scored' && (sel.event_n_flag ?? 0) > 0 && <span className="badge flag" style={{ marginLeft: 6 }}>문턱 초과 {pct(sel.event_flag_frac)} · {sel.event_n_flag} 토큰</span>}
            </div>
            <button className="close" onClick={() => setSel(null)} aria-label="닫기">×</button>
          </header>
          {sel.verdict === 'abstain' && <p style={{ margin: '8px 0 0', fontSize: 12.5, color: 'var(--muted)' }}>{sel.abstain_reason}. 이 오름은 분모에 남고 점수는 만들지 않습니다.</p>}
          <div className="frames">
            <figure><img src={`/data/frames/${sel.oreum_id}_${pair[0]}.jpg`} alt="" /><figcaption>{pair[0]} · 사건 쌍 기준</figcaption></figure>
            <figure onClick={() => setFrameYear(y => (y === '2026' ? '2025' : '2026'))} style={{ cursor: 'pointer' }}>
              <img src={`/data/frames/${sel.oreum_id}_${frameYear}.jpg`} alt="" /><figcaption>{frameYear} · 눌러서 전후 전환</figcaption></figure>
            <figure><img src={`/data/frames/${sel.oreum_id}_delta.png`} alt="" style={{ background: '#0f1d1a' }} /><figcaption>Δz 토큰 맵 · 투명 = 유효하지 않은 토큰</figcaption></figure>
          </div>
          <p style={{ margin: '10px 0 0', fontSize: 12, color: 'var(--muted)' }}>
            연도별 유효 토큰: {YEARS.map(y => `${y} ${pct(sel.valid_token_fraction?.[y], 0)}`).join(' · ')}
            {sel.verdict === 'scored' && <> · 귀무 쌍 깃발 {pct(sel.null_flag_frac)} · 지속 토큰 {sel.persistent_tokens} · 사건에만 {sel.event_only_tokens}</>}
            {sel.secondary_20m && <> · 20 m(부): {sel.secondary_20m.verdict === 'scored' ? `#${sel.secondary_20m.rank} ${pct(sel.secondary_20m.event_flag_frac)}` : 'abstain'}</>}
          </p>
          {showKorea && Object.keys(sel.korea_tags ?? {}).length > 0 && (
            <p style={{ margin: '6px 0 0', fontSize: 12 }}>한국 지정: {Object.entries(sel.korea_tags).map(([k, v]) => `${k}(${v.join(', ')})`).join(' · ')}</p>
          )}
        </section>
      )}

      <p className="footer-note">Sentinel-2 L2A (ESA/Copernicus via Planetary Computer) · 모델 Ai2 OlmoEarth v1 Base, frozen · 계약 {summary?.contract_sha256?.slice(0, 12)}…
        <br />독립 프로젝트입니다. Ai2 와 제휴·보증 관계가 없습니다.</p>
    </div>
  );
}
