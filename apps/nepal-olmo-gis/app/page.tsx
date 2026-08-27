'use client';

import { AttributionControl, LngLatBounds, Map as MapLibreMap, NavigationControl } from 'maplibre-gl';
import type { Feature, FeatureCollection } from 'geojson';
import Image from 'next/image';
import { useEffect, useMemo, useRef, useState } from 'react';
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

type Scenario = {
  generated_at: string;
  event: { name: string; cause_status: string; evidence_status: string };
  scene_records: SceneRecord[];
  olmoearth: { input_contract: string; anchors: number; embedding_status: string; post_event_delta: string };
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

const researchPoints: FeatureCollection = {
  type: 'FeatureCollection',
  features: [
    { type: 'Feature', properties: { id: 'A', name: 'Rasuwagadhi impact AOI' }, geometry: { type: 'Point', coordinates: [85.3780644, 28.2786794] } },
    { type: 'Feature', properties: { id: 'B', name: 'Gyirong border checkpoint' }, geometry: { type: 'Point', coordinates: [85.3763336, 28.2828546] } },
    { type: 'Feature', properties: { id: 'C', name: 'Rishing reference' }, geometry: { type: 'Point', coordinates: [84.3103107, 27.8790412] } },
  ],
};

const pointCards = [
  { id: 'A', name: 'Rasuwagadhi impact AOI', coordinates: [85.3780644, 28.2786794], distance: '0.00 km', role: 'FOCUS' },
  { id: 'B', name: 'Gyirong border checkpoint', coordinates: [85.3763336, 28.2828546], distance: '0.49 km', role: 'BORDER' },
  { id: 'C', name: 'Rishing reference', coordinates: [84.3103107, 27.8790412], distance: '113.79 km', role: 'SEPARATE' },
];

const timeline = [
  { id: 's2-2026-07-03', date: '03 JUL', sensor: 'S2', state: 'READY' },
  { id: 's1-2026-07-11', date: '11 JUL', sensor: 'S1', state: 'READY' },
  { id: 's2-2026-07-23', date: '23 JUL', sensor: 'S2', state: 'READY' },
  { id: 's1-2026-08-04', date: '04 AUG', sensor: 'S1', state: 'READY' },
  { id: 's2-2026-08-12', date: '12 AUG', sensor: 'S2', state: 'READY' },
  { id: 's1-2026-08-24', date: '24 AUG', sensor: 'S1', state: 'READY' },
  { id: 'event-2026-08-26', date: '26 AUG', sensor: 'EVENT', state: 'IMPACT' },
  { id: 's2-2026-08-27', date: '27 AUG', sensor: 'S2', state: 'PENDING' },
  { id: 's1-2026-08-28', date: '28 AUG', sensor: 'S1', state: 'PLANNED' },
];

const rasterStyle = {
  version: 8 as const,
  sources: {
    osm: {
      type: 'raster' as const,
      tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
      tileSize: 256,
      attribution: '© OpenStreetMap contributors',
    },
  },
  layers: [{ id: 'osm', type: 'raster' as const, source: 'osm' }],
};

export default function Home() {
  const mapNode = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const flowSpeedRef = useRef(0.034);
  const flowPlayingRef = useRef(true);
  const [mapReady, setMapReady] = useState(false);
  const [scenario, setScenario] = useState<Scenario | null>(null);
  const [hydrography, setHydrography] = useState<Hydrography | null>(null);
  const [activeSceneId, setActiveSceneId] = useState('s2-2026-08-12');
  const [selectedPoint, setSelectedPoint] = useState('A');
  const [overlayOpacity, setOverlayOpacity] = useState(0.78);
  const [showAnchors, setShowAnchors] = useState(true);
  const [flowPlaying, setFlowPlaying] = useState(true);
  const [flowSpeed, setFlowSpeed] = useState(0.034);
  const [wasmStatus, setWasmStatus] = useState<'loading' | 'ready' | 'failed'>('loading');

  useEffect(() => {
    Promise.all([
      fetch('/data/scenario.json').then((response) => response.json() as Promise<Scenario>),
      fetch('/data/hydrography.geojson').then((response) => response.json() as Promise<Hydrography>),
    ]).then(([nextScenario, nextHydrography]) => {
      setScenario(nextScenario);
      setHydrography(nextHydrography);
    }).catch(() => setWasmStatus('failed'));
  }, []);

  useEffect(() => {
    if (!mapNode.current || mapRef.current) return;
    const map = new MapLibreMap({
      container: mapNode.current,
      style: rasterStyle,
      center: [85.365, 28.235],
      zoom: 11.3,
      pitch: 48,
      bearing: -18,
      attributionControl: false,
    });
    map.addControl(new NavigationControl({ showCompass: true }), 'bottom-right');
    map.addControl(new AttributionControl({ compact: true }), 'bottom-right');
    map.on('load', () => {
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
      setMapReady(true);
    });
    map.on('click', 'point-core', (event) => {
      const id = event.features?.[0]?.properties?.id;
      if (id) setSelectedPoint(id);
    });
    map.on('mouseenter', 'point-core', () => { map.getCanvas().style.cursor = 'pointer'; });
    map.on('mouseleave', 'point-core', () => { map.getCanvas().style.cursor = ''; });
    mapRef.current = map;
    return () => { map.remove(); mapRef.current = null; };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!mapReady || !map || !hydrography || map.getSource('hydrography')) return;
    map.addSource('hydrography', { type: 'geojson', data: hydrography as FeatureCollection });
    map.addLayer({ id: 'river-casing', type: 'line', source: 'hydrography', paint: { 'line-color': '#06100e', 'line-width': 8, 'line-opacity': 0.82 } }, 'point-halo');
    map.addLayer({ id: 'river-route', type: 'line', source: 'hydrography', paint: { 'line-color': '#5fffd7', 'line-width': 2.2, 'line-opacity': 0.8, 'line-dasharray': [1.2, 1.6] } }, 'point-halo');
  }, [hydrography, mapReady]);

  useEffect(() => {
    const map = mapRef.current;
    if (!mapReady || !map) return;
    fetch('/data/olmo-input-anchors.geojson').then((response) => response.json() as Promise<FeatureCollection>).then((anchors) => {
      if (map.getSource('olmo-anchors')) return;
      map.addSource('olmo-anchors', { type: 'geojson', data: anchors });
      map.addLayer({ id: 'olmo-anchor-fill', type: 'fill', source: 'olmo-anchors', paint: { 'fill-color': '#5fffd7', 'fill-opacity': 0.045 } }, 'point-halo');
      map.addLayer({ id: 'olmo-anchor-line', type: 'line', source: 'olmo-anchors', paint: { 'line-color': '#b7ffe9', 'line-width': 1, 'line-opacity': 0.52, 'line-dasharray': [3, 2] } }, 'point-halo');
    }).catch(() => undefined);
  }, [mapReady]);

  useEffect(() => {
    const map = mapRef.current;
    if (!mapReady || !map || !scenario) return;
    if (map.getLayer('satellite-scene')) map.removeLayer('satellite-scene');
    if (map.getSource('satellite-scene')) map.removeSource('satellite-scene');
    const scene = scenario.scene_records.find((item) => item.id === activeSceneId);
    if (!scene) return;
    map.addSource('satellite-scene', { type: 'image', url: scene.image, coordinates: scene.coordinates });
    map.addLayer({ id: 'satellite-scene', type: 'raster', source: 'satellite-scene', paint: { 'raster-opacity': overlayOpacity, 'raster-fade-duration': 120 } }, 'point-halo');
  }, [activeSceneId, mapReady, overlayOpacity, scenario]);

  useEffect(() => {
    const map = mapRef.current;
    if (!mapReady || !map || !map.getLayer('olmo-anchor-fill')) return;
    map.setLayoutProperty('olmo-anchor-fill', 'visibility', showAnchors ? 'visible' : 'none');
    map.setLayoutProperty('olmo-anchor-line', 'visibility', showAnchors ? 'visible' : 'none');
  }, [mapReady, showAnchors]);

  useEffect(() => {
    flowPlayingRef.current = flowPlaying;
  }, [flowPlaying]);

  useEffect(() => {
    flowSpeedRef.current = flowSpeed;
  }, [flowSpeed]);

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

  const activeTimeline = timeline.find((item) => item.id === activeSceneId) ?? timeline[4];
  const activeScene = scenario?.scene_records.find((item) => item.id === activeSceneId) ?? null;
  const latestBaseline = scenario?.scene_records.find((item) => item.id === 's1-2026-08-24') ?? null;
  const previewScene = activeScene ?? latestBaseline;
  const selectedCard = pointCards.find((item) => item.id === selectedPoint) ?? pointCards[0];

  const selectedPlace = useMemo(() => {
    if (selectedPoint === 'A') return 'Rasuwa, Nepal · Pasang Lhamu Hwy';
    if (selectedPoint === 'B') return 'Gyirong, Tibet · G216';
    return 'Rishing-03, Tanahun · separate basin audit';
  }, [selectedPoint]);

  const focusPoint = (id: string) => {
    setSelectedPoint(id);
    const card = pointCards.find((item) => item.id === id);
    if (!card) return;
    mapRef.current?.flyTo({ center: card.coordinates as [number, number], zoom: id === 'C' ? 10 : 14, pitch: id === 'C' ? 20 : 50, duration: 1300 });
  };

  const fitCorridor = () => {
    mapRef.current?.fitBounds(new LngLatBounds([85.302, 28.135], [85.386, 28.288]), { padding: { top: 110, right: 345, bottom: 150, left: 345 }, duration: 1500 });
  };

  return (
    <main className="app-shell">
      <div ref={mapNode} className="map-stage" aria-label="Rasuwagadhi satellite and simulation map" />
      <canvas ref={canvasRef} className="flow-canvas" aria-hidden="true" />
      <div className="terrain-wash" aria-hidden="true" />

      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark"><span /></div>
          <div><p className="eyebrow">AI2 / PLANETARY INTELLIGENCE PROTOTYPE</p><h1>OLMoEarth <span>Live Twin</span></h1></div>
        </div>
        <button className="corridor-button" onClick={fitCorridor}>⌖ FIT RIVER CORRIDOR</button>
        <div className="event-status"><span className="live-dot" /><div><strong>RASUWA · NEPAL</strong><small>26 AUG 2026 · INVESTIGATION</small></div></div>
      </header>

      <aside className="left-rail glass-panel">
        <div className="panel-heading"><span>01</span><div><p>AREA OF INTEREST</p><strong>Coordinate audit</strong></div></div>
        <div className="coordinate-list">
          {pointCards.map((point) => (
            <button key={point.id} className={selectedPoint === point.id ? 'coordinate active' : 'coordinate'} onClick={() => focusPoint(point.id)}>
              <span>{point.id}</span><div><strong>{point.name}</strong><small>{point.coordinates[1].toFixed(6)}, {point.coordinates[0].toFixed(6)}</small></div><em>{point.role}</em>
            </button>
          ))}
        </div>
        <div className="selected-place"><span>{selectedCard.distance} FROM A</span><strong>{selectedPlace}</strong></div>
        <p className="audit-note"><b>C is 113.79 km away.</b> It is not used as the event flow endpoint; it remains a separate transfer/reference AOI.</p>
        <div className="layer-controls">
          <label><span>Satellite overlay</span><b>{Math.round(overlayOpacity * 100)}%</b></label>
          <input type="range" min="0" max="1" step="0.02" value={overlayOpacity} onChange={(event) => setOverlayOpacity(Number(event.target.value))} />
          <button className={showAnchors ? 'toggle active' : 'toggle'} onClick={() => setShowAnchors((value) => !value)}><i /> OLMo input windows</button>
        </div>
      </aside>

      <aside className="right-rail glass-panel">
        <div className="panel-heading"><span>02</span><div><p>EVIDENCE LENS</p><strong>Before → after contract</strong></div></div>
        <div className="compare-strip">
          <div className="scene-preview">
            {previewScene ? <Image src={previewScene.image} alt={`${previewScene.sensor} pre-event observation`} fill unoptimized sizes="150px" /> : <span className="loading-grid" />}
            <span>PRE · {previewScene?.acquired_at.slice(0, 10) ?? 'LOADING'}</span>
          </div>
          <div className="compare-arrow">→</div>
          <div className="scene-preview pending-preview"><span className="waiting-cross" /><span>POST · PENDING</span></div>
        </div>
        <div className="pipeline-stack">
          <div className="pipeline-row ready"><span>S1</span><div><strong>Radar baseline</strong><small>4 acquisitions · local GeoTIFF</small></div><b>READY</b></div>
          <div className="pipeline-row ready"><span>S2</span><div><strong>Optical baseline</strong><small>4 acquisitions · true color from 12 bands</small></div><b>READY</b></div>
          <div className="pipeline-row ready"><span>OE</span><div><strong>OLMoEarth contract</strong><small>5 anchors · S1+S2 · 4 periods</small></div><b>INPUT</b></div>
          <div className="pipeline-row pending"><span>Δ</span><div><strong>Embedding delta</strong><small>post-event scene required</small></div><b>BLOCKED</b></div>
          <div className={`pipeline-row ${wasmStatus === 'ready' ? 'ready' : 'preview'}`}><span>W</span><div><strong>Flow layer</strong><small>Rust/WASM · {scenario?.simulation.route_points ?? '—'} route nodes</small></div><b>{wasmStatus.toUpperCase()}</b></div>
        </div>
        <div className="flow-control">
          <button onClick={() => setFlowPlaying((value) => !value)}>{flowPlaying ? 'Ⅱ' : '▶'}</button>
          <div><label><span>ILLUSTRATIVE FLOW</span><b>{(flowSpeed / 0.034).toFixed(1)}×</b></label><input type="range" min="0.012" max="0.09" step="0.002" value={flowSpeed} onChange={(event) => setFlowSpeed(Number(event.target.value))} /></div>
        </div>
        <div className="truth-box"><span>CLAIM BOUNDARY</span><p>Particles follow the verified OSM Bhote Koshi→Trishuli centerline. They show interface flow, not flood depth, arrival time, or hazard.</p></div>
      </aside>

      <section className="timeline glass-panel" aria-label="Satellite acquisition timeline">
        <div className="timeline-title"><span>03</span><div><p>SCENE TIMELINE</p><strong>{activeTimeline.date} · {activeTimeline.sensor} · {activeTimeline.state}</strong></div></div>
        <div className="scene-track">
          {timeline.map((scene) => (
            <button key={scene.id} className={scene.id === activeSceneId ? 'scene active' : 'scene'} onClick={() => setActiveSceneId(scene.id)}>
              <span className={`scene-node ${scene.state.toLowerCase()}`} /><strong>{scene.date}</strong><small>{scene.sensor}</small><em>{scene.state}</em>
            </button>
          ))}
        </div>
      </section>

      <div className="map-legend"><span><i className="mint" />Verified river route</span><span><i className="white" />OLMo input 2.56 km</span><span><i className="amber" />Unverified / pending</span></div>
      <div className="provenance-stamp">DATA SNAPSHOT {scenario?.generated_at.slice(0, 16).replace('T', ' ') ?? 'LOADING'} UTC · OSM ODbL · ESA COPERNICUS</div>
    </main>
  );
}
