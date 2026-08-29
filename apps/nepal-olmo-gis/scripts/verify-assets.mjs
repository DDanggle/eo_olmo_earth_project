import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';

const root = resolve(fileURLToPath(new URL('.', import.meta.url)), '..');
const scenario = JSON.parse(await readFile(resolve(root, 'public/data/scenario.json'), 'utf8'));
const hydrography = JSON.parse(await readFile(resolve(root, 'public/data/hydrography.geojson'), 'utf8'));

assert.equal(scenario.schema, 'olmoearth-nepal-live-twin/v2');
assert.ok(scenario.scene_records.length >= 9);
assert.ok(scenario.scene_records.some((scene) => scene.id === 's2-2026-08-27'));
assert.equal(scenario.olmoearth.anchors, 5);
assert.equal(scenario.research.confirmatory_transfer.regions, 8);
assert.equal(scenario.research.confirmatory_transfer.wins_reuse_vs_raw_strong, 6);
assert.equal(scenario.downstream_visual.purpose, 'visual_only_downstream_context_not_part_of_five_anchor_olmo_contract');
assert.deepEqual(scenario.downstream_visual.records.map((record) => record.label), ['pre', 'post']);
assert.ok(scenario.points.find((point) => point.id === 'E')?.display_label === 'SOURCE ESTIMATE');
assert.ok(scenario.points.find((point) => point.id === 'E')?.map_label === 'E · SOURCE');
assert.ok(scenario.points.find((point) => point.id === 'C')?.in_event_chain === false);
assert.ok(scenario.points.find((point) => point.id === 'C')?.map_label === 'C · CONTROL');
assert.equal(scenario.olmoearth.embedding_status, 'not_run_in_this_web_snapshot');
assert.equal(scenario.live_observation.catalog_status, 'published');
assert.equal(scenario.live_observation.olmo_ready, false);
assert.equal(scenario.live_observation.selection_preflight_valid, true);
assert.equal(scenario.live_observation.materialization_seal_valid, false);
assert.equal(scenario.live_observation.materialization_status, 'partial_cube_contract_failed');
assert.deepEqual(scenario.live_observation.period_readiness, { sentinel1: 3, sentinel2_l2a: 4 });
assert.equal(scenario.decision.action, 'DO NOT EMBED');
assert.match(scenario.decision.reason, /S1 3\/4; S2 4\/4/);
assert.match(scenario.decision.next_gate, /Sentinel-1D.*2026-08-31/);
assert.ok(scenario.ops_log.some((event) => event.type === 'SEAL_INVALID'));
assert.ok(scenario.ops_log.some((event) => event.type === 'COVERAGE_MISS'));
assert.equal(scenario.scheduled_scenes.find((scene) => scene.id === 's1d_20260828')?.state, 'missed_coverage');
assert.equal(new Set(scenario.ops_log.map((event) => event.event_id)).size, scenario.ops_log.length);
assert.ok(scenario.live_observation.cloud_cover_tile_pct > 0 && scenario.live_observation.cloud_cover_tile_pct <= 100);
assert.match(scenario.live_observation.product_name, /^S2B_MSIL2A_20260827/);
assert.equal(scenario.simulation.claim, 'illustrative_kinematic_preview_not_hazard_forecast');
assert.ok(hydrography.simulation_route.length >= 40 && hydrography.simulation_route.length <= 96);
assert.equal(hydrography.features.length, 11);

for (const scene of scenario.scene_records) {
  const image = await readFile(resolve(root, 'public', scene.image.slice(1)));
  assert.ok(image.length > 1_000, `${scene.id} rendered image is unexpectedly small`);
  assert.deepEqual([...image.subarray(0, 8)], [137, 80, 78, 71, 13, 10, 26, 10]);
  assert.equal(scene.coordinates.length, 4);
  assert.match(scene.source_sha256, /^[a-f0-9]{64}$/);
}

const wasmBytes = await readFile(resolve(root, 'public/wasm/nepal_flow.wasm'));
const mapWorker = await readFile(resolve(root, 'public/maplibre-gl-worker.mjs'));
const mapWorkerShared = await readFile(resolve(root, 'public/maplibre-gl-shared.mjs'));
assert.ok(mapWorker.length > 10_000, 'MapLibre worker entry is missing or truncated');
assert.ok(mapWorkerShared.length > 100_000, 'MapLibre shared worker bundle is missing or truncated');
const instantiated = await WebAssembly.instantiate(wasmBytes, {});
const wasm = instantiated.instance.exports;
assert.equal(wasm.abi_version(), 1);
wasm.clear_route();
hydrography.simulation_route.forEach(([lon, lat], index) => wasm.set_route_point(index, lon, lat));
wasm.reset(20260826);
wasm.step(0.016, 0.034);
const count = wasm.particle_count();
assert.equal(count, 280);
const values = new Float32Array(wasm.memory.buffer, wasm.particles_ptr(), count * 3);
assert.ok(values.every(Number.isFinite));
assert.ok(values[0] > 80 && values[0] < 90);
assert.ok(values[1] > 20 && values[1] < 35);

console.log(JSON.stringify({ scenes: scenario.scene_records.length, anchors: scenario.olmoearth.anchors, route_points: hydrography.simulation_route.length, wasm_particles: count, map_worker_bytes: mapWorker.length + mapWorkerShared.length }, null, 2));
