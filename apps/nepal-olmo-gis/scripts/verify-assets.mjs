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
assert.equal(scenario.research.ai_run_ledger.length, 6);
assert.equal(scenario.research.ai_run_ledger.find((run) => run.id === 'nepal_pre_event_representation')?.state, 'EXECUTED');
assert.match(scenario.research.ai_run_ledger.find((run) => run.id === 'nepal_pre_event_representation')?.output ?? '', /15 sealed embedding rasters.*768, 64, 64/);
assert.equal(scenario.research.ai_run_ledger.find((run) => run.id === 'pre_event_forecast')?.state, 'NEGATIVE_RESULT');
const liveVerdict = typeof scenario.olmoearth.post_event_delta === 'object' && scenario.olmoearth.post_event_delta && scenario.olmoearth.post_event_delta.live_mode;
assert.equal(scenario.research.ai_run_ledger.find((run) => run.id === 'nepal_post_event_delta')?.state, liveVerdict ? 'EXECUTED' : 'WAITING_INPUT');
assert.equal(scenario.research.ai_run_ledger.find((run) => run.id === 'matched_second_geofm')?.state, 'NOT_RUN');
assert.equal(scenario.downstream_visual.purpose, 'visual_only_downstream_context_not_part_of_five_anchor_olmo_contract');
assert.deepEqual(scenario.downstream_visual.records.map((record) => record.label), ['pre', 'post']);
assert.ok(scenario.points.find((point) => point.id === 'E')?.display_label === 'SOURCE ESTIMATE');
assert.ok(scenario.points.find((point) => point.id === 'E')?.map_label === 'E · SOURCE');
assert.ok(scenario.points.find((point) => point.id === 'C')?.in_event_chain === false);
assert.ok(scenario.points.find((point) => point.id === 'C')?.map_label === 'C · CONTROL');
assert.equal(scenario.olmoearth.embedding_status, 'not_run_in_this_web_snapshot');
assert.equal(scenario.live_observation.catalog_status, 'published');
// 2026-08-29: 라이브 판정 전/후 두 상태 모두 검증함 (판정 전 = 대기 불변식, 판정 후 = 봉인 불변식)
if (liveVerdict) {
  assert.equal(scenario.live_observation.olmo_ready, true);
  assert.equal(scenario.live_observation.selection_preflight_valid, true);
  assert.equal(scenario.live_observation.materialization_seal_valid, true);
  assert.equal(scenario.headline?.sealed_total, 5);
  assert.ok(scenario.candidates && scenario.candidates.windows === 27);
} else {
  assert.equal(scenario.live_observation.olmo_ready, false);
  assert.equal(scenario.live_observation.selection_preflight_valid, false);
  assert.equal(scenario.live_observation.materialization_seal_valid, false);
  assert.equal(scenario.live_observation.materialization_status, 'blocked_provider_selection');
}
assert.equal(scenario.live_observation.coverage_status, 'operational_anchors_covered');
assert.equal(scenario.live_observation.operational_anchor_count, 5);
assert.equal(scenario.decision.action, 'WAIT FOR PROVIDER SYNC');
assert.match(scenario.decision.reason, /covering 5\/5 anchors/);
assert.match(scenario.decision.next_gate, /2026-08-28 scene.*5\/5 anchors/);
assert.ok(scenario.ops_log.some((event) => event.type === 'SEAL_INVALID'));
assert.ok(scenario.ops_log.some((event) => event.type === 'COVERAGE_PASS'));
assert.equal(scenario.scheduled_scenes.find((scene) => scene.id === 's2c_20260829')?.state, 'acquired_pending_catalog');
assert.equal(new Set(scenario.ops_log.map((event) => event.event_id)).size, scenario.ops_log.length);
assert.equal(scenario.live_observation.cloud_cover_tile_pct, null);
assert.match(scenario.live_observation.product_name, /^S1D_IW_GRDH_1SDV_20260828/);
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
