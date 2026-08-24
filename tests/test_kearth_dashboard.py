from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from build_kearth_oreum_registry import load_access_status, render_dashboard  # noqa: E402
from render_kearth_dashboard import render_existing_registry  # noqa: E402


def fixture_payload() -> dict:
    return {
        "schema": "kearth-oreum-evidence-registry-v2",
        "generated_at": "2026-08-22T00:00:00+00:00",
        "registry_sha256": "a" * 64,
        "summary": {
            "official_inventory": 368,
            "inventory_coverage": 368,
            "attachment_corroborated": 188,
            "osm_peak_resolved": 243,
            "farmmap_point_state_c": 7,
            "model_screened": 243,
            "official_causal_evidence_ab": 0,
            "official_causal_evidence_rate": 0.0,
            "model_high_stable": 8,
            "rgb_persistent_confirmed": 0,
            "rgb_rejected": 8,
            "rgb_uncertain": 1,
            "decision_counts": {"abstain": 367, "investigate": 1},
        },
        "records": [],
    }


class DashboardRenderingTests(unittest.TestCase):
    def test_five_tracks_and_custody_boundaries_are_visible(self) -> None:
        dashboard = render_dashboard(fixture_payload())
        self.assertEqual(dashboard.count('class="track-card" data-track='), 5)
        for heading in (
            "공공데이터 신청·보유",
            "현재 데이터·실험",
            "비즈니스 가능성",
            "한국형 연구",
            "8월 EarthRoute 노트",
            "“내가 가진 데이터”의 현재 경계",
        ):
            self.assertIn(heading, dashboard)
        self.assertIn("API 463응답 · VWorld 지적 256/257", dashboard)
        self.assertIn("대표점 OK 후 257점 확장", dashboard)
        self.assertIn("VWorld는 256/257 대표 필지", dashboard)
        self.assertIn("API snapshot v3", dashboard)
        self.assertIn("unique PNU 235개", dashboard)
        self.assertIn("exact PNU 사건 1건도 관측구간 정렬은 0건", dashboard)
        self.assertIn("필지 source 충돌 1건", dashboard)
        self.assertNotIn("INCORRECT_KEY", dashboard)
        self.assertIn("토지피복 WMS</a>: 공개 WMS 성공", dashboard)
        self.assertIn("API 결합 대시보드 열기", dashboard)
        self.assertIn("5,184 manifest행", dashboard)
        self.assertNotIn("ECVAM_API_KEY", dashboard)

    def test_access_manifest_contains_no_secret_values(self) -> None:
        access_status = load_access_status(ROOT / "config/kearth_public_access.json")
        self.assertEqual(
            {service["id"] for service in access_status["services"]},
            {
                "vworld_cadastral",
                "eia_area",
                "building_hub",
                "gk2a",
                "vworld_context",
                "ngii_aerial",
                "mcee_landcover",
                "ecvam_context",
            },
        )
        ecvam = next(
            service
            for service in access_status["services"]
            if service["id"] == "ecvam_context"
        )
        self.assertEqual(ecvam["application_state"], "optional_not_applied")
        self.assertEqual(ecvam["credential_env"], "ECVAM_API_KEY")

    def test_render_existing_registry_rejects_wrong_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            registry = root / "registry.json"
            output = root / "dashboard.html"
            payload = fixture_payload()
            payload["summary"]["official_inventory"] = 367
            import json

            registry.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "368-record"):
                render_existing_registry(registry, output)


if __name__ == "__main__":
    unittest.main()
