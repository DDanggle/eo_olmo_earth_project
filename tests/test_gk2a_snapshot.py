import importlib.util
import gzip
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "code" / "gk2a_snapshot.py"
SPEC = importlib.util.spec_from_file_location("gk2a_snapshot", SCRIPT)
assert SPEC and SPEC.loader
GK2A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GK2A)


class GK2ASnapshotContractTest(unittest.TestCase):
    @staticmethod
    def write_successes(day: Path, names: list[str]) -> None:
        records = []
        for name in names:
            raw = (name + "\n").encode()
            (day / name).write_bytes(gzip.compress(raw))
            records.append({
                "file": name,
                "ok": True,
                "sha256": hashlib.sha256(raw).hexdigest(),
            })
        (day / "manifest.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    def test_current_schedule_has_57_unique_files(self):
        self.assertEqual(len(GK2A.expected_jobs()), 57)
        self.assertEqual(len(GK2A.expected_filenames()), 57)

    def test_status_does_not_require_api_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = os.environ.copy()
            env.pop("DATA_GO_KR_SERVICE_KEY", None)
            env["GK2A_ROOT"] = tmp
            run = subprocess.run(
                [sys.executable, str(SCRIPT), "--status"],
                capture_output=True,
                check=False,
                env=env,
                text=True,
            )
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("수집 없음", run.stdout)
        self.assertNotIn("DATA_GO_KR_SERVICE_KEY 없음", run.stderr)

    def test_status_fails_closed_on_partial_day(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            day = root / "2026" / "08" / "24"
            day.mkdir(parents=True)
            self.write_successes(day, sorted(GK2A.expected_filenames())[:-1])

            old_root = GK2A.ROOT
            GK2A.ROOT = root
            try:
                output = StringIO()
                with redirect_stdout(output):
                    GK2A.report_status()
            finally:
                GK2A.ROOT = old_root

        rendered = output.getvalue()
        self.assertIn("불완전 날짜 1일", rendered)
        self.assertIn("complete 56/57", rendered)
        self.assertIn("MISSING 1", rendered)

    def test_status_counts_explicit_no_data_as_terminal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            day = root / "2026" / "08" / "24"
            day.mkdir(parents=True)
            names = sorted(GK2A.expected_filenames())
            self.write_successes(day, names[:-1])
            with (day / "manifest.jsonl").open("a", encoding="utf-8") as manifest:
                manifest.write(json.dumps({
                    "file": names[-1],
                    "resultCode": "03",
                    "ok": False,
                }) + "\n")

            old_root = GK2A.ROOT
            GK2A.ROOT = root
            try:
                output = StringIO()
                with redirect_stdout(output):
                    GK2A.report_status()
            finally:
                GK2A.ROOT = old_root

        rendered = output.getvalue()
        self.assertIn("불완전 날짜 0일", rendered)
        self.assertIn("complete 57/57 (data 56, no_data  1)", rendered)
        self.assertIn("OK", rendered)

    def test_status_rejects_file_with_wrong_raw_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            day = root / "2026" / "08" / "24"
            day.mkdir(parents=True)
            names = sorted(GK2A.expected_filenames())
            self.write_successes(day, names)
            (day / names[0]).write_bytes(gzip.compress(b"tampered"))

            old_root = GK2A.ROOT
            GK2A.ROOT = root
            try:
                output = StringIO()
                with redirect_stdout(output):
                    GK2A.report_status()
            finally:
                GK2A.ROOT = old_root

        rendered = output.getvalue()
        self.assertIn("complete 56/57", rendered)
        self.assertIn("MISSING 1", rendered)


if __name__ == "__main__":
    unittest.main()
