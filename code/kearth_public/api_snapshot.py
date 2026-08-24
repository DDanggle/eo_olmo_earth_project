"""Secret-safe primitives for reproducible Korean public API snapshots.

The module deliberately separates a request's public identity from credentials.
Raw response bytes are retained after credential redaction, while manifests never
store the fully expanded URL.  This keeps failed API calls useful as coverage
evidence without turning research artifacts into secret-bearing logs.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping


Transport = Callable[[urllib.request.Request, float], tuple[int, Mapping[str, str], bytes]]


def read_env_file(path: Path) -> dict[str, str]:
    """Read a dotenv file without executing shell syntax."""

    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"invalid dotenv line {line_number}")
        name, value = line.split("=", 1)
        name = name.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"invalid dotenv name at line {line_number}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'\"', "'"}:
            value = value[1:-1]
        values[name] = value
    return values


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def request_hash(source_id: str, endpoint: str, public_params: Mapping[str, object]) -> str:
    payload = {
        "source_id": source_id,
        "endpoint": endpoint,
        "public_params": {str(key): str(value) for key, value in public_params.items()},
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def redact_bytes(payload: bytes, secrets: Mapping[str, str]) -> bytes:
    result = payload
    for secret in secrets.values():
        if not secret:
            continue
        variants = {secret, urllib.parse.quote(secret, safe=""), urllib.parse.quote_plus(secret)}
        for variant in variants:
            result = result.replace(variant.encode("utf-8"), b"<redacted>")
    return result


def _default_transport(
    request: urllib.request.Request, timeout: float
) -> tuple[int, Mapping[str, str], bytes]:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, dict(response.headers.items()), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers.items()), exc.read()


@dataclass(frozen=True, slots=True)
class RequestSpec:
    source_id: str
    endpoint: str
    public_params: Mapping[str, object]
    secret_params: Mapping[str, str]
    response_suffix: str = ".bin"
    target_id: str | None = None

    @property
    def identity(self) -> str:
        return request_hash(self.source_id, self.endpoint, self.public_params)


def execute_request(
    spec: RequestSpec,
    *,
    output_dir: Path,
    env: Mapping[str, str],
    retrieved_at: str | None = None,
    timeout: float = 60,
    transport: Transport = _default_transport,
) -> dict[str, object]:
    """Execute one request and retain a credential-free record plus raw bytes."""

    retrieved_at = retrieved_at or datetime.now(timezone.utc).isoformat()
    record: dict[str, object] = {
        "schema": "kearth-api-request-v1",
        "source_id": spec.source_id,
        "target_id": spec.target_id,
        "endpoint": spec.endpoint,
        "public_params": {str(k): str(v) for k, v in spec.public_params.items()},
        "credential_env_names": sorted(set(spec.secret_params.values())),
        "request_hash": spec.identity,
        "retrieved_at": retrieved_at,
    }
    missing = sorted(
        env_name
        for env_name in set(spec.secret_params.values())
        if not str(env.get(env_name, "")).strip()
    )
    if missing:
        record.update(
            {
                "outcome": "not_requested_missing_credential",
                "missing_credential_env_names": missing,
                "http_status": None,
                "raw_file": None,
            }
        )
        return record

    params = {str(key): str(value) for key, value in spec.public_params.items()}
    used_secrets: dict[str, str] = {}
    for request_name, env_name in spec.secret_params.items():
        value = str(env[env_name]).strip()
        # data.go.kr exposes both encoded and decoded variants of a service key.
        # Normalize a visibly percent-encoded value before urlencode so it is not
        # encoded twice; raw '+' and '/' characters remain handled by urlencode.
        request_value = urllib.parse.unquote(value) if re.search(r"%[0-9A-Fa-f]{2}", value) else value
        params[request_name] = request_value
        used_secrets[env_name] = value
        used_secrets[f"{env_name}_request"] = request_value
    url = f"{spec.endpoint}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": "kearth-research-snapshot/1"})

    try:
        status, headers, body = transport(request, timeout)
        body = redact_bytes(body, used_secrets)
        raw_dir = output_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_name = f"{spec.source_id}__{spec.identity[:20]}{spec.response_suffix}"
        raw_path = raw_dir / raw_name
        raw_path.write_bytes(body)
        content_type = str(headers.get("Content-Type", headers.get("content-type", "")))
        record.update(
            {
                "outcome": "http_success" if 200 <= status < 300 else "http_error",
                "http_status": status,
                "content_type": content_type,
                "raw_file": str(Path("raw") / raw_name),
                "raw_bytes": len(body),
                "raw_sha256": hashlib.sha256(body).hexdigest(),
            }
        )
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        record.update(
            {
                "outcome": "transport_error",
                "http_status": None,
                "raw_file": None,
                "error_class": type(exc).__name__,
                "error_message": redact_bytes(str(reason).encode(), used_secrets)
                .decode("utf-8", errors="replace")[:500],
            }
        )
    return record


def load_json_response(output_dir: Path, record: Mapping[str, object]) -> object | None:
    raw_file = record.get("raw_file")
    if not raw_file:
        return None
    body = (output_dir / str(raw_file)).read_bytes().lstrip(b"\xef\xbb\xbf")
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def data_go_items(payload: object) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Normalize the common data.go.kr JSON envelope without hiding API errors."""

    if not isinstance(payload, dict):
        return [], {"parse_status": "not_json_object"}
    response = payload.get("response", payload)
    if not isinstance(response, dict):
        return [], {"parse_status": "missing_response"}
    header = response.get("header", {})
    body = response.get("body", {})
    if not isinstance(body, dict):
        body = {}
    raw_items = body.get("items", [])
    if isinstance(raw_items, dict):
        raw_items = raw_items.get("item", [])
    if isinstance(raw_items, dict):
        items = [raw_items]
    elif isinstance(raw_items, list):
        items = [item for item in raw_items if isinstance(item, dict)]
    else:
        items = []
    meta: dict[str, object] = {
        "parse_status": "parsed",
        "header": header if isinstance(header, dict) else {},
        "page_no": body.get("pageNo"),
        "num_of_rows": body.get("numOfRows"),
        "total_count": body.get("totalCount"),
    }
    return items, meta


def vworld_features(payload: object) -> tuple[list[dict[str, object]], dict[str, object]]:
    if not isinstance(payload, dict):
        return [], {"parse_status": "not_json_object"}
    response = payload.get("response", {})
    if not isinstance(response, dict):
        return [], {"parse_status": "missing_response"}
    result = response.get("result", {})
    collection = result.get("featureCollection", {}) if isinstance(result, dict) else {}
    features = collection.get("features", []) if isinstance(collection, dict) else []
    if not isinstance(features, list):
        features = []
    return [item for item in features if isinstance(item, dict)], {
        "parse_status": "parsed",
        "status": response.get("status"),
        "record": response.get("record"),
        "error": response.get("error"),
    }


def vworld_semantic_status(
    meta: Mapping[str, object], feature_count: int | None = None
) -> str:
    """Classify VWorld business status without treating an empty point as a key error."""

    status = str(meta.get("status") or "").strip().upper()
    if status == "OK":
        return "api_no_features" if feature_count == 0 else "api_success"
    if status == "NOT_FOUND":
        return "api_no_features"
    return "api_error"


def eia_features(payload: bytes) -> list[dict[str, object]]:
    """Extract EIA attributes and lon/lat polygon rings from a WFS/GML response."""

    try:
        root = ET.fromstring(payload)
    except ET.ParseError:
        return []
    result: list[dict[str, object]] = []
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] != "BSNS_AREA":
            continue
        attributes: dict[str, object] = {}
        rings: list[list[list[float]]] = []
        for child in element:
            name = child.tag.rsplit("}", 1)[-1]
            if name != "the_geom":
                attributes[name] = (child.text or "").strip()
        for descendant in element.iter():
            if descendant.tag.rsplit("}", 1)[-1] != "posList":
                continue
            try:
                values = [float(value) for value in (descendant.text or "").split()]
            except ValueError:
                continue
            if len(values) >= 6 and len(values) % 2 == 0:
                rings.append([[values[index], values[index + 1]] for index in range(0, len(values), 2)])
        result.append({"attributes": attributes, "rings_lon_lat": rings})
    return result


def point_in_ring(lon: float, lat: float, ring: list[list[float]]) -> bool:
    """Ray-casting point-in-ring test; boundary points are treated as inside."""

    if len(ring) < 3:
        return False
    inside = False
    previous = ring[-1]
    for current in ring:
        x1, y1 = previous
        x2, y2 = current
        cross = (x2 - x1) * (lat - y1) - (y2 - y1) * (lon - x1)
        if abs(cross) < 1e-12 and min(x1, x2) <= lon <= max(x1, x2) and min(y1, y2) <= lat <= max(y1, y2):
            return True
        if (y1 > lat) != (y2 > lat):
            intersection_x = (x2 - x1) * (lat - y1) / (y2 - y1) + x1
            if lon < intersection_x:
                inside = not inside
        previous = current
    return inside
