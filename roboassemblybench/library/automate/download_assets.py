"""Download the public Isaac Sim 5.1 AutoMate asset bundle.

The AutoMate task implementation is distributed with Isaac Lab, while its
binary assets are hosted in NVIDIA's public S3 content bucket.  This helper
keeps that separation explicit and downloads the exact ``Isaac/IsaacLab``
tree expected by the vendored task configs.

Examples::

    python roboassemblybench/library/automate/download_assets.py
    python roboassemblybench/library/automate/download_assets.py --workers 8

The downloader intentionally uses a direct connection to the S3 endpoint by
default.  Some development proxies terminate the large S3 downloads early;
``--use-env-proxy`` restores urllib's normal environment-proxy behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock


S3_ENDPOINT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
S3_PREFIX = "Assets/Isaac/5.1/Isaac/IsaacLab/AutoMate/"
DEFAULT_DESTINATION = Path(__file__).resolve().parent / "assets" / "AutoMate"
MANIFEST_NAME = "automate_assets_manifest.json"
XML_NAMESPACE = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}


@dataclass(frozen=True)
class AssetObject:
    key: str
    relative_path: str
    size: int
    etag: str


def _opener(use_env_proxy: bool) -> urllib.request.OpenerDirector:
    if use_env_proxy:
        return urllib.request.build_opener()
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def list_assets(*, use_env_proxy: bool = False) -> list[AssetObject]:
    """List source assets from S3, following all ListObjectsV2 pages."""

    opener = _opener(use_env_proxy)
    objects: list[AssetObject] = []
    continuation_token: str | None = None
    while True:
        query: dict[str, str] = {
            "list-type": "2",
            "prefix": S3_PREFIX,
            "max-keys": "1000",
        }
        if continuation_token:
            query["continuation-token"] = continuation_token
        url = f"{S3_ENDPOINT}?{urllib.parse.urlencode(query)}"
        with opener.open(url, timeout=120) as response:
            root = ET.fromstring(response.read())

        for item in root.findall("s:Contents", XML_NAMESPACE):
            key = item.findtext("s:Key", default="", namespaces=XML_NAMESPACE)
            if not key.startswith(S3_PREFIX):
                continue
            relative_path = key[len(S3_PREFIX) :]
            # Thumbnails and a generated evaluation result are not simulation
            # inputs and only add noise/size to a reproducible asset mirror.
            if relative_path.startswith(".thumbs/") or relative_path.endswith(".h5"):
                continue
            size = int(item.findtext("s:Size", default="0", namespaces=XML_NAMESPACE))
            etag = item.findtext("s:ETag", default="", namespaces=XML_NAMESPACE).strip('"')
            objects.append(AssetObject(key=key, relative_path=relative_path, size=size, etag=etag))

        is_truncated = root.findtext("s:IsTruncated", default="false", namespaces=XML_NAMESPACE)
        if is_truncated.lower() != "true":
            break
        continuation_token = root.findtext("s:NextContinuationToken", namespaces=XML_NAMESPACE)
        if not continuation_token:
            raise RuntimeError("S3 reported a truncated listing without a continuation token.")

    return sorted(objects, key=lambda item: item.relative_path)


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - S3 ETags use MD5 for non-multipart objects.
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_one(
    item: AssetObject,
    destination: Path,
    *,
    use_env_proxy: bool,
    verify: bool,
) -> tuple[str, str]:
    output_path = destination / item.relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.is_file() and output_path.stat().st_size == item.size:
        if not verify or "-" in item.etag or _md5(output_path) == item.etag:
            output_path.chmod(0o644)
            return item.relative_path, "cached"

    url = S3_ENDPOINT + urllib.parse.quote(item.key, safe="/")
    opener = _opener(use_env_proxy)
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{output_path.name}.", suffix=".part", dir=output_path.parent, delete=False
    ) as temp_stream:
        temp_path = Path(temp_stream.name)
        try:
            with opener.open(url, timeout=180) as response:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    temp_stream.write(chunk)
            temp_stream.flush()
            os.fsync(temp_stream.fileno())
            if temp_path.stat().st_size != item.size:
                raise RuntimeError(
                    f"Size mismatch for {item.relative_path}: "
                    f"expected {item.size}, got {temp_path.stat().st_size}"
                )
            if verify and "-" not in item.etag and _md5(temp_path) != item.etag:
                raise RuntimeError(f"ETag mismatch for {item.relative_path}")
            os.replace(temp_path, output_path)
            output_path.chmod(0o644)
        finally:
            if temp_path.exists():
                temp_path.unlink()
    return item.relative_path, "downloaded"


def download_assets(
    destination: Path = DEFAULT_DESTINATION,
    *,
    workers: int = 4,
    use_env_proxy: bool = False,
    verify: bool = True,
) -> dict:
    """Download the AutoMate input assets and write a reproducibility manifest."""

    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    objects = list_assets(use_env_proxy=use_env_proxy)
    total_bytes = sum(item.size for item in objects)
    print(f"AutoMate objects: {len(objects)}, bytes: {total_bytes} ({total_bytes / 1e9:.3f} GB)")

    lock = Lock()
    completed = 0
    downloaded = 0
    cached = 0

    def report(result: tuple[str, str]) -> None:
        nonlocal completed, downloaded, cached
        _, status = result
        with lock:
            completed += 1
            downloaded += status == "downloaded"
            cached += status == "cached"
            if completed == 1 or completed % 25 == 0 or completed == len(objects):
                print(f"[{completed}/{len(objects)}] downloaded={downloaded} cached={cached}", flush=True)

    with ThreadPoolExecutor(max_workers=max(int(workers), 1)) as executor:
        futures = {
            executor.submit(
                _download_one,
                item,
                destination,
                use_env_proxy=use_env_proxy,
                verify=verify,
            ): item
            for item in objects
        }
        for future in as_completed(futures):
            item = futures[future]
            try:
                report(future.result())
            except Exception as exc:
                for pending in futures:
                    pending.cancel()
                raise RuntimeError(f"Failed to download {item.relative_path}") from exc

    manifest = {
        "source": S3_ENDPOINT,
        "source_prefix": S3_PREFIX,
        "isaac_sim_version": "5.1",
        "isaac_lab_asset_root": "${ISAACLAB_NUCLEUS_DIR}/AutoMate",
        "destination": (
            "roboassemblybench/library/automate/assets/AutoMate"
            if destination == DEFAULT_DESTINATION.resolve()
            else str(destination)
        ),
        "objects": [asdict(item) for item in objects],
    }
    manifest_path = destination.parent.parent / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Asset mirror ready: {destination}")
    print(f"Manifest: {manifest_path}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--use-env-proxy", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args(argv)
    try:
        download_assets(
            args.destination,
            workers=args.workers,
            use_env_proxy=args.use_env_proxy,
            verify=not args.no_verify,
        )
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
