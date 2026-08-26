#!/usr/bin/env python3
"""Stream successful episodes from the production host to a shared disk.

The destination host runs this process.  It reads only successful entries from
the source manifests over SSH, streams one episode at a time through tar, and
commits a durable SQLite record only after the episode directory has been
extracted and atomically moved into its final location.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import os
import shlex
import signal
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any


REMOTE_JQ_LISTING_SCRIPT = r'''
set -euo pipefail
root="$1"
shift
for manifest in "$@"; do
  jq -c \
    --arg manifest "$manifest" \
    --arg prefix "$root/" \
    '
      (.successful_episodes // {})[]
      | select((.valid // true) == true)
      | (.metadata_path // "") as $metadata
      | select($metadata | startswith($prefix))
      | ($metadata | ltrimstr($prefix)) as $relative_metadata
      | {
          source_manifest: $manifest,
          source_metadata: $relative_metadata,
          source_episode: ($relative_metadata | split("/") | .[:-1] | join("/")),
          seed: .seed,
          layout_seed: .layout_seed,
          frame_count: .frame_count
        }
    ' \
    "$root/$manifest"
done
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--destination-root", required=True)
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--source-key", required=True)
    parser.add_argument("--source-port", type=int, default=45217)
    parser.add_argument("--poll-seconds", type=int, default=45)
    parser.add_argument("--ssh-connect-timeout", type=int, default=30)
    parser.add_argument("--ssh-command-timeout", type=int, default=120)
    parser.add_argument("--database-path", default="", help="Shared SQLite ledger path; defaults to STATE_ROOT/transfers.sqlite3.")
    parser.add_argument("--lock-path", default="", help="Per-worker lock path; defaults to STATE_ROOT/daemon.lock.")
    parser.add_argument("--partition-index", type=int, default=0)
    parser.add_argument("--partition-count", type=int, default=1)
    parser.add_argument("--max-episodes", type=int, default=0, help="Stop after this many new transfers; 0 means continuous.")
    parser.add_argument("--once", action="store_true", help="Scan and transfer currently visible successes, then exit.")
    return parser.parse_args()


def configure_logging(state_root: Path) -> None:
    state_root.mkdir(parents=True, exist_ok=True)
    log_path = state_root / "stream.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )


def init_database(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path, timeout=60)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS transfers (
            source_episode TEXT PRIMARY KEY,
            source_manifest TEXT NOT NULL,
            source_metadata TEXT NOT NULL,
            destination_episode TEXT NOT NULL,
            stage TEXT NOT NULL,
            task TEXT NOT NULL,
            profile TEXT NOT NULL,
            shard TEXT NOT NULL,
            seed INTEGER,
            layout_seed INTEGER,
            frame_count INTEGER,
            status TEXT NOT NULL,
            bytes INTEGER,
            started_at REAL,
            completed_at REAL,
            attempts INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            source_ssh_exit INTEGER
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    connection.commit()
    return connection


def is_safe_relative_path(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts and "" not in path.parts


def path_bytes(path: Path) -> int:
    total = 0
    for root, directories, files in os.walk(path, followlinks=False):
        for name in files:
            file_path = Path(root) / name
            try:
                if file_path.is_file():
                    total += file_path.stat().st_size
            except OSError:
                pass
    return total


def metadata_is_successful(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool((payload.get("metrics") or {}).get("success", False))


class Streamer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.source_root = args.source_root.rstrip("/")
        self.destination_root = Path(args.destination_root).resolve()
        self.state_root = Path(args.state_root).resolve()
        self.source_key = Path(args.source_key).resolve()
        self.source_port = args.source_port
        self.ssh_connect_timeout = args.ssh_connect_timeout
        self.ssh_command_timeout = max(10, args.ssh_command_timeout)
        self.poll_seconds = max(5, args.poll_seconds)
        self.max_episodes = max(0, args.max_episodes)
        self.once = args.once
        self.partition_index = int(args.partition_index)
        self.partition_count = int(args.partition_count)
        if self.partition_count < 1 or not 0 <= self.partition_index < self.partition_count:
            raise ValueError("partition-index must be in [0, partition-count)")
        self.database_path = Path(args.database_path).expanduser().resolve() if args.database_path else self.state_root / "transfers.sqlite3"
        self.lock_path = Path(args.lock_path).expanduser().resolve() if args.lock_path else self.state_root / "daemon.lock"
        self.running = True
        self.manifest_revisions: dict[str, str] = self.load_manifest_revisions()
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.destination_root.mkdir(parents=True, exist_ok=True)
        self.incoming_root = self.state_root / "incoming"
        self.incoming_root.mkdir(parents=True, exist_ok=True)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_handle = self.lock_path.open("a+")
        try:
            fcntl.flock(self.lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("another stream daemon is already running") from exc
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = init_database(self.database_path)
        self.ssh_base = [
            "ssh",
            "-i",
            str(self.source_key),
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"UserKnownHostsFile={self.state_root / 'known_hosts'}",
            "-o",
            f"ConnectTimeout={self.ssh_connect_timeout}",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
            "-p",
            str(self.source_port),
            "root@127.0.0.1",
        ]

    def load_manifest_revisions(self) -> dict[str, str]:
        path = Path(self.state_root / "manifest_revisions.json")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    def save_manifest_revisions(self) -> None:
        path = self.state_root / "manifest_revisions.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.manifest_revisions, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)

    def ssh_capture(self, remote_command: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [*self.ssh_base, remote_command],
            text=True,
            capture_output=True,
            check=False,
            timeout=self.ssh_command_timeout,
        )

    def source_listing(self) -> list[dict[str, Any]]:
        root = shlex.quote(self.source_root)
        find_command = (
            f"cd {root} && "
            r"find stage1 -mindepth 4 -maxdepth 4 -type f -name collection_manifest.json "
            r"-printf '%p\t%T@\t%s\n' && "
            r"find rendered -mindepth 5 -maxdepth 5 -type f -name replay_manifest.json "
            r"-printf '%p\t%T@\t%s\n'"
        )
        completed = self.ssh_capture(find_command)
        if completed.returncode != 0:
            detail = completed.stderr.strip()[-2000:]
            raise RuntimeError(f"source manifest scan failed ({completed.returncode}): {detail}")
        revisions: dict[str, str] = {}
        for line in completed.stdout.splitlines():
            try:
                relative_manifest, modified, size = line.split("\t", 2)
            except ValueError:
                logging.warning("ignored malformed source manifest listing: %r", line)
                continue
            if not is_safe_relative_path(relative_manifest):
                logging.warning("ignored unsafe source manifest path: %r", relative_manifest)
                continue
            revision = f"{modified}:{size}"
            if self.manifest_revisions.get(relative_manifest) == revision:
                continue
            revisions[relative_manifest] = revision
        if not revisions:
            return []
        jq_command = "bash -s -- " + " ".join(
            [shlex.quote(self.source_root), *[shlex.quote(path) for path in revisions]]
        )
        manifest_result = subprocess.run(
            [*self.ssh_base, jq_command],
            input=REMOTE_JQ_LISTING_SCRIPT,
            text=True,
            capture_output=True,
            check=False,
            timeout=self.ssh_command_timeout,
        )
        if manifest_result.returncode != 0:
            detail = manifest_result.stderr.strip()[-2000:]
            raise RuntimeError(f"source manifest filtering failed ({manifest_result.returncode}): {detail}")
        records: list[dict[str, Any]] = []
        for line in manifest_result.stdout.splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logging.warning("ignored malformed filtered manifest record")
                continue
            relative_manifest = str(record.get("source_manifest") or "")
            relative_metadata = str(record.get("source_metadata") or "")
            relative_episode = str(record.get("source_episode") or "")
            if relative_manifest not in revisions:
                continue
            if not is_safe_relative_path(relative_metadata) or not is_safe_relative_path(relative_episode):
                continue
            parts = PurePosixPath(relative_manifest).parts
            stage = parts[0] if parts and parts[0] in {"stage1", "rendered"} else "unknown"
            task = parts[1] if len(parts) > 1 else "unknown"
            profile = "position" if stage == "stage1" else (parts[2] if len(parts) > 2 else "unknown")
            shard = "unknown"
            if "shards" in parts:
                index = parts.index("shards")
                if index + 1 < len(parts):
                    shard = parts[index + 1]
            record.update({"stage": stage, "task": task, "profile": profile, "shard": shard})
            records.append(record)
        self.manifest_revisions.update(revisions)
        unique: dict[str, dict[str, Any]] = {}
        for record in records:
            source_episode = str(record.get("source_episode") or "")
            if is_safe_relative_path(source_episode) and self.owns(source_episode):
                unique[source_episode] = record
        return sorted(unique.values(), key=lambda item: (0 if item["stage"] == "stage1" else 1, item["source_episode"]))

    def owns(self, source_episode: str) -> bool:
        if self.partition_count == 1:
            return True
        digest = int(hashlib.sha256(source_episode.encode("utf-8")).hexdigest()[:16], 16)
        return digest % self.partition_count == self.partition_index

    def enqueue(self, records: list[dict[str, Any]]) -> None:
        for record in records:
            self.db.execute(
                """
                INSERT INTO transfers (
                    source_episode, source_manifest, source_metadata, destination_episode,
                    stage, task, profile, shard, seed, layout_seed, frame_count,
                    status, attempts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', 0)
                ON CONFLICT(source_episode) DO UPDATE SET
                    source_manifest=excluded.source_manifest,
                    source_metadata=excluded.source_metadata,
                    destination_episode=excluded.destination_episode,
                    stage=excluded.stage,
                    task=excluded.task,
                    profile=excluded.profile,
                    shard=excluded.shard,
                    seed=excluded.seed,
                    layout_seed=excluded.layout_seed,
                    frame_count=excluded.frame_count
                """,
                (
                    record["source_episode"],
                    record["source_manifest"],
                    record["source_metadata"],
                    record["source_episode"],
                    record["stage"],
                    record["task"],
                    record["profile"],
                    record["shard"],
                    record.get("seed"),
                    record.get("layout_seed"),
                    record.get("frame_count"),
                ),
            )
        self.db.commit()
        self.save_manifest_revisions()

    def pending_records(self) -> list[dict[str, Any]]:
        columns = (
            "source_episode", "source_manifest", "source_metadata", "stage", "task",
            "profile", "shard", "seed", "layout_seed", "frame_count",
        )
        rows = self.db.execute(
            f"SELECT {','.join(columns)} FROM transfers "
            "WHERE status NOT IN ('complete', 'recovered') "
            "ORDER BY CASE stage WHEN 'stage1' THEN 0 ELSE 1 END, source_episode"
        ).fetchall()
        return [dict(zip(columns, row)) for row in rows if self.owns(str(row[0]))]

    def status(self, source_episode: str) -> str | None:
        row = self.db.execute(
            "SELECT status FROM transfers WHERE source_episode = ?",
            (source_episode,),
        ).fetchone()
        return str(row[0]) if row else None

    def upsert_started(self, record: dict[str, Any]) -> None:
        now = time.time()
        self.db.execute(
            """
            INSERT INTO transfers (
                source_episode, source_manifest, source_metadata, destination_episode,
                stage, task, profile, shard, seed, layout_seed, frame_count,
                status, started_at, attempts, last_error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'transferring', ?, 1, NULL)
            ON CONFLICT(source_episode) DO UPDATE SET
                source_manifest=excluded.source_manifest,
                source_metadata=excluded.source_metadata,
                destination_episode=excluded.destination_episode,
                stage=excluded.stage,
                task=excluded.task,
                profile=excluded.profile,
                shard=excluded.shard,
                seed=excluded.seed,
                layout_seed=excluded.layout_seed,
                frame_count=excluded.frame_count,
                status='transferring',
                started_at=excluded.started_at,
                attempts=transfers.attempts + 1,
                last_error=NULL
            """,
            (
                record["source_episode"],
                record["source_manifest"],
                record["source_metadata"],
                record["source_episode"],
                record["stage"],
                record["task"],
                record["profile"],
                record["shard"],
                record.get("seed"),
                record.get("layout_seed"),
                record.get("frame_count"),
                now,
            ),
        )
        self.db.commit()

    def mark_failed(self, source_episode: str, error: str, ssh_exit: int | None = None) -> None:
        self.db.execute(
            "UPDATE transfers SET status='failed', last_error=?, source_ssh_exit=? WHERE source_episode=?",
            (error[-4000:], ssh_exit, source_episode),
        )
        self.db.commit()

    def mark_complete(self, record: dict[str, Any], bytes_count: int, status: str = "complete") -> None:
        self.db.execute(
            """
            UPDATE transfers
            SET status=?, bytes=?, completed_at=?, last_error=NULL
            WHERE source_episode=?
            """,
            (status, bytes_count, time.time(), record["source_episode"]),
        )
        self.db.commit()

    def destination_path(self, source_episode: str) -> Path:
        return self.destination_root / Path(*PurePosixPath(source_episode).parts)

    def recover_or_skip(self, record: dict[str, Any]) -> bool:
        source_episode = record["source_episode"]
        destination = self.destination_path(source_episode)
        state = self.status(source_episode)
        if state in {"complete", "recovered"}:
            if (destination / "metadata.json").is_file():
                return True
            logging.warning("ledger says complete but destination is missing; retrying %s", source_episode)
        if destination.is_dir() and (destination / "metadata.json").is_file() and metadata_is_successful(destination / "metadata.json"):
            if state != "complete":
                self.upsert_started(record)
                self.mark_complete(record, path_bytes(destination), status="recovered")
                logging.info("recovered previously transferred episode: %s", source_episode)
            return True
        return False

    def transfer_one(self, record: dict[str, Any]) -> bool:
        source_episode = record["source_episode"]
        if self.recover_or_skip(record):
            return False
        self.upsert_started(record)
        key = hashlib.sha256(source_episode.encode("utf-8")).hexdigest()[:32]
        stage_root = self.incoming_root / key
        staged_episode = stage_root / Path(*PurePosixPath(source_episode).parts)
        destination = self.destination_path(source_episode)
        shutil.rmtree(stage_root, ignore_errors=True)
        stage_root.mkdir(parents=True, exist_ok=True)
        source_command = [*self.ssh_base, "tar", "-C", self.source_root, "-cf", "-", "--", source_episode]
        logging.info("transferring %s (%s/%s/%s seed=%s)", source_episode, record["task"], record["profile"], record["shard"], record.get("seed"))
        source_process = subprocess.Popen(
            source_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert source_process.stdout is not None
        extractor = subprocess.Popen(
            ["tar", "-C", str(stage_root), "-xf", "-", "--no-same-owner", "--no-same-permissions"],
            stdin=source_process.stdout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        source_process.stdout.close()
        _, extractor_error = extractor.communicate()
        source_error = source_process.stderr.read() if source_process.stderr is not None else b""
        source_exit = source_process.wait()
        if source_exit != 0 or extractor.returncode != 0:
            detail = (source_error + extractor_error)[-4000:].decode("utf-8", "replace").strip()
            shutil.rmtree(stage_root, ignore_errors=True)
            self.mark_failed(source_episode, detail or "tar stream failed", source_exit)
            logging.error("transfer failed source_exit=%s extractor_exit=%s episode=%s %s", source_exit, extractor.returncode, source_episode, detail)
            return False
        staged_metadata = staged_episode / "metadata.json"
        if not staged_episode.is_dir() or not staged_metadata.is_file():
            shutil.rmtree(stage_root, ignore_errors=True)
            self.mark_failed(source_episode, "extracted episode is incomplete: metadata.json is missing", source_exit)
            logging.error("incomplete extracted episode: %s", source_episode)
            return False
        if not metadata_is_successful(staged_metadata):
            shutil.rmtree(stage_root, ignore_errors=True)
            self.mark_failed(source_episode, "extracted metadata does not report metrics.success=true", source_exit)
            logging.error("source manifest success failed metadata verification: %s", source_episode)
            return False
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.is_dir() and (destination / "metadata.json").is_file() and metadata_is_successful(destination / "metadata.json"):
                shutil.rmtree(stage_root, ignore_errors=True)
                self.mark_complete(record, path_bytes(destination), status="recovered")
                return False
            shutil.rmtree(stage_root, ignore_errors=True)
            self.mark_failed(source_episode, f"destination already exists and is not a successful episode: {destination}", source_exit)
            return False
        os.replace(staged_episode, destination)
        shutil.rmtree(stage_root, ignore_errors=True)
        bytes_count = path_bytes(destination)
        self.mark_complete(record, bytes_count)
        logging.info("transfer complete bytes=%d episode=%s", bytes_count, source_episode)
        return True

    def run(self) -> None:
        transferred = 0
        consecutive_errors = 0
        while self.running:
            try:
                records = self.source_listing()
                self.enqueue(records)
                pending = self.pending_records()
                logging.info("scan: newly_discovered=%d pending=%d transferred_this_run=%d", len(records), len(pending), transferred)
                if pending and self.running:
                    if self.transfer_one(pending[0]):
                        transferred += 1
                        if self.max_episodes and transferred >= self.max_episodes:
                            return
                consecutive_errors = 0
                if not pending and self.once:
                    return
                if not pending:
                    time.sleep(self.poll_seconds)
            except Exception as exc:
                consecutive_errors += 1
                logging.exception("stream loop failed (%d consecutive): %s", consecutive_errors, exc)
                if self.once:
                    raise
                time.sleep(min(self.poll_seconds * min(consecutive_errors, 10), 300))

    def stop(self, *_args: Any) -> None:
        self.running = False


def main() -> int:
    args = parse_args()
    configure_logging(Path(args.state_root).resolve())
    streamer = Streamer(args)
    signal.signal(signal.SIGTERM, streamer.stop)
    signal.signal(signal.SIGINT, streamer.stop)
    logging.info(
        "stream daemon started source=%s destination=%s poll=%ss partition=%d/%d ledger=%s",
        args.source_root,
        args.destination_root,
        args.poll_seconds,
        args.partition_index,
        args.partition_count,
        args.database_path or str(Path(args.state_root).resolve() / "transfers.sqlite3"),
    )
    try:
        streamer.run()
    finally:
        streamer.db.close()
        streamer.lock_handle.close()
        logging.info("stream daemon stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
