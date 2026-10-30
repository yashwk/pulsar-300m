#!/usr/bin/env python3
"""High-performance in-process telemetry synchronization from Modal persistent volume.

Syncs live telemetry (live_status.json, metrics.jsonl, metrics.csv, train.log,
run_config.json, config.json) from the remote Modal 'pulsar-checkpoints' volume
into the local flat checkpoints directory.

Features:
- Native in-process Modal Python SDK (zero CLI subprocess spawning overhead).
- Metadata-driven delta sync (only downloads files when mtime or size changes).
- Atomic temporary file replacement (prevents torn reads by monitor readers).
- Thread-safe stop event for embedded execution within monitor/web_server.py.
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from typing import Dict, Tuple

FILES_TO_SYNC = (
    "live_status.json",
    "metrics.jsonl",
    "metrics.csv",
    "train.log",
    "run_config.json",
    "config.json",
)


class CloudVolumeSyncer:
    """In-process synchronizer reading telemetry from a Modal Volume."""

    def __init__(self, checkpoint_dir: str = "checkpoints", volume_name: str = "pulsar-checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        self.volume_name = volume_name
        self._cached_meta: Dict[str, Tuple[int, int]] = {}
        self._vol = None
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _get_volume(self):
        if self._vol is None:
            import modal
            self._vol = modal.Volume.from_name(self.volume_name, create_if_missing=True)
        return self._vol

    def sync_once(self) -> int:
        """Perform a single metadata-driven synchronization pass.

        Returns:
            Number of files downloaded and updated.
        """
        vol = self._get_volume()
        try:
            entries = vol.listdir("/")
        except Exception as exc:
            print(f"[Sync Monitor Warning] Failed to query remote volume: {exc}")
            return 0

        remote_files = {e.path: e for e in entries if e.path in FILES_TO_SYNC}
        updated_count = 0

        for fname, entry in remote_files.items():
            last_meta = self._cached_meta.get(fname)
            current_meta = (int(getattr(entry, "size", 0)), int(getattr(entry, "mtime", 0)))

            dest = os.path.join(self.checkpoint_dir, fname)
            # If local file is missing, force download even if cached_meta matches
            if last_meta == current_meta and os.path.exists(dest):
                continue

            tmp_dest = dest + ".tmp"
            try:
                with open(tmp_dest, "wb") as f:
                    vol.read_file_into_fileobj(fname, f)
                os.replace(tmp_dest, dest)
                self._cached_meta[fname] = current_meta
                updated_count += 1
            except Exception as exc:
                if os.path.exists(tmp_dest):
                    try:
                        os.remove(tmp_dest)
                    except OSError:
                        pass
                print(f"[Sync Monitor Warning] Failed to sync '{fname}': {exc}")

        return updated_count


def sync_cloud_telemetry(
    checkpoint_dir: str = "checkpoints",
    interval: float = 5.0,
    stop_event: threading.Event | None = None,
    volume_name: str = "pulsar-checkpoints",
):
    """Run persistent telemetry synchronization loop."""
    syncer = CloudVolumeSyncer(checkpoint_dir=checkpoint_dir, volume_name=volume_name)
    print(f"[Sync Monitor] Starting in-process SDK sync from Modal '{volume_name}' -> '{checkpoint_dir}' (every {interval}s)")

    while stop_event is None or not stop_event.is_set():
        t0 = time.time()
        updated = syncer.sync_once()
        elapsed_ms = (time.time() - t0) * 1000
        if updated > 0:
            print(f"[Sync Monitor] Updated {updated} file(s) in {elapsed_ms:.1f}ms")

        # Sleep with granular interrupt checks if stop_event is provided
        if stop_event is not None:
            if stop_event.wait(timeout=interval):
                break
        else:
            time.sleep(interval)


def start_sync_thread(
    checkpoint_dir: str = "checkpoints",
    interval: float = 5.0,
    stop_event: threading.Event | None = None,
    volume_name: str = "pulsar-checkpoints",
) -> threading.Thread:
    """Launch synchronization in a daemon thread."""
    thread = threading.Thread(
        target=sync_cloud_telemetry,
        kwargs={
            "checkpoint_dir": checkpoint_dir,
            "interval": interval,
            "stop_event": stop_event,
            "volume_name": volume_name,
        },
        daemon=True,
    )
    thread.start()
    return thread


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync telemetry from Modal volume to local directory")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Local directory to write synced telemetry (default: checkpoints)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Sync interval in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--volume-name",
        type=str,
        default="pulsar-checkpoints",
        help="Modal volume name (default: pulsar-checkpoints)",
    )
    args = parser.parse_args()
    sync_cloud_telemetry(
        checkpoint_dir=args.checkpoint_dir,
        interval=args.interval,
        volume_name=args.volume_name,
    )
