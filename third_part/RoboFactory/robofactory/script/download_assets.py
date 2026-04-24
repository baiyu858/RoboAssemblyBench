import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from huggingface_hub import logging as hf_logging
from huggingface_hub import snapshot_download
from huggingface_hub.utils.tqdm import tqdm as hf_tqdm


LOGGER = logging.getLogger("download_assets")


class LoggingTqdm(hf_tqdm):
    logger = LOGGER
    log_interval_seconds = 5.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_log_time = 0.0
        self._last_log_message = ""
        self._emit_log(force=True)

    def update(self, n=1):
        displayed = super().update(n)
        self._emit_log()
        return displayed

    def close(self):
        try:
            self._emit_log(force=True)
        except Exception:
            pass
        super().close()

    def _emit_log(self, force=False):
        if self.disable:
            return

        now = time.monotonic()
        if not force and now - self._last_log_time < self.log_interval_seconds:
            return

        message = self._format_progress_message()
        if force or message != self._last_log_message:
            self.logger.info(message)
            self._last_log_time = now
            self._last_log_message = message

    def _format_progress_message(self):
        desc = self.desc or "progress"
        unit = self.unit or "it"
        current = self._format_value(self.n)
        total_is_known = self.total is not None and self.total > 0
        suffix = f"{current} {unit}"

        if total_is_known:
            total = self._format_value(self.total)
            percent = (self.n / self.total) * 100
            suffix = f"{current}/{total} {unit} ({percent:.1f}%)"

        rate = self.format_dict.get("rate")
        if rate:
            suffix = f"{suffix}, {rate:.2f} {unit}/s"

        return f"{desc}: {suffix}"

    @staticmethod
    def _format_value(value):
        if value is None:
            return "?"
        if isinstance(value, int):
            return str(value)
        if float(value).is_integer():
            return str(int(value))
        return f"{value:.2f}"


def parse_args():
    parser = argparse.ArgumentParser(description="Download RoboFactory assets from Hugging Face.")
    parser.add_argument("--repo-id", default="sparklexfantasy/RoboFactory_asset")
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--local-dir", default="./assets")
    parser.add_argument("--log-file", default=None, help="Path to the log file. Defaults to logs/download_assets_<timestamp>.log")
    parser.add_argument("--log-interval", type=float, default=5.0, help="Seconds between progress log updates.")
    return parser.parse_args()


def build_log_path(log_file):
    if log_file:
        return Path(log_file)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("logs") / f"download_assets_{timestamp}.log"


def configure_logging(log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
    root_logger.addHandler(file_handler)

    hf_logging.set_verbosity_info()
    hf_logger = logging.getLogger("huggingface_hub")
    hf_logger.handlers.clear()
    hf_logger.propagate = True
    hf_logger.setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def main():
    args = parse_args()
    log_path = build_log_path(args.log_file).resolve()

    LoggingTqdm.log_interval_seconds = args.log_interval
    configure_logging(log_path)

    print(f"Logging progress to {log_path}", flush=True)
    LOGGER.info(
        "Starting asset download: repo_id=%s repo_type=%s local_dir=%s",
        args.repo_id,
        args.repo_type,
        Path(args.local_dir).resolve(),
    )

    snapshot_download(
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        repo_type=args.repo_type,
        tqdm_class=LoggingTqdm,
    )

    LOGGER.info("Asset download completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOGGER.exception("Asset download failed.")
        raise
