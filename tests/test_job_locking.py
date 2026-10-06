import json
import os
import subprocess
import sys
from pathlib import Path

from job_locking import advisory_file_lock, advisory_file_lock_held


def test_leftover_lock_file_is_not_a_lock(tmp_path):
    lock_path = tmp_path / "weekly_positioning.lock"
    lock_path.write_text("1", encoding="utf-8")

    with advisory_file_lock(lock_path):
        assert advisory_file_lock_held(lock_path)
        contents = lock_path.read_text(encoding="utf-8")
        metadata = json.loads(contents.split("\n", 1)[1] if os.name == "nt" else contents)
        assert metadata["pid"] > 0
        assert metadata["host"]
        assert metadata["acquired_at_utc"]

    # The stable file remains for diagnostics, but release of the OS lock lets
    # the next scheduled attempt acquire it immediately.
    assert lock_path.exists()
    assert not advisory_file_lock_held(lock_path)
    with advisory_file_lock(lock_path):
        contents = lock_path.read_text(encoding="utf-8")
        metadata = json.loads(contents.split("\n", 1)[1] if os.name == "nt" else contents)
        assert metadata["pid"] > 0


def test_leftover_file_does_not_appear_to_be_running(tmp_path):
    lock_path = tmp_path / "nightly_analytics.lock"
    lock_path.write_text('{"pid": 1}', encoding="utf-8")
    assert not advisory_file_lock_held(lock_path)


def test_active_lock_rejects_a_second_owner(tmp_path):
    lock_path = tmp_path / "nightly_analytics.lock"
    with advisory_file_lock(lock_path):
        code = (
            "from job_locking import advisory_file_lock; from pathlib import Path; import sys; "
            "try:\n with advisory_file_lock(Path(sys.argv[1])): sys.exit(1)\n"
            "except FileExistsError: sys.exit(0)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code, str(lock_path)],
            cwd=Path(__file__).resolve().parents[1],
            check=False,
        )
        assert result.returncode == 0
