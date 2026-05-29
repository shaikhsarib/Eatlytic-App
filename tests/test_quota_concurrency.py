import pytest
import sqlite3
import threading
from app.database.connection import db_conn, init_db, check_and_increment_scan
import app.database.connection as db_mod

@pytest.fixture(autouse=True)
def use_test_db(tmp_path, monkeypatch):
    """Redirect DB to a temp file for each test."""
    db_path = str(tmp_path / "test_quota.db")
    monkeypatch.setattr(db_mod, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(db_mod, "DB_FILE", db_path)
    db_mod.init_db()
    yield

def test_check_and_increment_scan_quota():
    device_key = "concurrency_device_99"
    limit = 5
    
    # 1. Increment scans sequentially up to limit
    for i in range(5):
        status = check_and_increment_scan(device_key, limit=limit, increment=True)
        assert status["allowed"] is True
        assert status["scans_used"] == i + 1
        assert status["scans_remaining"] == limit - (i + 1)
        assert status["is_pro"] is False
        
    # 6th attempt should be blocked
    blocked_status = check_and_increment_scan(device_key, limit=limit, increment=True)
    assert blocked_status["allowed"] is False
    assert blocked_status["scans_used"] == 5
    assert blocked_status["scans_remaining"] == 0
    
def test_quota_multithreaded_concurrency():
    """Verify that multiple threads concurrently checking/incrementing do not violate SQLite transaction integrity."""
    device_key = "threaded_device_88"
    limit = 20
    
    results = []
    
    def worker():
        try:
            status = check_and_increment_scan(device_key, limit=limit, increment=True)
            results.append(status)
        except Exception as e:
            results.append({"error": str(e)})

    threads = [threading.Thread(target=worker) for _ in range(15)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
        
    # Assert all operations succeeded without any locking or database locked errors
    for r in results:
        assert "error" not in r
        
    # Assert database scan count matches number of threads
    with db_conn() as conn:
        row = conn.execute("SELECT scan_count FROM devices WHERE device_key=?", (device_key,)).fetchone()
        assert row is not None
        assert row["scan_count"] == 15
