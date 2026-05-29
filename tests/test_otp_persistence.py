import pytest
import sqlite3
from app.database.connection import db_conn, init_db
import app.database.connection as db_mod
from app.services.user_auth import send_email_otp, verify_email_otp

@pytest.fixture(autouse=True)
def use_test_db(tmp_path, monkeypatch):
    """Redirect DB to a temp file for each test."""
    db_path = str(tmp_path / "test_otp.db")
    monkeypatch.setattr(db_mod, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(db_mod, "DB_FILE", db_path)
    db_mod.init_db()
    yield

def test_otp_persistence_and_lockout():
    email = "persistent_otp@example.com"
    
    # 1. Generate and persist OTP
    otp = send_email_otp(email)
    assert len(otp) == 6
    assert otp.isdigit()
    
    # Verify it exists in database
    with db_conn() as conn:
        row = conn.execute("SELECT result_json FROM ai_cache WHERE cache_key=?", (f"otp:{email}",)).fetchone()
        assert row is not None
        
    # 2. Verify incorrect OTP increments attempt counter
    res = verify_email_otp(email, "000000")
    assert res is None
    
    # Verify attempts incremented in DB
    with db_conn() as conn:
        row = conn.execute("SELECT result_json FROM ai_cache WHERE cache_key=?", (f"otp:{email}",)).fetchone()
        assert row is not None
        import json
        entry = json.loads(row["result_json"])
        assert entry["attempts"] == 1
        
    # 3. Complete lockout after 5 failed attempts
    for _ in range(4):
        verify_email_otp(email, "000000")
        
    # The 6th verify (5th failure was reached) should lockout and delete OTP from DB
    res_lockout = verify_email_otp(email, "000000")
    assert res_lockout is None
    
    with db_conn() as conn:
        row = conn.execute("SELECT result_json FROM ai_cache WHERE cache_key=?", (f"otp:{email}",)).fetchone()
        assert row is None
        
    # 4. Generate new OTP and test successful verification
    new_otp = send_email_otp(email)
    user = verify_email_otp(email, new_otp)
    assert user is not None
    assert user["email"] == email
    
    # Successful verify should delete OTP from DB
    with db_conn() as conn:
        row = conn.execute("SELECT result_json FROM ai_cache WHERE cache_key=?", (f"otp:{email}",)).fetchone()
        assert row is None
