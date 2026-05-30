import json
import logging
from app.database.connection import db_conn, _supabase, _supabase_execute_with_retry

logger = logging.getLogger(__name__)

def get_ai_cache(key: str):
    if _supabase:
        try:
            res = _supabase_execute_with_retry(lambda: _supabase.table("ai_cache").select("*").eq("cache_key", key).execute())
            if res.data: return json.loads(res.data[0]["result_json"])
        except Exception: pass
    with db_conn() as c:
        row = c.execute("SELECT result_json FROM ai_cache WHERE cache_key=?", (key,)).fetchone()
    return json.loads(row["result_json"]) if row else None

def set_ai_cache(key: str, value: dict):
    if _supabase:
        try: _supabase_execute_with_retry(lambda: _supabase.table("ai_cache").upsert({"cache_key": key, "result_json": json.dumps(value)}).execute())
        except: pass
    with db_conn() as c:
        c.execute("INSERT OR REPLACE INTO ai_cache(cache_key,result_json) VALUES(?,?)", (key, json.dumps(value)))

def get_ocr_cache(key: str):
    if _supabase:
        try:
            res = _supabase_execute_with_retry(lambda: _supabase.table("ocr_cache").select("*").eq("cache_key", key).execute())
            if res.data: return json.loads(res.data[0]["result_json"])
        except Exception: pass
    with db_conn() as c:
        row = c.execute("SELECT result_json FROM ocr_cache WHERE cache_key=?", (key,)).fetchone()
    return json.loads(row["result_json"]) if row else None

def set_ocr_cache(key: str, value: dict):
    if _supabase:
        try: _supabase_execute_with_retry(lambda: _supabase.table("ocr_cache").upsert({"cache_key": key, "result_json": json.dumps(value)}).execute())
        except: pass
    with db_conn() as c:
        c.execute("INSERT OR REPLACE INTO ocr_cache(cache_key,result_json) VALUES(?,?)", (key, json.dumps(value)))

_bktree = None
_bktree_lock = None

def _get_bktree():
    global _bktree, _bktree_lock
    if _bktree_lock is None:
        import threading
        _bktree_lock = threading.Lock()
    if _bktree is None:
        with _bktree_lock:
            if _bktree is None:
                from app.ai.perception.bk_tree import BKTree
                tree = BKTree()
                with db_conn() as c:
                    rows = c.execute("SELECT hash_key, result_json FROM image_fingerprints").fetchall()
                for row in rows:
                    tree.insert(row["hash_key"], row["result_json"])
                _bktree = tree
    return _bktree

def get_image_fingerprint_match(hash_key: str):
    if not hash_key:
        return None
    with db_conn() as c:
        row = c.execute("SELECT result_json FROM image_fingerprints WHERE hash_key=?", (hash_key,)).fetchone()
    if row:
        return json.loads(row["result_json"])
        
    tree = _get_bktree()
    matches = tree.search(hash_key, max_distance=6)
    if matches:
        matches.sort(key=lambda x: x[0])
        return json.loads(matches[0][1])
    return None

def set_image_fingerprint(hash_key: str, value: dict):
    if _supabase:
        try: _supabase_execute_with_retry(lambda: _supabase.table("image_fingerprints").upsert({"hash_key": hash_key, "result_json": json.dumps(value)}).execute())
        except: pass
    with db_conn() as c:
        c.execute("INSERT OR REPLACE INTO image_fingerprints(hash_key, result_json) VALUES(?,?)", (hash_key, json.dumps(value)))
    tree = _get_bktree()
    tree.insert(hash_key, json.dumps(value))

def get_research_cache(query: str):
    if _supabase:
        try:
            res = _supabase_execute_with_retry(lambda: _supabase.table("research_cache").select("*").eq("query", query).execute())
            if res.data: return res.data[0]["result"]
        except Exception: pass
    with db_conn() as c:
        row = c.execute("SELECT result FROM research_cache WHERE query=?", (query,)).fetchone()
    return row["result"] if row else None

def set_research_cache(query: str, result: str):
    if _supabase:
        try: _supabase_execute_with_retry(lambda: _supabase.table("research_cache").upsert({"query": query, "result": result}).execute())
        except: pass
    with db_conn() as c:
        c.execute("INSERT OR REPLACE INTO research_cache(query, result) VALUES(?,?)", (query, result))
