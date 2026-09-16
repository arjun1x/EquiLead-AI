"""Access control, rate limiting, secrets at rest, retention, and hash-chain verification.

Command line:
  python security.py keygen                         print a new SECRETS_KEY
  python security.py encrypt-secrets KEY=VALUE ...  write secrets.enc (needs SECRETS_KEY in the environment)
  python security.py set-role EMAIL ROLE            analyst | reviewer | admin
  python security.py purge [--days N]               delete decisions older than N days (default RETENTION_DAYS)
  python security.py verify                         check both hash chains
"""
import datetime as dt
import json
import os
import sys
import threading
import time
from pathlib import Path

from models import ROLES

ROOT = Path(__file__).resolve().parent


# ── Rate limiting ────────────────────────────────────────

class RateLimiter:
    """Fixed-window counter per key, held in memory. Correct for one process; put a shared store
    (Redis or the database) behind the same interface when running several workers."""

    def __init__(self, limit, window=60):
        self.limit = int(limit)
        self.window = window
        self._hits = {}
        self._lock = threading.Lock()

    def allow(self, key):
        if self.limit <= 0:
            return True
        bucket = int(time.monotonic() // self.window)
        with self._lock:
            if len(self._hits) > 10_000:
                self._hits = {k: v for k, v in self._hits.items() if v[0] == bucket}
            stored_bucket, count = self._hits.get(key, (bucket, 0))
            if stored_bucket != bucket:
                count = 0
            count += 1
            self._hits[key] = (bucket, count)
            return count <= self.limit

    def reset(self):
        with self._lock:
            self._hits.clear()


limiter = RateLimiter(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))


def client_key(request):
    host = request.client.host if request.client else "unknown"
    if os.getenv("TRUST_PROXY") == "1":
        forwarded = request.headers.get("x-forwarded-for", "")
        host = (forwarded.split(",")[0].strip() or host) if forwarded else host
    return host


# ── Roles ────────────────────────────────────────────────

def role_rank(role):
    return ROLES.get(role or "analyst", ROLES["analyst"])


def has_role(user, minimum):
    return bool(user) and role_rank(user.get("role")) >= ROLES[minimum]


# ── Secrets at rest ──────────────────────────────────────

def _fernet(key):
    try:
        from cryptography.fernet import Fernet
    except ImportError as exc:
        raise RuntimeError("Install the cryptography package to use encrypted secrets.") from exc
    return Fernet(key.encode() if isinstance(key, str) else key)


def keygen():
    from cryptography.fernet import Fernet
    return Fernet.generate_key().decode()


def write_secrets(key, path, values):
    Path(path).write_bytes(_fernet(key).encrypt(json.dumps(values).encode()))


def read_secrets(key, path):
    return json.loads(_fernet(key).decrypt(Path(path).read_bytes()))


def load_secrets(env=os.environ):
    """Decrypt secrets.enc into the environment for any variable that is not already set.
    Fails closed: a configured key with a missing file, or a missing crypto library, stops startup."""
    key = env.get("SECRETS_KEY")
    if not key:
        return {}
    path = Path(env.get("SECRETS_FILE") or ROOT / "secrets.enc")
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise RuntimeError(f"SECRETS_KEY is set but {path.name} was not found.")
    values = read_secrets(key, path)
    for name, value in values.items():
        if not env.get(name):
            env[name] = str(value)
    return values


# ── Retention ────────────────────────────────────────────

def purge_expired(days=None, actor="system"):
    """Delete decision records older than the retention window. The audit chain keeps the hashes of
    the deleted records so the decision chain still verifies."""
    from sqlalchemy import select
    from models import Session, LoanDecision, append_audit, utcnow
    days = int(os.getenv("RETENTION_DAYS", "0")) if days is None else int(days)
    if days <= 0:
        return 0
    cutoff = utcnow().replace(tzinfo=None) - dt.timedelta(days=days)
    with Session() as db:
        rows = db.scalars(select(LoanDecision).where(LoanDecision.created_at < cutoff)).all()
        ids = [row.id for row in rows]
        hashes = [row.record_hash for row in rows if row.record_hash]
        for row in rows:
            db.delete(row)
        append_audit(db, actor, "retention_purge", None, {"days": days, "deleted": len(ids), "ids": ids, "hashes": hashes})
        db.commit()
    return len(ids)


# ── Command line ─────────────────────────────────────────

def _main(argv):
    command = argv[0] if argv else "help"
    if command == "keygen":
        print(keygen())
    elif command == "encrypt-secrets":
        key = os.getenv("SECRETS_KEY")
        if not key:
            sys.exit("Set SECRETS_KEY first (python security.py keygen).")
        values = dict(item.split("=", 1) for item in argv[1:])
        path = ROOT / os.getenv("SECRETS_FILE", "secrets.enc")
        write_secrets(key, path, values)
        print(f"Wrote {len(values)} secret(s) to {path.name}. Do not commit it.")
    elif command == "set-role":
        from models import Session, User, init_db, append_audit
        email, role = argv[1].strip().lower(), argv[2]
        if role not in ROLES:
            sys.exit(f"Role must be one of: {', '.join(ROLES)}")
        init_db()
        with Session() as db:
            account = db.get(User, email)
            if not account:
                sys.exit("No such user.")
            account.role = role
            append_audit(db, "cli", "role_changed", None, {"email": email, "role": role})
            db.commit()
        print(f"{email} is now {role}.")
    elif command == "purge":
        from models import init_db
        init_db()
        days = int(argv[argv.index("--days") + 1]) if "--days" in argv else None
        print(f"Deleted {purge_expired(days, actor='cli')} record(s).")
    elif command == "verify":
        from models import Session, init_db, verify_decision_chain, verify_audit_chain
        init_db()
        with Session() as db:
            print("decisions:", verify_decision_chain(db))
            print("audit    :", verify_audit_chain(db))
    else:
        print(__doc__)


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    _main(sys.argv[1:])
