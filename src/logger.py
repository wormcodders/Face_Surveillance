# presence logger: keeps an SQLite DB with sessions (first_seen,last_seen,duration) and helper to export CSV
import sqlite3
import os
from datetime import datetime

class PresenceLogger:
    def __init__(self, db_path="logs/face_presence.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_table()
        self.active = {}  # (name, cam) -> {'first':str, 'last':str}

    def _init_table(self):
        c = self.conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS presence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                camera TEXT,
                first_seen TEXT,
                last_seen TEXT,
                duration REAL
            )
        """)
        self.conn.commit()

    def _now(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def update(self, name, camera):
        now = self._now()
        key = (name, camera)
        c = self.conn.cursor()
        if key not in self.active:
            # insert
            self.active[key] = {'first': now, 'last': now}
            c.execute("INSERT INTO presence (name, camera, first_seen, last_seen, duration) VALUES (?,?,?,?,?)",
                      (name, camera, now, now, 0.0))
            self.conn.commit()
        else:
            self.active[key]['last'] = now
            first = datetime.strptime(self.active[key]['first'], "%Y-%m-%d %H:%M:%S")
            last = datetime.strptime(self.active[key]['last'], "%Y-%m-%d %H:%M:%S")
            dur = (last - first).total_seconds()
            # update latest row for this person+camera
            c.execute("""
                UPDATE presence
                SET last_seen = ?, duration = ?
                WHERE id = (SELECT id FROM presence WHERE name=? AND camera=? ORDER BY id DESC LIMIT 1)
            """, (now, dur, name, camera))
            self.conn.commit()

    def fetch(self, limit=500):
        c = self.conn.cursor()
        rows = c.execute("SELECT id, name, camera, first_seen, last_seen, duration FROM presence ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return rows

    def export_csv(self, out_path="logs/presence_export.csv"):
        rows = self.fetch(limit=1000000)
        import csv
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id","name","camera","first_seen","last_seen","duration"])
            w.writerows(rows)
        return out_path
