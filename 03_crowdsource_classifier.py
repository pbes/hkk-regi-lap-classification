"""
Crowdsource classifier Flask webapp.

Logic:
- Cards with ID < 9405 need human classification.
- AI predictions come from card_flags_predicted.csv.
- A card is "closed" when VOTES_REQUIRED matching votes exist (AI counts as one voter).
- Users identified by a random token stored in localStorage (client-side).
- Each user can vote on a card only once.
- Points are awarded retroactively when a card is closed: voters who match the
  final label set get +1 point.
- Top 10 leaderboard.
"""

import json
import os
import random
import sqlite3
from pathlib import Path

import pandas as pd
from flask import Flask, g, jsonify, render_template, request

app = Flask(__name__, template_folder="templates")

# How many matching votes are required to close a card.
# AI prediction counts as one vote. Set to 2 or 3.
VOTES_REQUIRED = 2

DB_PATH = Path(__file__).parent / "crowdsource.db"
CSV_PREDICTIONS = Path(__file__).parent / "card_flags_predicted.csv"
CSV_ALL_CARDS = Path(__file__).parent / "card_flags.csv"

ALL_FLAGS = [
    "token", "visszavétel", "semmizés", "lopás", "dobatás",
    "gyógyulás", "sebzés", "erőforrás", "keresés", "húzás",
    "reakció", "leszedés",
]


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        g.db = sqlite3.connect(str(DB_PATH))
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA journal_mode=WAL")
    return g.db


@app.teardown_appcontext
def close_db(exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    """Create tables and import AI predictions."""
    db = sqlite3.connect(str(DB_PATH))
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS cards (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            text TEXT NOT NULL,
            is_reaction INTEGER NOT NULL DEFAULT 0,
            ai_flags TEXT NOT NULL DEFAULT '',
            final_flags TEXT DEFAULT NULL,
            closed INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS votes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            card_id INTEGER NOT NULL,
            user_token TEXT NOT NULL,
            flags TEXT NOT NULL DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(card_id, user_token)
        );
        CREATE TABLE IF NOT EXISTS user_points (
            user_token TEXT PRIMARY KEY,
            points INTEGER NOT NULL DEFAULT 0
        );
    """)

    # Import cards if empty
    count = db.execute("SELECT COUNT(*) FROM cards").fetchone()[0]
    if count == 0:
        # Load all cards
        df = pd.read_csv(CSV_ALL_CARDS, encoding="utf-8")
        df = df[df["id"] < 9405].copy()
        df["text"] = df["text"].fillna("")
        df["flags"] = df["flags"].fillna("")
        df["is_reaction"] = df["is_reaction"].map({"True": 1, "False": 0, True: 1, False: 0}).fillna(0).astype(int)

        # Load AI predictions
        ai_flags = {}
        if CSV_PREDICTIONS.exists():
            pred_df = pd.read_csv(CSV_PREDICTIONS, encoding="utf-8")
            pred_df["predicted_flags"] = pred_df["predicted_flags"].fillna("")
            for _, row in pred_df.iterrows():
                ai_flags[int(row["id"])] = row["predicted_flags"]

        for _, row in df.iterrows():
            card_id = int(row["id"])
            db.execute(
                "INSERT OR IGNORE INTO cards (id, name, text, is_reaction, ai_flags) VALUES (?, ?, ?, ?, ?)",
                (card_id, row["name"], row["text"], int(row["is_reaction"]), ai_flags.get(card_id, "")),
            )
        db.commit()

    db.close()


def _normalize_flags(flags_str: str) -> str:
    """Sort and deduplicate flags for comparison."""
    flags = sorted(set(f.strip() for f in flags_str.split("|") if f.strip()))
    return "|".join(flags)


def _try_close_card(db: sqlite3.Connection, card_id: int):
    """Check if the card can be closed (2 matching votes from different voters)."""
    card = db.execute("SELECT * FROM cards WHERE id = ?", (card_id,)).fetchone()
    if not card or card["closed"]:
        return

    votes = db.execute(
        "SELECT user_token, flags FROM votes WHERE card_id = ?", (card_id,)
    ).fetchall()

    # Collect all votes including AI as a pseudo-voter
    all_votes: list[tuple[str, str]] = []
    ai_flags = _normalize_flags(card["ai_flags"])
    if ai_flags:
        all_votes.append(("__AI__", ai_flags))

    for v in votes:
        all_votes.append((v["user_token"], _normalize_flags(v["flags"])))

    # Find flags that at least 2 different voters agree on
    from collections import Counter
    vote_groups: dict[str, list[str]] = {}
    for token, flags in all_votes:
        vote_groups.setdefault(flags, []).append(token)

    for flags, voters in vote_groups.items():
        unique_voters = set(voters)
        if len(unique_voters) >= VOTES_REQUIRED:
            # Card is closed with these flags
            db.execute(
                "UPDATE cards SET closed = 1, final_flags = ? WHERE id = ?",
                (flags, card_id),
            )
            # Award points to matching human voters
            for v in votes:
                if _normalize_flags(v["flags"]) == flags:
                    db.execute("""
                        INSERT INTO user_points (user_token, points) VALUES (?, 1)
                        ON CONFLICT(user_token) DO UPDATE SET points = points + 1
                    """, (v["user_token"],))
            db.commit()
            return


@app.route("/")
def index():
    return render_template("crowdsource.html", flags=ALL_FLAGS)


@app.route("/api/next_card", methods=["POST"])
def next_card():
    """Get a random open card that this user hasn't voted on yet."""
    data = request.get_json(force=True)
    user_token = data.get("user_token", "")
    if not user_token:
        return jsonify({"error": "missing user_token"}), 400

    db = get_db()
    card = db.execute("""
        SELECT c.id, c.name, c.text, c.is_reaction
        FROM cards c
        WHERE c.closed = 0
          AND c.id NOT IN (SELECT card_id FROM votes WHERE user_token = ?)
        ORDER BY RANDOM()
        LIMIT 1
    """, (user_token,)).fetchone()

    if not card:
        return jsonify({"done": True, "message": "Nincs több kártya, köszönjük!"})

    return jsonify({
        "done": False,
        "card": {
            "id": card["id"],
            "name": card["name"],
            "text": card["text"],
            "is_reaction": bool(card["is_reaction"]),
            "image_url": f"https://lapkereso.hkk.hu/HKKCardImage.php?cardID={card['id']}",
        },
    })


@app.route("/api/vote", methods=["POST"])
def vote():
    """Submit a vote for a card."""
    data = request.get_json(force=True)
    user_token = data.get("user_token", "")
    card_id = data.get("card_id")
    flags = data.get("flags", [])

    if not user_token or card_id is None:
        return jsonify({"error": "missing fields"}), 400

    # Validate flags
    flags = [f for f in flags if f in ALL_FLAGS]
    flags_str = "|".join(sorted(flags))

    db = get_db()

    # Check card exists and is open
    card = db.execute("SELECT * FROM cards WHERE id = ? AND closed = 0", (card_id,)).fetchone()
    if not card:
        return jsonify({"error": "card not found or already closed"}), 404

    # Insert vote (unique constraint prevents duplicates)
    try:
        db.execute(
            "INSERT INTO votes (card_id, user_token, flags) VALUES (?, ?, ?)",
            (card_id, user_token, flags_str),
        )
        db.commit()
    except sqlite3.IntegrityError:
        return jsonify({"error": "already voted on this card"}), 409

    # Try to close the card
    _try_close_card(db, card_id)

    return jsonify({"ok": True})


@app.route("/api/leaderboard")
def leaderboard():
    """Top 10 users by points."""
    db = get_db()
    rows = db.execute(
        "SELECT user_token, points FROM user_points ORDER BY points DESC LIMIT 10"
    ).fetchall()
    return jsonify([{"user_token": r["user_token"][:8] + "...", "points": r["points"]} for r in rows])


@app.route("/api/stats")
def stats():
    """Overall progress stats."""
    db = get_db()
    total = db.execute("SELECT COUNT(*) FROM cards").fetchone()[0]
    closed = db.execute("SELECT COUNT(*) FROM cards WHERE closed = 1").fetchone()[0]
    total_votes = db.execute("SELECT COUNT(*) FROM votes").fetchone()[0]
    return jsonify({"total_cards": total, "closed_cards": closed, "total_votes": total_votes})


if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=5000)
