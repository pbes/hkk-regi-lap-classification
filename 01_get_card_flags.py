import csv
import os
import time
import mysql.connector
import requests
from dataclasses import dataclass
from urllib.parse import quote_plus
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 3306)),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", "secret"),
    "database": os.getenv("DB_NAME", "hkk-lapkereso-db"),
}

QUERY = "SELECT id, name, text, type FROM cards c"

API_URL_TEMPLATE = (
    "https://webbolt.hkk.hu/api/api.php"
    "?lapkereso/kereses"
    "&nev={name}"
    "&cardFlags=0"
    "&teszteloiLapok=0"
    "&behoCK=66fd6c346fab3b370c4f95d9f1c03776674a11c5"
)

FLAG_BITS: dict[int, str] = {
    2048: "token",
    1024: "visszavétel",
    512: "semmizés",
    256: "lopás",
    128: "dobatás",
    64: "gyógyulás",
    32: "sebzés",
    16: "erőforrás",
    8: "keresés",
    4: "húzás",
    2: "reakció",
    1: "leszedés",
}

CSV_OUTPUT = "card_flags.csv"


@dataclass
class CardDTO:
    id: int
    name: str
    text: str
    is_reaction: bool
    flags: list[str] | None = None


def normalize_type(type_str: str) -> str:
    """Normalize compound tag written with a space into a single word."""
    return type_str.replace("reakció drágítás", "reakciódrágítás")


def is_reaction_card(type_str: str) -> bool:
    """Return True if the card has the exact 'reakció' tag (not 'reakciódrágítás')."""
    normalized = normalize_type(type_str)
    tags = [t for t in normalized.split(";") if t]
    return "reakció" in tags


def decode_flags(flags: int) -> list[str]:
    """Decompose a flags bitmask into flag names, largest bit first."""
    result = []
    for bit in sorted(FLAG_BITS.keys(), reverse=True):
        if flags >= bit:
            result.append(FLAG_BITS[bit])
            flags -= bit
        if flags == 0:
            break
    return result


def fetch_card_flags(card: "CardDTO") -> list[str] | None:
    url = API_URL_TEMPLATE.format(name=quote_plus(card.name))
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            for item in data.get("cards", []):
                if item.get("ID") == card.id:
                    flags_val = item.get("flags")
                    if flags_val:
                        return decode_flags(int(flags_val))
            return None
        except Exception as exc:
            last_exc = exc
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to fetch flags for card {card.id} after 3 attempts") from last_exc


def fetch_cards() -> list[CardDTO]:
    conn = mysql.connector.connect(**DB_CONFIG)
    try:
        cursor = conn.cursor()
        cursor.execute(QUERY)
        cards = []
        for row in cursor.fetchall():
            card_id, name, text, card_type = row
            cards.append(
                CardDTO(
                    id=card_id,
                    name=name,
                    text=text,
                    is_reaction=is_reaction_card(card_type or ""),
                )
            )
        return cards
    finally:
        conn.close()


if __name__ == "__main__":
    # Load already-processed IDs to resume
    done_ids: set[int] = set()
    csv_exists = False
    try:
        with open(CSV_OUTPUT, newline="", encoding="utf-8") as existing:
            reader = csv.DictReader(existing)
            for row in reader:
                done_ids.add(int(row["id"]))
        csv_exists = bool(done_ids)
    except FileNotFoundError:
        pass

    all_cards = fetch_cards()
    remaining = [c for c in all_cards if c.id not in done_ids]
    print(f"{len(done_ids)} already done, {len(remaining)} remaining out of {len(all_cards)} total")

    with open(CSV_OUTPUT, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not csv_exists:
            writer.writerow(["id", "name", "is_reaction", "text", "flags"])

        for i, card in enumerate(remaining):
            card.flags = fetch_card_flags(card)
            print(f"[{len(done_ids) + i + 1}/{len(all_cards)}] {card.id} {card.name!r} -> {card.flags}")
            writer.writerow([
                card.id,
                card.name,
                card.is_reaction,
                card.text.replace("\n", " ").replace("\r", ""),
                "|".join(card.flags) if card.flags else "",
            ])
            f.flush()
            if i < len(remaining) - 1:
                time.sleep(0.2)

    print(f"\nWritten to {CSV_OUTPUT}")
