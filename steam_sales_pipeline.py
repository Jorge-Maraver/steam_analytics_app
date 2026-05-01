"""
steam_sales_pipeline.py

Pipeline completa para:
1) Extraer datos de Steam Store API, Steam Reviews API, Steam Web API y SteamSpy.
2) Construir dos datasets:
   - dataset_full_post_release.csv: máxima información posible, incluyendo variables post-lanzamiento.
   - dataset_pre_release.csv: solo variables que tienen sentido antes de salir a la venta.
3) Opcionalmente entrenar modelos baseline para estimar owners/ventas usando SteamSpy como target.

IMPORTANTE:
- Añade tu clave en STEAM_API_KEY o como variable de entorno.
- SteamSpy se asume correcto, como pediste.
- Para entrenar un modelo útil necesitas muchos juegos, no solo 4.
- Tamagotchi Plaza no parece tener appid de Steam; queda marcado como no disponible.

Instalación:
    pip install requests pandas numpy scikit-learn tqdm python-dateutil

Uso rápido para tus 4 juegos:
    python steam_sales_pipeline.py --mode target --steam-key TU_CLAVE

Uso para crear dataset histórico de entrenamiento:
    python steam_sales_pipeline.py --mode sample --max-games 1000 --steam-key TU_CLAVE

Uso para entrenar baseline tras extraer datos:
    python steam_sales_pipeline.py --mode sample --max-games 1000 --steam-key TU_CLAVE --train-models
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
import random
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from dateutil import parser as date_parser
from tqdm import tqdm

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler


# ============================================================
# CONFIG
# ============================================================

DEFAULT_TARGET_GAMES = {
    "Monster Hunter Wilds": 2246340,
    "Slay the Spire 2": 2868840,
    "Skull and Bones": 2853730,
    "Tamagotchi Plaza": None,
}

OUTPUT_DIR = Path("steam_sales_pipeline_output")
RAW_DIR = OUTPUT_DIR / "raw"
DATASET_DIR = OUTPUT_DIR / "datasets"
MODEL_DIR = OUTPUT_DIR / "models"

for d in [OUTPUT_DIR, RAW_DIR, DATASET_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# HTTP CLIENT
# ============================================================

@dataclass
class ApiClient:
    steam_key: Optional[str] = None
    sleep_seconds: float = 1.25
    timeout: int = 30
    max_retries: int = 4

    def __post_init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; SteamSalesPipeline/1.0; +local-script)"
        })

    def get_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        last_error = None

        for attempt in range(self.max_retries):
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)

                if r.status_code == 429:
                    wait = (attempt + 1) * 10
                    print(f"[RATE LIMIT] 429. Esperando {wait}s...")
                    time.sleep(wait)
                    continue

                r.raise_for_status()

                if not r.text.strip():
                    return {}

                return r.json()

            except requests.HTTPError as e:
                last_error = e
                status_code = e.response.status_code if e.response is not None else None
                if status_code in {400, 401, 403, 404}:
                    raise

                wait = (attempt + 1) * 3
                print(f"[WARN] Error GET {url}: {e}. Reintento en {wait}s...")
                time.sleep(wait)

            except Exception as e:
                last_error = e
                wait = (attempt + 1) * 3
                print(f"[WARN] Error GET {url}: {e}. Reintento en {wait}s...")
                time.sleep(wait)

        raise RuntimeError(f"No se pudo obtener JSON tras {self.max_retries} intentos: {url} | {last_error}")

    def polite_sleep(self):
        time.sleep(self.sleep_seconds + random.random() * 0.35)


# ============================================================
# API FUNCTIONS
# ============================================================

def get_steamspy_app_list(client: ApiClient, max_pages: int = 4) -> List[Dict[str, Any]]:
    """
    Fallback para construir muestras cuando ISteamApps/GetAppList no responde.
    SteamSpy pagina request=all en bloques de 1000 apps.
    """
    apps: List[Dict[str, Any]] = []

    for page in range(max_pages):
        url = "https://steamspy.com/api.php"
        data = client.get_json(url, params={"request": "all", "page": page})
        if not isinstance(data, dict) or not data:
            break

        for appid, item in data.items():
            if not isinstance(item, dict):
                continue
            apps.append({
                "appid": safe_int(item.get("appid") or appid),
                "name": item.get("name"),
            })

        client.polite_sleep()

    return [a for a in apps if a.get("appid")]


def get_steam_app_list(client: ApiClient, steamspy_pages: int = 4) -> List[Dict[str, Any]]:
    cache_path = RAW_DIR / "steam_app_list_cache.json"
    urls = [
        "https://api.steampowered.com/ISteamApps/GetAppList/v2/",
        "https://api.steampowered.com/ISteamApps/GetAppList/v0002/",
    ]

    for url in urls:
        try:
            data = client.get_json(url, params={"format": "json"})
            apps = data.get("applist", {}).get("apps", [])
            if apps:
                save_json(cache_path, apps)
                return apps
        except Exception as e:
            print(f"[WARN] No se pudo usar Steam GetAppList ({url}): {e}")

    if cache_path.exists():
        print(f"[INFO] Usando cache local de apps: {cache_path}")
        return load_json(cache_path)

    print("[INFO] Usando SteamSpy como fallback para listar apps.")
    apps = get_steamspy_app_list(client, max_pages=steamspy_pages)
    if apps:
        save_json(cache_path, apps)
    return apps


def get_store_appdetails(
    client: ApiClient,
    appid: int,
    cc: str = "us",
    lang: str = "english",
) -> Dict[str, Any]:
    url = "https://store.steampowered.com/api/appdetails"
    params = {"appids": appid, "cc": cc, "l": lang}
    data = client.get_json(url, params=params)
    block = data.get(str(appid), {})
    if not block.get("success"):
        return {"success": False, "data": None}
    return {"success": True, "data": block.get("data", {})}


def get_reviews(
    client: ApiClient,
    appid: int,
    num_per_page: int = 100,
    max_pages: int = 3,
    language: str = "all",
) -> Dict[str, Any]:
    """
    Descarga resumen + varias páginas de reviews.
    Para el dataset se usa sobre todo query_summary y agregados.
    """
    url = f"https://store.steampowered.com/appreviews/{appid}"

    all_reviews = []
    query_summary = {}
    cursor = "*"

    for page in range(max_pages):
        params = {
            "json": 1,
            "language": language,
            "purchase_type": "all",
            "filter": "recent",
            "review_type": "all",
            "num_per_page": num_per_page,
            "cursor": cursor,
        }
        data = client.get_json(url, params=params)

        if page == 0:
            query_summary = data.get("query_summary", {}) or {}

        reviews = data.get("reviews", []) or []
        all_reviews.extend(reviews)

        new_cursor = data.get("cursor")
        if not reviews or not new_cursor or new_cursor == cursor:
            break

        cursor = new_cursor
        client.polite_sleep()

    return {
        "query_summary": query_summary,
        "reviews": all_reviews,
    }


def get_news(client: ApiClient, appid: int, count: int = 20, max_length: int = 500) -> Dict[str, Any]:
    url = "https://api.steampowered.com/ISteamNews/GetNewsForApp/v2/"
    params = {
        "appid": appid,
        "count": count,
        "maxlength": max_length,
        "format": "json",
    }
    return client.get_json(url, params=params)


def get_current_players(client: ApiClient, appid: int) -> Dict[str, Any]:
    url = "https://api.steampowered.com/ISteamUserStats/GetNumberOfCurrentPlayers/v1/"
    params = {"appid": appid, "format": "json"}
    if client.steam_key:
        params["key"] = client.steam_key
    return client.get_json(url, params=params)


def get_global_achievements(client: ApiClient, appid: int) -> Dict[str, Any]:
    url = "https://api.steampowered.com/ISteamUserStats/GetGlobalAchievementPercentagesForApp/v2/"
    params = {"gameid": appid, "format": "json"}
    if client.steam_key:
        params["key"] = client.steam_key
    return client.get_json(url, params=params)


def get_steamspy_appdetails(client: ApiClient, appid: int) -> Dict[str, Any]:
    url = "https://steamspy.com/api.php"
    params = {"request": "appdetails", "appid": appid}
    return client.get_json(url, params=params)


# ============================================================
# UTILS
# ============================================================

def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_int(x: Any) -> Optional[int]:
    try:
        if x is None or x == "":
            return None
        return int(float(str(x).replace(",", "")))
    except Exception:
        return None


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(str(x).replace(",", ""))
    except Exception:
        return None


def parse_owners_midpoint(owners: Any) -> Optional[float]:
    """
    SteamSpy suele devolver owners como texto tipo:
        "200,000 .. 500,000"
    Devolvemos el punto medio.
    """
    if not owners or not isinstance(owners, str):
        return None

    nums = re.findall(r"[\d,]+", owners)
    nums = [safe_int(n) for n in nums]
    nums = [n for n in nums if n is not None]

    if len(nums) >= 2:
        return float((nums[0] + nums[1]) / 2)
    if len(nums) == 1:
        return float(nums[0])
    return None


def parse_release_date(store_data: Dict[str, Any]) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[int], Optional[bool]]:
    rd = store_data.get("release_date") or {}
    date_text = rd.get("date")
    coming_soon = rd.get("coming_soon")

    if not date_text:
        return None, None, None, None, coming_soon

    # Steam puede devolver formatos ambiguos: "Coming soon", "2025", "Q2 2026", etc.
    try:
        dt = date_parser.parse(date_text, fuzzy=True, default=datetime(1900, 1, 1))
        if dt.year == 1900:
            return date_text, None, None, None, coming_soon
        return date_text, dt.year, dt.month, ((dt.month - 1) // 3) + 1, coming_soon
    except Exception:
        year_match = re.search(r"(20\d{2}|19\d{2})", date_text)
        year = int(year_match.group(1)) if year_match else None
        return date_text, year, None, None, coming_soon


def clean_html(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def list_names(items: Any) -> List[str]:
    if not isinstance(items, list):
        return []
    out = []
    for item in items:
        if isinstance(item, dict) and item.get("description"):
            out.append(str(item["description"]))
        elif isinstance(item, str):
            out.append(item)
    return out


def dict_keys_true(d: Any) -> List[str]:
    if not isinstance(d, dict):
        return []
    return [k for k, v in d.items() if bool(v)]


def price_to_float(price_overview: Any, field: str) -> Optional[float]:
    """
    Steam suele devolver precios en céntimos.
    """
    if not isinstance(price_overview, dict):
        return None
    value = price_overview.get(field)
    if value is None:
        return None
    try:
        return float(value) / 100.0
    except Exception:
        return None


def count_supported_languages(supported_languages: Any) -> int:
    txt = clean_html(supported_languages)
    if not txt:
        return 0
    # Suele venir separado por comas y con notas tipo "*"
    parts = [p.strip() for p in re.split(r",|;", txt) if p.strip()]
    return len(parts)


def has_language(supported_languages: Any, lang: str) -> int:
    txt = clean_html(supported_languages).lower()
    return int(lang.lower() in txt)


def review_aggregates(reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not reviews:
        return {
            "review_sample_count": 0,
            "review_sample_positive_ratio": None,
            "review_sample_avg_playtime_forever": None,
            "review_sample_avg_playtime_at_review": None,
            "review_sample_avg_votes_up": None,
            "review_sample_avg_weighted_score": None,
            "review_sample_received_for_free_ratio": None,
        }

    voted = [1 if r.get("voted_up") else 0 for r in reviews if r.get("voted_up") is not None]

    play_forever = []
    play_at_review = []
    votes_up = []
    weighted = []
    free = []

    for r in reviews:
        author = r.get("author") or {}
        if author.get("playtime_forever") is not None:
            play_forever.append(author.get("playtime_forever"))
        if author.get("playtime_at_review") is not None:
            play_at_review.append(author.get("playtime_at_review"))
        if r.get("votes_up") is not None:
            votes_up.append(r.get("votes_up"))
        if r.get("weighted_vote_score") is not None:
            weighted.append(safe_float(r.get("weighted_vote_score")))
        if r.get("received_for_free") is not None:
            free.append(1 if r.get("received_for_free") else 0)

    return {
        "review_sample_count": len(reviews),
        "review_sample_positive_ratio": float(np.mean(voted)) if voted else None,
        "review_sample_avg_playtime_forever": float(np.mean(play_forever)) if play_forever else None,
        "review_sample_avg_playtime_at_review": float(np.mean(play_at_review)) if play_at_review else None,
        "review_sample_avg_votes_up": float(np.mean(votes_up)) if votes_up else None,
        "review_sample_avg_weighted_score": float(np.mean(weighted)) if weighted else None,
        "review_sample_received_for_free_ratio": float(np.mean(free)) if free else None,
    }


def achievements_aggregates(ach_data: Dict[str, Any]) -> Dict[str, Any]:
    achs = (
        ach_data.get("achievementpercentages", {})
        .get("achievements", [])
        if isinstance(ach_data, dict)
        else []
    )
    if not achs:
        return {
            "achievements_count": 0,
            "achievement_avg_percent": None,
            "achievement_median_percent": None,
            "achievement_min_percent": None,
            "achievement_max_percent": None,
        }

    vals = [safe_float(a.get("percent")) for a in achs]
    vals = [v for v in vals if v is not None]

    return {
        "achievements_count": len(achs),
        "achievement_avg_percent": float(np.mean(vals)) if vals else None,
        "achievement_median_percent": float(np.median(vals)) if vals else None,
        "achievement_min_percent": float(np.min(vals)) if vals else None,
        "achievement_max_percent": float(np.max(vals)) if vals else None,
    }


def news_aggregates(news_data: Dict[str, Any], release_year: Optional[int]) -> Dict[str, Any]:
    items = []
    if isinstance(news_data, dict):
        items = news_data.get("appnews", {}).get("newsitems", []) or []

    dates = [safe_int(n.get("date")) for n in items]
    dates = [d for d in dates if d is not None]

    now_ts = int(datetime.now(timezone.utc).timestamp())
    days_since_last = None
    if dates:
        days_since_last = (now_ts - max(dates)) / 86400.0

    prelaunch_count = None
    if release_year:
        # Aproximación: si tenemos fecha solo por año, cuenta noticias anteriores a 31 dic de ese año.
        cutoff = int(datetime(release_year, 12, 31, tzinfo=timezone.utc).timestamp())
        prelaunch_count = sum(1 for d in dates if d <= cutoff)

    return {
        "news_count": len(items),
        "days_since_last_news": days_since_last,
        "news_prelaunch_count_approx": prelaunch_count,
    }


def price_category(price: Optional[float]) -> str:
    if price is None:
        return "unknown"
    if price <= 0:
        return "free"
    if price < 10:
        return "low"
    if price < 30:
        return "mid"
    if price < 60:
        return "high"
    return "premium"


def big_publisher_heuristic(publisher: str) -> int:
    big = {
        "capcom", "ubisoft", "electronic arts", "ea", "activision", "blizzard",
        "bandai namco", "sony", "playstation", "xbox", "microsoft", "bethesda",
        "take-two", "2k", "rockstar", "sega", "square enix", "warner", "konami",
        "nintendo", "devolver", "paradox", "focus entertainment", "team17"
    }
    p = publisher.lower()
    return int(any(b in p for b in big))


# ============================================================
# EXTRACTION
# ============================================================

def extract_one_game(
    client: ApiClient,
    appid: Optional[int],
    query_name: Optional[str] = None,
    reviews_pages: int = 3,
    force: bool = False,
) -> Dict[str, Any]:
    if appid is None:
        return {
            "query_name": query_name,
            "appid": None,
            "steam_available": False,
            "reason": "No Steam appid provided.",
        }

    raw_path = RAW_DIR / f"{appid}.json"
    if raw_path.exists() and not force:
        return load_json(raw_path)

    bundle = {
        "query_name": query_name,
        "appid": appid,
        "steam_available": True,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "errors": {},
    }

    try:
        bundle["store"] = get_store_appdetails(client, appid)
    except Exception as e:
        bundle["errors"]["store"] = str(e)

    client.polite_sleep()

    try:
        bundle["reviews"] = get_reviews(client, appid, max_pages=reviews_pages)
    except Exception as e:
        bundle["errors"]["reviews"] = str(e)

    client.polite_sleep()

    try:
        bundle["news"] = get_news(client, appid)
    except Exception as e:
        bundle["errors"]["news"] = str(e)

    client.polite_sleep()

    try:
        bundle["current_players"] = get_current_players(client, appid)
    except Exception as e:
        bundle["errors"]["current_players"] = str(e)

    client.polite_sleep()

    try:
        bundle["global_achievements"] = get_global_achievements(client, appid)
    except Exception as e:
        bundle["errors"]["global_achievements"] = str(e)

    client.polite_sleep()

    try:
        bundle["steamspy"] = get_steamspy_appdetails(client, appid)
    except Exception as e:
        bundle["errors"]["steamspy"] = str(e)

    save_json(raw_path, bundle)
    return bundle


def extract_target_games(client: ApiClient, force: bool = False) -> List[Dict[str, Any]]:
    out = []
    for name, appid in tqdm(DEFAULT_TARGET_GAMES.items(), desc="Target games"):
        out.append(extract_one_game(client, appid, query_name=name, force=force))
    save_json(RAW_DIR / "target_games_bundle.json", out)
    return out


def extract_sample_games(
    client: ApiClient,
    max_games: int = 1000,
    reviews_pages: int = 1,
    force: bool = False,
    min_appid: Optional[int] = None,
    max_appid: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Construye un dataset histórico amplio.
    Estrategia:
    - Descarga lista de apps.
    - Baraja apps.
    - Prueba appdetails.
    - Conserva solo type='game' con SteamSpy owners parseable.
    """
    steamspy_pages = max(1, min(100, math.ceil((max_games * 4) / 1000)))
    apps = get_steam_app_list(client, steamspy_pages=steamspy_pages)

    if min_appid is not None:
        apps = [a for a in apps if safe_int(a.get("appid")) and int(a["appid"]) >= min_appid]
    if max_appid is not None:
        apps = [a for a in apps if safe_int(a.get("appid")) and int(a["appid"]) <= max_appid]

    random.shuffle(apps)

    bundles = []
    seen = set()

    pbar = tqdm(apps, desc="Sample games")
    for app in pbar:
        if len(bundles) >= max_games:
            break

        appid = safe_int(app.get("appid"))
        if not appid or appid in seen:
            continue
        seen.add(appid)

        try:
            bundle = extract_one_game(
                client,
                appid,
                query_name=app.get("name"),
                reviews_pages=reviews_pages,
                force=force,
            )

            store_data = (bundle.get("store") or {}).get("data") or {}
            spy = bundle.get("steamspy") or {}

            if store_data.get("type") != "game":
                continue

            owners_mid = parse_owners_midpoint(spy.get("owners"))
            if owners_mid is None or owners_mid <= 0:
                continue

            bundles.append(bundle)
            pbar.set_postfix({"kept": len(bundles), "appid": appid})

        except Exception as e:
            print(f"[WARN] App {appid} ignorada: {e}")
            continue

    save_json(RAW_DIR / "sample_games_bundle.json", bundles)
    return bundles


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def bundle_to_row(bundle: Dict[str, Any]) -> Dict[str, Any]:
    appid = bundle.get("appid")
    store_data = ((bundle.get("store") or {}).get("data") or {}) if bundle.get("store") else {}
    reviews_data = bundle.get("reviews") or {}
    query_summary = reviews_data.get("query_summary") or {}
    review_list = reviews_data.get("reviews") or []
    news_data = bundle.get("news") or {}
    current_players = bundle.get("current_players") or {}
    achievements_data = bundle.get("global_achievements") or {}
    spy = bundle.get("steamspy") or {}

    price_overview = store_data.get("price_overview") or {}

    release_date_text, release_year, release_month, release_quarter, coming_soon = parse_release_date(store_data)

    developers = store_data.get("developers") or []
    publishers = store_data.get("publishers") or []
    genres = list_names(store_data.get("genres"))
    categories = list_names(store_data.get("categories"))
    platforms = store_data.get("platforms") or {}
    tags = spy.get("tags") if isinstance(spy.get("tags"), dict) else {}

    final_price = price_to_float(price_overview, "final")
    initial_price = price_to_float(price_overview, "initial")
    discount_percent = safe_float(price_overview.get("discount_percent"))

    description = clean_html(store_data.get("short_description"))
    detailed_description = clean_html(store_data.get("detailed_description"))
    about_the_game = clean_html(store_data.get("about_the_game"))
    full_text = " ".join([description, detailed_description, about_the_game]).strip()

    owners_raw = spy.get("owners")
    owners_mid = parse_owners_midpoint(owners_raw)

    recs_total = None
    if isinstance(store_data.get("recommendations"), dict):
        recs_total = safe_int(store_data["recommendations"].get("total"))

    achievements_store_total = None
    if isinstance(store_data.get("achievements"), dict):
        achievements_store_total = safe_int(store_data["achievements"].get("total"))

    news_aggs = news_aggregates(news_data, release_year)
    review_aggs = review_aggregates(review_list)
    ach_aggs = achievements_aggregates(achievements_data)

    row = {
        # ids
        "appid": appid,
        "query_name": bundle.get("query_name"),
        "name": store_data.get("name") or spy.get("name"),
        "steam_available": bundle.get("steam_available", True),

        # target
        "owners_raw": owners_raw,
        "owners_midpoint": owners_mid,
        "log_owners_midpoint": math.log1p(owners_mid) if owners_mid is not None else None,

        # store metadata
        "type": store_data.get("type"),
        "required_age": safe_int(store_data.get("required_age")),
        "is_free": int(bool(store_data.get("is_free"))) if store_data.get("is_free") is not None else None,
        "release_date_text": release_date_text,
        "release_year": release_year,
        "release_month": release_month,
        "release_quarter": release_quarter,
        "coming_soon": int(bool(coming_soon)) if coming_soon is not None else None,

        # company
        "developers": developers,
        "publishers": publishers,
        "developer_main": developers[0] if developers else None,
        "publisher_main": publishers[0] if publishers else None,
        "developers_count": len(developers),
        "publishers_count": len(publishers),
        "publisher_is_big_heuristic": big_publisher_heuristic(" ".join(publishers)) if publishers else 0,

        # price
        "currency": price_overview.get("currency"),
        "price_final": final_price,
        "price_initial": initial_price,
        "discount_percent": discount_percent,
        "is_discounted": int((discount_percent or 0) > 0),
        "price_category": price_category(final_price),

        # market / accessibility
        "supported_languages_raw": clean_html(store_data.get("supported_languages")),
        "supported_languages_count": count_supported_languages(store_data.get("supported_languages")),
        "has_english": has_language(store_data.get("supported_languages"), "English"),
        "has_spanish": has_language(store_data.get("supported_languages"), "Spanish"),
        "has_simplified_chinese": has_language(store_data.get("supported_languages"), "Simplified Chinese"),
        "has_traditional_chinese": has_language(store_data.get("supported_languages"), "Traditional Chinese"),

        # platforms
        "platform_windows": int(bool(platforms.get("windows"))),
        "platform_mac": int(bool(platforms.get("mac"))),
        "platform_linux": int(bool(platforms.get("linux"))),
        "platforms_count": sum(int(bool(platforms.get(k))) for k in ["windows", "mac", "linux"]),

        # lists
        "genres": genres,
        "categories": categories,
        "steamspy_tags": list(tags.keys()),
        "steamspy_tags_weighted": tags,

        # visual / media
        "screenshots_count": len(store_data.get("screenshots") or []),
        "movies_count": len(store_data.get("movies") or []),
        "has_trailer": int(len(store_data.get("movies") or []) > 0),
        "dlc_count": len(store_data.get("dlc") or []),
        "has_dlc": int(len(store_data.get("dlc") or []) > 0),

        # text
        "short_description": description,
        "full_text": full_text,
        "description_length": len(description),
        "full_text_length": len(full_text),
        "full_text_word_count": len(full_text.split()) if full_text else 0,

        # post-release review / popularity
        "steam_recommendations_total": recs_total,
        "review_total": safe_int(query_summary.get("total_reviews")),
        "review_total_positive": safe_int(query_summary.get("total_positive")),
        "review_total_negative": safe_int(query_summary.get("total_negative")),
        "review_score": safe_float(query_summary.get("review_score")),
        "review_score_desc": query_summary.get("review_score_desc"),
        "review_positive_ratio": (
            safe_int(query_summary.get("total_positive")) /
            max(1, safe_int(query_summary.get("total_reviews")) or 0)
            if safe_int(query_summary.get("total_reviews")) else None
        ),

        # review sample aggs
        **review_aggs,

        # current players
        "current_players": safe_int((current_players.get("response") or {}).get("player_count")),

        # SteamSpy post-release
        "steamspy_positive": safe_int(spy.get("positive")),
        "steamspy_negative": safe_int(spy.get("negative")),
        "steamspy_userscore": safe_float(spy.get("userscore")),
        "steamspy_score_rank": safe_float(spy.get("score_rank")),
        "steamspy_average_forever": safe_int(spy.get("average_forever")),
        "steamspy_average_2weeks": safe_int(spy.get("average_2weeks")),
        "steamspy_median_forever": safe_int(spy.get("median_forever")),
        "steamspy_median_2weeks": safe_int(spy.get("median_2weeks")),
        "steamspy_ccu": safe_int(spy.get("ccu")),
        "steamspy_price": safe_int(spy.get("price")),
        "steamspy_initialprice": safe_int(spy.get("initialprice")),
        "steamspy_discount": safe_int(spy.get("discount")),

        # achievements
        "achievements_store_total": achievements_store_total,
        **ach_aggs,

        # news
        **news_aggs,

        # raw errors
        "errors": bundle.get("errors", {}),
    }

    return row


def add_developer_publisher_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea features históricas por developer/publisher usando SOLO juegos anteriores por release_year.
    Esto reduce leakage para el dataset pre-release.
    Si release_year falta, usa agregados generales con cuidado.
    """
    df = df.copy()

    for col in ["developer_main", "publisher_main"]:
        prefix = "developer" if col == "developer_main" else "publisher"

        df[f"{prefix}_past_games_count"] = 0
        df[f"{prefix}_past_avg_log_owners"] = np.nan
        df[f"{prefix}_past_median_log_owners"] = np.nan
        df[f"{prefix}_past_max_log_owners"] = np.nan
        df[f"{prefix}_has_past_hit_1m"] = 0

        for idx, row in df.iterrows():
            entity = row.get(col)
            year = row.get("release_year")

            if not entity or pd.isna(entity):
                continue

            hist = df[df[col] == entity]

            if not pd.isna(year):
                hist = hist[hist["release_year"].fillna(9999) < year]
            else:
                hist = hist[hist.index != idx]

            hist = hist[hist["log_owners_midpoint"].notna()]

            if len(hist) == 0:
                continue

            df.at[idx, f"{prefix}_past_games_count"] = len(hist)
            df.at[idx, f"{prefix}_past_avg_log_owners"] = hist["log_owners_midpoint"].mean()
            df.at[idx, f"{prefix}_past_median_log_owners"] = hist["log_owners_midpoint"].median()
            df.at[idx, f"{prefix}_past_max_log_owners"] = hist["log_owners_midpoint"].max()
            df.at[idx, f"{prefix}_has_past_hit_1m"] = int((hist["owners_midpoint"] >= 1_000_000).any())

    return df


def add_competition_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "release_year" not in df or "release_month" not in df:
        df["competition_same_month_count"] = np.nan
        return df

    month_counts = (
        df.groupby(["release_year", "release_month"], dropna=True)
        .size()
        .rename("competition_same_month_count")
        .reset_index()
    )

    df = df.merge(month_counts, on=["release_year", "release_month"], how="left")
    df["competition_same_month_count"] = df["competition_same_month_count"].fillna(0)
    return df


def build_datasets(bundles: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = [bundle_to_row(b) for b in bundles]
    df = pd.DataFrame(rows)

    # Solo juegos con target válido
    if "owners_midpoint" in df.columns:
        df = df[df["owners_midpoint"].notna()].copy()

    df = add_developer_publisher_history_features(df)
    df = add_competition_features(df)

    # Dataset completo post-release: mantiene casi todo.
    full_df = df.copy()

    # Dataset pre-release: elimina variables que normalmente NO existen antes de salir.
    pre_release_cols = [
        # ids / target
        "appid", "name", "owners_raw", "owners_midpoint", "log_owners_midpoint",

        # metadata prelaunch
        "type", "required_age", "is_free", "release_date_text", "release_year",
        "release_month", "release_quarter", "coming_soon",

        # empresa
        "developers", "publishers", "developer_main", "publisher_main",
        "developers_count", "publishers_count", "publisher_is_big_heuristic",
        "developer_past_games_count", "developer_past_avg_log_owners",
        "developer_past_median_log_owners", "developer_past_max_log_owners",
        "developer_has_past_hit_1m",
        "publisher_past_games_count", "publisher_past_avg_log_owners",
        "publisher_past_median_log_owners", "publisher_past_max_log_owners",
        "publisher_has_past_hit_1m",

        # precio y mercado
        "currency", "price_final", "price_initial", "discount_percent",
        "is_discounted", "price_category",
        "supported_languages_count", "has_english", "has_spanish",
        "has_simplified_chinese", "has_traditional_chinese",
        "platform_windows", "platform_mac", "platform_linux", "platforms_count",

        # contenido ficha
        "genres", "categories", "steamspy_tags",
        "screenshots_count", "movies_count", "has_trailer",
        "description_length", "full_text_length", "full_text_word_count",
        "short_description", "full_text",

        # mercado temporal
        "competition_same_month_count",

        # aproximación de marketing pre-release si tienes noticias antes/durante año de lanzamiento
        "news_prelaunch_count_approx",
    ]

    pre_release_cols = [c for c in pre_release_cols if c in full_df.columns]
    pre_df = full_df[pre_release_cols].copy()

    return full_df, pre_df


# ============================================================
# CSV SAFE EXPORT
# ============================================================

def stringify_complex_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].apply(lambda x: isinstance(x, (list, dict))).any():
            out[col] = out[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else x)
    return out


def export_datasets(full_df: pd.DataFrame, pre_df: pd.DataFrame) -> None:
    full_path = DATASET_DIR / "dataset_full_post_release.csv"
    pre_path = DATASET_DIR / "dataset_pre_release.csv"

    stringify_complex_columns(full_df).to_csv(full_path, index=False, encoding="utf-8")
    stringify_complex_columns(pre_df).to_csv(pre_path, index=False, encoding="utf-8")

    full_df.to_json(DATASET_DIR / "dataset_full_post_release.json", orient="records", force_ascii=False, indent=2)
    pre_df.to_json(DATASET_DIR / "dataset_pre_release.json", orient="records", force_ascii=False, indent=2)

    print(f"\nDatasets guardados:")
    print(f"- {full_path}")
    print(f"- {pre_path}")


# ============================================================
# MODELING BASELINE
# ============================================================

def parse_json_list_cell(x: Any) -> List[str]:
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            y = json.loads(x)
            if isinstance(y, list):
                return [str(v) for v in y]
        except Exception:
            pass
    return []


def expand_multilabel(df: pd.DataFrame, col: str, top_k: int = 50) -> pd.DataFrame:
    """
    Expande columnas tipo genres/categories/tags a one-hot, quedándose con top_k valores.
    """
    lists = df[col].apply(parse_json_list_cell if df[col].dtype == "object" else lambda x: x if isinstance(x, list) else [])
    counts = {}
    for vals in lists:
        for v in vals:
            counts[v] = counts.get(v, 0) + 1

    top = set([k for k, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_k]])

    expanded = pd.DataFrame(index=df.index)
    for v in sorted(top):
        safe_name = re.sub(r"[^a-zA-Z0-9_]+", "_", v.lower()).strip("_")
        expanded[f"{col}_{safe_name}"] = lists.apply(lambda xs: int(v in xs))

    return expanded


def prepare_ml_dataframe(df: pd.DataFrame, pre_release: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df = df[df["log_owners_midpoint"].notna()].copy()

    y = df["log_owners_midpoint"]

    # Excluir target y variables no modelables / identificadores
    drop_cols = {
        "owners_raw", "owners_midpoint", "log_owners_midpoint",
        "appid", "name", "query_name", "errors",
        "developers", "publishers",
        "supported_languages_raw",
        "release_date_text",
        "steamspy_tags_weighted",
    }

    # No usar texto crudo aquí salvo full_text con TF-IDF en pipeline separado.
    # Para baseline tabular simple, usamos longitudes ya calculadas.
    drop_cols.update({"short_description", "full_text"})

    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Expandir multilabels
    parts = [X.drop(columns=[c for c in ["genres", "categories", "steamspy_tags"] if c in X.columns], errors="ignore")]

    for col in ["genres", "categories", "steamspy_tags"]:
        if col in X.columns:
            parts.append(expand_multilabel(X, col, top_k=50))

    X2 = pd.concat(parts, axis=1)

    # Convertir objetos restantes a string categórica
    for c in X2.columns:
        if X2[c].dtype == "object":
            X2[c] = X2[c].fillna("unknown").astype(str)

    return X2, y


def train_baseline(df: pd.DataFrame, name: str) -> None:
    X, y = prepare_ml_dataframe(df)

    if len(X) < 50:
        print(f"[WARN] Dataset {name} tiene solo {len(X)} filas. Entrenar con tan poco no es fiable.")
        if len(X) < 10:
            return

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_cols),
        ],
        remainder="drop",
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1,
    )

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
    )

    pipe.fit(X_train, y_train)
    pred_log = pipe.predict(X_test)

    pred_owners = np.expm1(pred_log)
    real_owners = np.expm1(y_test)

    mae_log = mean_absolute_error(y_test, pred_log)
    rmse_log = mean_squared_error(y_test, pred_log, squared=False)
    r2 = r2_score(y_test, pred_log)

    mae_owners = mean_absolute_error(real_owners, pred_owners)

    print(f"\nModelo baseline: {name}")
    print(f"- Filas: {len(X)}")
    print(f"- MAE log owners: {mae_log:.4f}")
    print(f"- RMSE log owners: {rmse_log:.4f}")
    print(f"- R2 log owners: {r2:.4f}")
    print(f"- MAE owners aprox: {mae_owners:,.0f}")

    # Importancias si están disponibles
    try:
        fitted_model = pipe.named_steps["model"]
        feature_names = pipe.named_steps["prep"].get_feature_names_out()
        importances = fitted_model.feature_importances_
        fi = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False)
        fi.to_csv(MODEL_DIR / f"feature_importance_{name}.csv", index=False)
        print(f"- Importancias guardadas en {MODEL_DIR / f'feature_importance_{name}.csv'}")
    except Exception as e:
        print(f"[WARN] No se pudieron guardar importancias: {e}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["target", "sample", "from-raw"], default="target")
    parser.add_argument("--steam-key", default=os.getenv("STEAM_API_KEY"))
    parser.add_argument("--max-games", type=int, default=1000)
    parser.add_argument("--reviews-pages", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--train-models", action="store_true")
    parser.add_argument("--min-appid", type=int, default=None)
    parser.add_argument("--max-appid", type=int, default=None)
    args = parser.parse_args()

    client = ApiClient(steam_key=args.steam_key)

    if args.mode == "target":
        bundles = extract_target_games(client, force=args.force)

    elif args.mode == "sample":
        bundles = extract_sample_games(
            client,
            max_games=args.max_games,
            reviews_pages=args.reviews_pages,
            force=args.force,
            min_appid=args.min_appid,
            max_appid=args.max_appid,
        )

    elif args.mode == "from-raw":
        raw_files = sorted(RAW_DIR.glob("*.json"))
        bundles = []
        for p in raw_files:
            if p.name.endswith("_bundle.json"):
                continue
            try:
                bundles.append(load_json(p))
            except Exception:
                pass
    else:
        raise ValueError(args.mode)

    full_df, pre_df = build_datasets(bundles)
    export_datasets(full_df, pre_df)

    print("\nResumen:")
    print(f"- Full post-release shape: {full_df.shape}")
    print(f"- Pre-release shape: {pre_df.shape}")

    if len(full_df) > 0:
        print("\nColumnas full:")
        print(list(full_df.columns))

        print("\nColumnas pre-release:")
        print(list(pre_df.columns))

    if args.train_models:
        train_baseline(full_df, "full_post_release")
        train_baseline(pre_df, "pre_release")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
