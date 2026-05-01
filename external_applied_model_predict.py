from __future__ import annotations

import argparse
from difflib import SequenceMatcher
from pathlib import Path
import ast
import json
import os
import re
import time

import joblib
import numpy as np
import pandas as pd
import requests

from steam_sales_pipeline import (
    ApiClient,
    add_competition_features,
    add_developer_publisher_history_features,
    bundle_to_row,
    extract_one_game,
)


DATA_PATH_RAW_FULL = Path("steam_sales_pipeline_output/datasets/dataset_full_post_release.csv")
DATA_PATH_CLEAN = Path("steam_sales_pipeline_output/datasets/dataset_full_post_release_no_nulls.parquet")
RAW_DIR = Path("steam_sales_pipeline_output/raw")
CACHE_DIR = Path("steam_sales_pipeline_output/api_cache")
OUTPUT_DIR = Path("steam_sales_pipeline_output/external_applied_model")

STEAM_APP_LIST_CACHE = CACHE_DIR / "steam_app_list.json"
LOCAL_STEAM_APP_LIST_CACHE = RAW_DIR / "steam_app_list_cache.json"

MODEL_CANDIDATES = [
    Path("steam_sales_pipeline_output/model_results/random_forest_fine_tuning/random_forest_tuned_final.joblib"),
    Path("steam_sales_pipeline_output/model_results/best_extended_model_log_owners_midpoint_extra_trees.joblib"),
    Path("steam_sales_pipeline_output/model_results/best_model_log_owners_midpoint_extra_trees.joblib"),
]

SELECTED_CATEGORIES = [
    "Single-player",
    "Multi-player",
    "Online Multi-Player",
    "Co-op",
    "Online Co-op",
    "PvP",
    "Online PvP",
    "Steam Achievements",
    "Full controller support",
    "Steam Cloud",
    "Remote Play Together",
    "Cross-Platform Multiplayer",
]


def normalize_name(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_name(a), normalize_name(b)).ratio()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def request_json(url: str, params: dict | None = None, timeout: int = 30):
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def load_steam_app_list(force_refresh: bool = False) -> pd.DataFrame:
    if LOCAL_STEAM_APP_LIST_CACHE.exists() and not force_refresh:
        data = read_json(LOCAL_STEAM_APP_LIST_CACHE)
    elif STEAM_APP_LIST_CACHE.exists() and not force_refresh:
        data = read_json(STEAM_APP_LIST_CACHE)
        if isinstance(data, dict):
            data = data.get("applist", {}).get("apps", [])
    else:
        data = []

    app_df = pd.DataFrame(data)
    if app_df.empty:
        return pd.DataFrame(columns=["appid", "name", "name_norm"])

    app_df = app_df.dropna(subset=["appid", "name"])
    app_df["name_norm"] = app_df["name"].map(normalize_name)
    return app_df


def search_steam_store_candidates(game_name: str, limit: int = 10) -> pd.DataFrame:
    url = "https://store.steampowered.com/api/storesearch/"
    data = request_json(url, params={"term": game_name, "l": "english", "cc": "us"})
    rows = []
    for item in data.get("items", []):
        if item.get("type") != "app":
            continue
        rows.append(
            {
                "appid": item.get("id"),
                "name": item.get("name"),
                "source": "steam_store_search",
                "similarity": similarity(game_name, item.get("name", "")),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["appid", "name", "source", "similarity"])
    return pd.DataFrame(rows).sort_values(["similarity", "name"], ascending=[False, True]).head(limit)


def find_app_candidates(game_name: str, app_df: pd.DataFrame | None = None, limit: int = 10) -> pd.DataFrame:
    store_candidates = search_steam_store_candidates(game_name, limit=limit)

    local_candidates = pd.DataFrame(columns=["appid", "name", "source", "similarity"])
    if app_df is not None and not app_df.empty:
        query_norm = normalize_name(game_name)
        candidates = app_df[app_df["name_norm"].str.contains(re.escape(query_norm), na=False)].copy()
        if candidates.empty:
            candidates = app_df.copy()
        candidates["similarity"] = candidates["name"].map(lambda value: similarity(game_name, value))
        candidates["source"] = "local_app_cache"
        local_candidates = candidates[["appid", "name", "source", "similarity"]].sort_values(
            ["similarity", "name"], ascending=[False, True]
        ).head(limit)

    combined = pd.concat([store_candidates, local_candidates], ignore_index=True)
    if combined.empty:
        return combined

    combined["appid"] = combined["appid"].astype(int)
    return (
        combined.sort_values(["similarity", "source", "name"], ascending=[False, False, True])
        .drop_duplicates(subset=["appid"])
        .head(limit)
        .reset_index(drop=True)
    )


def parse_list(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            return []
    return []


def parse_owners_range_mean(value):
    if pd.isna(value):
        return np.nan
    match = re.search(r"([\d,]+)\s*\.\.\s*([\d,]+)", str(value))
    if not match:
        return np.nan
    low = float(match.group(1).replace(",", ""))
    high = float(match.group(2).replace(",", ""))
    return (low + high) / 2


def normalize_company_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["developer_main_clean"] = df["developer_main"].astype("string")
    df["publisher_main_clean"] = df["publisher_main"].astype("string")

    developer_replacements = {
        r"^EA\b.*": "Electronic Arts",
        r"^Maxis.*": "Maxis",
        r"^Respawn.*": "Respawn Entertainment",
        r"^Ubisoft\b.*": "Ubisoft",
        r"^Rockstar\b.*": "Rockstar Games",
        r"^Bethesda Game Studios.*": "Bethesda Game Studios",
        r"^CAPCOM.*": "Capcom",
        r"^Capcom.*": "Capcom",
        r"^BANDAI NAMCO.*": "Bandai Namco",
        r"^Bandai Namco.*": "Bandai Namco",
        r"^Crytek.*": "Crytek",
        r"^Starbreeze.*": "Starbreeze Studios",
    }
    publisher_replacements = {
        r"^Electronic Arts.*": "Electronic Arts",
        r"^EA\b.*": "Electronic Arts",
        r"^Ubisoft.*": "Ubisoft",
        r"^Rockstar.*": "Rockstar Games",
        r"^Bethesda Softworks.*": "Bethesda Softworks",
        r"^CAPCOM.*": "Capcom",
        r"^Capcom.*": "Capcom",
        r"^BANDAI NAMCO.*": "Bandai Namco",
        r"^Bandai Namco.*": "Bandai Namco",
        r"^Warner Bros.*": "Warner Bros. Games",
        r"^WB Games.*": "Warner Bros. Games",
        r"^Kalypso Media.*": "Kalypso Media",
        r"^Amazon.*": "Amazon Games",
        r"^Daybreak Game Company.*": "Daybreak Game Company",
    }

    for pattern, replacement in developer_replacements.items():
        df["developer_main_clean"] = df["developer_main_clean"].str.replace(pattern, replacement, regex=True)
    for pattern, replacement in publisher_replacements.items():
        df["publisher_main_clean"] = df["publisher_main_clean"].str.replace(pattern, replacement, regex=True)

    df = df.drop(columns=["developers", "publishers", "developer_main", "publisher_main"], errors="ignore")
    return df.rename(columns={"developer_main_clean": "developer_main", "publisher_main_clean": "publisher_main"})


def apply_post_release_preprocessing(df_single: pd.DataFrame, reference_clean: pd.DataFrame) -> pd.DataFrame:
    df = df_single.copy()

    if "owners_raw" in df.columns:
        df["owners_raw_mean"] = df["owners_raw"].apply(parse_owners_range_mean)
        df = df.drop(columns=["owners_raw"])

    if "is_free" in df.columns and (df["is_free"] == 1).any():
        print("Warning: this game is free. Training removed free games, so prediction may be unreliable.")

    if "release_year" in df.columns:
        df["release_year"] = df["release_year"].fillna(reference_clean["release_year"].median())
    if "release_month" in df.columns:
        df["release_month"] = df["release_month"].fillna(reference_clean["release_month"].mode(dropna=True)[0])

    df = normalize_company_columns(df)

    for col in ["publisher_main", "developer_main"]:
        if col in df.columns:
            df[col] = df[col].fillna(reference_clean[col].mode(dropna=True)[0])

    for col in ["price_final", "price_initial", "discount_percent"]:
        if col in df.columns:
            df[col] = df[col].fillna(reference_clean[col].median())

    if "steam_recommendations_total" in df.columns:
        df["steam_recommendations_total"] = df["steam_recommendations_total"].fillna(
            reference_clean["steam_recommendations_total"].median()
        )

    median_cols = [
        "achievements_store_total",
        "achievement_avg_percent",
        "achievement_median_percent",
        "news_count",
        "days_since_last_news",
        "news_prelaunch_count_approx",
        "developer_past_games_count",
        "developer_past_avg_log_owners",
        "developer_past_median_log_owners",
        "developer_past_max_log_owners",
        "publisher_past_games_count",
        "publisher_past_avg_log_owners",
        "publisher_past_median_log_owners",
        "publisher_past_max_log_owners",
        "current_players",
    ]
    for col in median_cols:
        if col in df.columns and col in reference_clean.columns:
            df[col] = df[col].fillna(reference_clean[col].median())

    for col in ["developer_has_past_hit_1m", "publisher_has_past_hit_1m"]:
        if col in df.columns and col in reference_clean.columns:
            df[col] = df[col].fillna(reference_clean[col].mode(dropna=True)[0])

    genres_parsed = df["genres"].apply(parse_list) if "genres" in df.columns else pd.Series([[]] * len(df), index=df.index)
    for genre_col in [c for c in reference_clean.columns if c.startswith("genre_")]:
        genre = genre_col.replace("genre_", "")
        df[genre_col] = genres_parsed.apply(lambda values: genre in values)

    categories_parsed = (
        df["categories"].apply(parse_list) if "categories" in df.columns else pd.Series([[]] * len(df), index=df.index)
    )
    for category in SELECTED_CATEGORIES:
        col_name = "category_" + category.lower().replace(" ", "_").replace("-", "_")
        if col_name in reference_clean.columns:
            df[col_name] = categories_parsed.apply(lambda values: int(category in values))

    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    cols_to_drop = [
        "appid",
        "query_name",
        "steam_available",
        "is_free",
        "release_date_text",
        "supported_languages_raw",
        "steamspy_tags",
        "steamspy_tags_weighted",
        "description_length",
        "full_text_length",
        "errors",
        "achievements_count",
        "achievement_min_percent",
        "achievement_max_percent",
        "type",
        "coming_soon",
        "currency",
        "steamspy_score_rank",
        "genres",
        "categories",
    ]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    for col in reference_clean.columns:
        if col not in df.columns:
            if pd.api.types.is_numeric_dtype(reference_clean[col]):
                df[col] = reference_clean[col].median()
            else:
                mode = reference_clean[col].mode(dropna=True)
                df[col] = mode[0] if not mode.empty else pd.NA

    return df[reference_clean.columns]


def build_model_input(df_processed: pd.DataFrame, reference_clean: pd.DataFrame) -> pd.DataFrame:
    target_column = "log_owners_midpoint"
    base_drop_columns = ["name", "short_description", "full_text", "owners_midpoint", "owners_raw_mean"]
    expected_columns = reference_clean.drop(
        columns=[target_column] + [c for c in base_drop_columns if c in reference_clean.columns and c != target_column]
    ).columns.tolist()

    model_input = df_processed.drop(
        columns=[target_column] + [c for c in base_drop_columns if c in df_processed.columns and c != target_column],
        errors="ignore",
    )
    missing = sorted(set(expected_columns) - set(model_input.columns))
    extra = sorted(set(model_input.columns) - set(expected_columns))
    if missing or extra:
        raise ValueError(f"Model input schema mismatch. Missing={missing}, Extra={extra}")
    return model_input[expected_columns]


def choose_best_candidate(candidates: pd.DataFrame) -> tuple[int, str]:
    if candidates.empty:
        raise RuntimeError("No Steam candidates were found.")

    row = candidates.iloc[0]
    print(
        "\nAutomatically selected best Steam match: "
        f"appid={int(row['appid'])} | {row['name']} | score={row['similarity']:.3f} | {row['source']}"
    )
    if len(candidates) > 1:
        print("\nOther close candidates found:")
        for idx, candidate in candidates.iloc[1:5].iterrows():
            print(
                f"- appid={int(candidate['appid'])} | {candidate['name']} | "
                f"score={candidate['similarity']:.3f} | {candidate['source']}"
            )

    return int(row["appid"]), str(row["name"])


def load_model():
    model_path = next((path for path in MODEL_CANDIDATES if path.exists()), None)
    if model_path is None:
        raise FileNotFoundError("No saved model .joblib file was found.")
    return model_path, joblib.load(model_path)


def run_prediction(game_name: str, force: bool = False) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_raw_reference = pd.read_csv(DATA_PATH_RAW_FULL)
    df_clean_reference = pd.read_parquet(DATA_PATH_CLEAN)

    app_df = load_steam_app_list(force_refresh=False)
    candidates = find_app_candidates(game_name, app_df, limit=10)
    selected_appid, selected_query_name = choose_best_candidate(candidates)

    steam_key = os.getenv("STEAM_API_KEY")
    client = ApiClient(steam_key=steam_key, sleep_seconds=1.25)

    print(f"\nExtracting appid={selected_appid} ({selected_query_name})...")
    bundle = extract_one_game(
        client=client,
        appid=selected_appid,
        query_name=selected_query_name,
        reviews_pages=3,
        force=force,
    )
    if bundle.get("errors"):
        print("Extraction warnings/errors:", bundle["errors"])

    df_single_raw = pd.DataFrame([bundle_to_row(bundle)])

    historical = df_raw_reference.copy()
    if "appid" in historical.columns:
        historical = historical[historical["appid"] != selected_appid].copy()

    combined_for_history = pd.concat([historical, df_single_raw], ignore_index=True, sort=False)
    combined_for_history = add_developer_publisher_history_features(combined_for_history)
    combined_for_history = combined_for_history.drop(columns=["competition_same_month_count"], errors="ignore")
    combined_for_history = add_competition_features(combined_for_history)

    df_single_enriched = combined_for_history[combined_for_history["appid"] == selected_appid].tail(1).copy()
    df_single_processed = apply_post_release_preprocessing(df_single_enriched, df_clean_reference)
    model_input = build_model_input(df_single_processed, df_clean_reference)

    model_path, model = load_model()
    predicted_log_owners = float(model.predict(model_input)[0])
    predicted_owners = float(np.expm1(predicted_log_owners))

    actual_owners = (
        float(df_single_processed.iloc[0]["owners_midpoint"])
        if pd.notna(df_single_processed.iloc[0]["owners_midpoint"])
        else np.nan
    )
    actual_log_owners = (
        float(df_single_processed.iloc[0]["log_owners_midpoint"])
        if pd.notna(df_single_processed.iloc[0]["log_owners_midpoint"])
        else np.nan
    )

    summary = pd.DataFrame(
        [
            {
                "appid": selected_appid,
                "name": df_single_processed.iloc[0]["name"],
                "model_path": str(model_path),
                "predicted_log_owners": predicted_log_owners,
                "predicted_owners": predicted_owners,
                "steamspy_owners_range": df_single_raw.iloc[0].get("owners_raw"),
                "steamspy_owners_midpoint": actual_owners,
                "steamspy_log_owners_midpoint": actual_log_owners,
                "absolute_error_owners": abs(predicted_owners - actual_owners) if pd.notna(actual_owners) else np.nan,
                "prediction_to_steamspy_ratio": predicted_owners / actual_owners
                if pd.notna(actual_owners) and actual_owners != 0
                else np.nan,
            }
        ]
    )

    raw_output = OUTPUT_DIR / f"{selected_appid}_raw_pipeline_row.json"
    processed_output = OUTPUT_DIR / f"{selected_appid}_post_release_preprocessed.parquet"
    model_input_output = OUTPUT_DIR / f"{selected_appid}_model_input.parquet"
    prediction_output = OUTPUT_DIR / f"{selected_appid}_prediction_summary.csv"

    df_single_enriched.to_json(raw_output, orient="records", force_ascii=False, indent=2)
    df_single_processed.to_parquet(processed_output, index=False)
    model_input.to_parquet(model_input_output, index=False)
    summary.to_csv(prediction_output, index=False)

    print("\nPrediction complete")
    print(f"Game: {summary.loc[0, 'name']}")
    print(f"Predicted owners: {summary.loc[0, 'predicted_owners']:,.0f}")
    print(f"SteamSpy midpoint owners: {summary.loc[0, 'steamspy_owners_midpoint']:,.0f}")
    print(f"Prediction / SteamSpy ratio: {summary.loc[0, 'prediction_to_steamspy_ratio']:.2f}x")
    print("\nSaved files:")
    print(f"- {raw_output}")
    print(f"- {processed_output}")
    print(f"- {model_input_output}")
    print(f"- {prediction_output}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict Steam owners from a free-form game name.")
    parser.add_argument("game_name", nargs="*", help="Game name to search on Steam.")
    parser.add_argument("--force", action="store_true", help="Force refresh of cached raw API data for the selected appid.")
    args = parser.parse_args()

    print("External Steam applied-model prediction")
    print("---------------------------------------")
    game_name = " ".join(args.game_name).strip()
    if not game_name:
        game_name = input("Game name: ").strip()
    if not game_name:
        raise ValueError("Game name cannot be empty.")

    run_prediction(game_name, force=args.force)


if __name__ == "__main__":
    main()
