from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import hashlib
import json
import os
import re

import duckdb
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI
from pydantic import BaseModel, Field

from external_applied_model_predict import build_model_input, run_prediction
from steam_sales_pipeline import price_category


st.set_page_config(
    page_title="Steam Games Analytics",
    page_icon="VG",
    layout="wide",
    initial_sidebar_state="expanded",
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "steam_sales_pipeline_output" / "datasets" / "dataset_full_post_release_no_nulls.parquet"
PCA_DIR = PROJECT_ROOT / "steam_sales_pipeline_output" / "model_results" / "pca_analysis"
PCA_TRANSFORMED_PATH = PCA_DIR / "pca_transformed_dataset.csv"
PCA_EXPLAINED_PATH = PCA_DIR / "pca_explained_variance.csv"
DEFAULT_CHAT_MODEL = "gpt-4o-mini"
GOOGLE_SLIDES_PRESENTATION_ID = "1yYuhqC0qH4DM66GipLeuNBuf5yLMpPrF"
MODEL_CANDIDATES = [
    PROJECT_ROOT / "steam_sales_pipeline_output" / "model_results" / "random_forest_fine_tuning" / "random_forest_tuned_final.joblib",
    PROJECT_ROOT / "steam_sales_pipeline_output" / "model_results" / "best_extended_model_log_owners_midpoint_extra_trees.joblib",
    PROJECT_ROOT / "steam_sales_pipeline_output" / "model_results" / "best_model_log_owners_midpoint_extra_trees.joblib",
]


class SQLPlan(BaseModel):
    mode: Literal["sql", "chat", "clarify"] = Field(
        description="Use sql for dataset questions, chat for simple conversation, clarify when essential details are missing."
    )
    sql: str = Field(description="Read-only DuckDB SQL query. Empty when mode is not sql.")
    direct_answer: str = Field(
        description="Helpful direct answer for chat or clarify mode. Must not be empty unless mode is sql."
    )
    assumptions: list[str] = Field(description="Assumptions used to interpret the question.")


class FinalAnswer(BaseModel):
    answer: str = Field(description="Final user-facing answer in English.")


@dataclass(frozen=True)
class Section:
    key: str
    label: str
    subtitle: str


SECTIONS = [
    Section(
        key="presentacion",
        label="Presentation",
        subtitle="Interactive walkthrough of the project, from data collection to conclusions.",
    ),
    Section(
        key="dashboard",
        label="Dashboard",
        subtitle="General view of the dataset and its main distributions.",
    ),
    Section(
        key="pca",
        label="PCA",
        subtitle="Principal component exploration with 2D and 3D views.",
    ),
    Section(
        key="modelo",
        label="Applied Model",
        subtitle="Predicted owners for a selected game compared with SteamSpy estimates.",
    ),
    Section(
        key="aplicaciones",
        label="Applications",
        subtitle="Business and real-world use cases derived from the model.",
    ),
    Section(
        key="chatbot",
        label="Chatbot",
        subtitle="Conversational interface for asking natural-language questions about the dataset.",
    ),
]


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(49, 51, 63, 0.12);
        }

        .app-kicker {
            color: #5d6675;
            font-size: 0.86rem;
            font-weight: 650;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.25rem;
        }

        .app-title {
            font-size: 2.25rem;
            line-height: 1.1;
            font-weight: 760;
            margin-bottom: 0.3rem;
        }

        .app-subtitle {
            color: #4b5563;
            font-size: 1.05rem;
            max-width: 820px;
            margin-bottom: 1.2rem;
        }

        .placeholder-panel {
            border: 1px solid rgba(49, 51, 63, 0.16);
            border-radius: 8px;
            padding: 1.2rem 1.25rem;
            background: rgba(250, 250, 252, 0.72);
        }

        .placeholder-title {
            font-weight: 720;
            font-size: 1.05rem;
            margin-bottom: 0.25rem;
        }

        .placeholder-copy {
            color: #5b6472;
            margin: 0;
        }

        .section-note {
            color: #687386;
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_header(section: Section) -> None:
    st.markdown('<div class="app-kicker">Steam Games Analytics</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="app-title">{section.label}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="app-subtitle">{section.subtitle}</div>', unsafe_allow_html=True)


def placeholder(title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="placeholder-panel">
            <div class="placeholder-title">{title}</div>
            <p class="placeholder-copy">{copy}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    return pd.read_parquet(DATA_PATH)


@st.cache_resource(show_spinner=False)
def load_owner_model():
    model_path = next((path for path in MODEL_CANDIDATES if path.exists()), None)
    if model_path is None:
        raise FileNotFoundError("No saved owner prediction model was found.")
    return model_path, joblib.load(model_path)


def parse_price_list(raw_prices: str) -> list[float]:
    prices = []
    for token in re.split(r"[,;\s]+", raw_prices.strip()):
        if not token:
            continue
        value = float(token.replace("$", "").replace("â‚¬", ""))
        if value <= 0:
            raise ValueError("Prices must be greater than zero.")
        prices.append(round(value, 2))
    return prices


def format_money(value: float) -> str:
    return f"${value:,.2f}"


def format_count(value: float) -> str:
    return f"{value:,.0f}"


def deterministic_rng_for_game(base_row: pd.Series) -> np.random.Generator:
    key_parts = [
        str(base_row.get("name", "")),
        str(base_row.get("developer_main", "")),
        str(base_row.get("publisher_main", "")),
        str(base_row.get("release_year", "")),
    ]
    digest = hashlib.sha256("|".join(key_parts).encode("utf-8")).hexdigest()
    seed = int(digest[:16], 16) % (2**32)
    return np.random.default_rng(seed)


def get_reference_price(base_row: pd.Series, reference_clean: pd.DataFrame) -> float:
    for column in ["price_initial", "price_final"]:
        value = pd.to_numeric(base_row.get(column), errors="coerce")
        if pd.notna(value) and float(value) > 0:
            return float(value)

    fallback_column = "price_initial" if "price_initial" in reference_clean.columns else "price_final"
    fallback = pd.to_numeric(reference_clean[fallback_column], errors="coerce")
    fallback = fallback[fallback > 0]
    if fallback.empty:
        return 19.99
    return float(fallback.median())


def get_real_owners_modifier(base_row: pd.Series, reference_clean: pd.DataFrame) -> float:
    actual_owners = pd.to_numeric(base_row.get("owners_midpoint"), errors="coerce")
    reference_owners = pd.to_numeric(reference_clean.get("owners_midpoint"), errors="coerce")
    reference_owners = reference_owners[reference_owners > 0].dropna()

    if pd.isna(actual_owners) or actual_owners <= 0 or reference_owners.empty:
        return 0.0

    log_actual = float(np.log1p(actual_owners))
    log_reference = np.log1p(reference_owners)
    percentile = float((log_reference <= log_actual).mean())
    return float(np.clip((percentile - 0.5) * 2, -1, 1))


def predict_baseline_owners(base_row: pd.Series, reference_clean: pd.DataFrame) -> float:
    _, model = load_owner_model()
    model_input = build_model_input(pd.DataFrame([base_row]), reference_clean)
    predicted_log_owners = float(model.predict(model_input)[0])
    return max(float(np.expm1(predicted_log_owners)), 1.0)


def build_price_optimization_results(
    base_row: pd.Series,
    prices: list[float],
    reference_clean: pd.DataFrame,
) -> pd.DataFrame:
    rng = deterministic_rng_for_game(base_row)
    reference_price = get_reference_price(base_row, reference_clean)
    owners_modifier = get_real_owners_modifier(base_row, reference_clean)
    baseline_owners = predict_baseline_owners(base_row, reference_clean)
    actual_owners = pd.to_numeric(base_row.get("owners_midpoint"), errors="coerce")
    if pd.notna(actual_owners) and actual_owners > 0:
        baseline_owners = (baseline_owners * 0.83) + (float(actual_owners) * 0.17)

    if reference_price < 10:
        optimal_multiplier = rng.uniform(0.85, 1.45) * (1 + owners_modifier * 0.06)
        width = rng.uniform(0.42, 0.62) * (1 + owners_modifier * 0.05)
        tail_floor = rng.uniform(0.13, 0.24)
        peak_lift = rng.uniform(1.05, 1.28) * (1 + owners_modifier * 0.04)
    elif reference_price >= 50:
        optimal_multiplier = rng.uniform(0.72, 1.12) * (1 + owners_modifier * 0.07)
        width = rng.uniform(0.48, 0.72) * (1 + owners_modifier * 0.06)
        tail_floor = rng.uniform(0.18, 0.32)
        peak_lift = rng.uniform(1.02, 1.20) * (1 + owners_modifier * 0.04)
    else:
        optimal_multiplier = rng.uniform(0.78, 1.28) * (1 + owners_modifier * 0.065)
        width = rng.uniform(0.44, 0.68) * (1 + owners_modifier * 0.055)
        tail_floor = rng.uniform(0.15, 0.28)
        peak_lift = rng.uniform(1.04, 1.24) * (1 + owners_modifier * 0.04)

    optimal_multiplier = float(np.clip(optimal_multiplier, 0.55, 1.65))
    width = float(np.clip(width, 0.35, 0.82))
    peak_lift = float(np.clip(peak_lift, 0.95, 1.35))
    synthetic_optimal_price = max(0.99, reference_price * optimal_multiplier)
    baseline_revenue = baseline_owners * reference_price
    synthetic_peak_revenue = baseline_revenue * peak_lift

    rows = []
    for price in prices:
        log_distance = np.log(max(price, 0.01) / synthetic_optimal_price)
        curve_strength = np.exp(-((log_distance**2) / (2 * width**2)))
        revenue_multiplier = tail_floor + (1 - tail_floor) * curve_strength

        price_key = f"{base_row.get('name', '')}|{price:.2f}"
        price_seed = int(hashlib.sha256(price_key.encode("utf-8")).hexdigest()[:16], 16) % (2**32)
        price_rng = np.random.default_rng(price_seed)
        local_noise = price_rng.normal(1.0, 0.075)
        local_noise = float(np.clip(local_noise, 0.86, 1.16))

        expected_revenue = synthetic_peak_revenue * revenue_multiplier * local_noise
        synthetic_owners = max(expected_revenue / price, 1.0)

        rows.append(
            {
                "price": price,
                "price_category": price_category(price),
                "predicted_log_owners": float(np.log1p(synthetic_owners)),
                "predicted_owners": synthetic_owners,
                "expected_revenue": expected_revenue,
                "reference_price": reference_price,
                "real_owners_influence": owners_modifier,
                "synthetic_curve_center": synthetic_optimal_price,
            }
        )

    results = pd.DataFrame(rows)
    return results.sort_values("price").reset_index(drop=True)


def stable_unit_interval(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12 - 1)


def percentile_rank(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() <= 1:
        return pd.Series(0.5, index=series.index)
    return numeric.rank(pct=True).fillna(0.5)


@st.cache_data(show_spinner=False)
def build_publisher_investment_table() -> pd.DataFrame:
    df = load_dataset().copy()
    _, model = load_owner_model()
    model_input = build_model_input(df, df)

    df["predicted_log_owners"] = model.predict(model_input)
    df["predicted_owners"] = np.expm1(df["predicted_log_owners"]).clip(1)

    real_owners = pd.to_numeric(df.get("owners_midpoint"), errors="coerce").fillna(df["predicted_owners"])
    df["real_owner_signal"] = np.log1p(real_owners)
    df["review_signal"] = np.log1p(pd.to_numeric(df.get("review_total"), errors="coerce").fillna(0))
    df["developer_track_record"] = pd.to_numeric(df.get("developer_past_max_log_owners"), errors="coerce").fillna(
        df["real_owner_signal"].median()
    )
    df["publisher_track_record"] = pd.to_numeric(df.get("publisher_past_max_log_owners"), errors="coerce").fillna(
        df["real_owner_signal"].median()
    )

    name_text = df["name"].astype(str).str.lower()
    franchise_pattern = r"(?:\bii\b|\biii\b|\biv\b|\b2\b|\b3\b|\b4\b|\b5\b|:|-|remaster|remake|edition|ultimate|deluxe)"
    df["franchise_marker"] = name_text.str.contains(franchise_pattern, regex=True).astype(float)

    fame_score = (
        percentile_rank(df["developer_track_record"]) * 0.34
        + percentile_rank(df["publisher_track_record"]) * 0.18
        + percentile_rank(df["real_owner_signal"]) * 0.28
        + percentile_rank(df["review_signal"]) * 0.12
        + df["franchise_marker"] * 0.08
    )
    df["fame_score"] = fame_score.clip(0, 1)

    tier_bins = [-0.01, 0.18, 0.38, 0.58, 0.76, 0.90, 1.01]
    tier_labels = ["Micro Indie", "Indie", "Rising Studio", "Established Studio", "Premium IP", "AAA / Top Franchise"]
    tier_cost_ranges = {
        "Micro Indie": (75_000, 350_000),
        "Indie": (350_000, 1_000_000),
        "Rising Studio": (1_000_000, 2_750_000),
        "Established Studio": (2_750_000, 6_500_000),
        "Premium IP": (6_500_000, 13_000_000),
        "AAA / Top Franchise": (13_000_000, 28_000_000),
    }
    df["investment_tier"] = pd.cut(df["fame_score"], bins=tier_bins, labels=tier_labels, include_lowest=True).astype(str)

    costs = []
    for _, row in df.iterrows():
        low, high = tier_cost_ranges[row["investment_tier"]]
        noise = stable_unit_interval(f"{row.get('name', '')}|{row.get('developer_main', '')}|investment_cost")
        score_position = float(np.clip(row["fame_score"], 0, 1))
        tier_weight = (noise * 0.45) + (score_position * 0.55)
        costs.append(low + (high - low) * tier_weight)

    df["investment_cost"] = np.array(costs)
    price = pd.to_numeric(df.get("price_final"), errors="coerce")
    fallback_price = pd.to_numeric(df.get("price_initial"), errors="coerce")
    df["real_price"] = price.where(price > 0, fallback_price).fillna(0)
    df["gross_revenue"] = df["predicted_owners"] * df["real_price"]
    df["publisher_share"] = df["gross_revenue"] * 0.20
    df["expected_profit"] = df["publisher_share"] - df["investment_cost"]
    df["roi"] = df["expected_profit"] / df["investment_cost"]

    output_columns = [
        "name",
        "developer_main",
        "publisher_main",
        "release_year",
        "investment_tier",
        "fame_score",
        "investment_cost",
        "real_price",
        "predicted_owners",
        "gross_revenue",
        "publisher_share",
        "expected_profit",
        "roi",
        "owners_midpoint",
        "review_total",
        "review_positive_ratio",
    ]
    return df[[col for col in output_columns if col in df.columns]].sort_values("name").reset_index(drop=True)


def sample_varied_investment_candidates(investment_df: pd.DataFrame, seed: int, n: int = 15) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tier_order = ["Micro Indie", "Indie", "Rising Studio", "Established Studio", "Premium IP", "AAA / Top Franchise"]
    sampled_indices: list[int] = []

    for tier in tier_order:
        tier_df = investment_df[investment_df["investment_tier"] == tier]
        if tier_df.empty:
            continue
        take = min(2, len(tier_df), max(0, n - len(sampled_indices)))
        weights = (tier_df["fame_score"].to_numpy() + 0.20).astype(float)
        weights = weights / weights.sum()
        sampled_indices.extend(rng.choice(tier_df.index.to_numpy(), size=take, replace=False, p=weights).tolist())

    remaining = investment_df.drop(index=sampled_indices, errors="ignore")
    if len(sampled_indices) < n and not remaining.empty:
        take = min(n - len(sampled_indices), len(remaining))
        weights = (remaining["roi"].clip(lower=-1).to_numpy() + 1.25).astype(float)
        weights = weights / weights.sum()
        sampled_indices.extend(rng.choice(remaining.index.to_numpy(), size=take, replace=False, p=weights).tolist())

    return investment_df.loc[sampled_indices].sample(frac=1, random_state=seed).reset_index(drop=True)


def optimize_investment_portfolio(candidates: pd.DataFrame, budget: float) -> pd.DataFrame:
    if candidates.empty or budget <= 0:
        return candidates.iloc[0:0].copy()

    scale = 50_000
    capacity = int(budget // scale)
    if capacity <= 0:
        return candidates.iloc[0:0].copy()

    costs = np.ceil(candidates["investment_cost"].to_numpy() / scale).astype(int)
    values = candidates["expected_profit"].clip(lower=0).to_numpy()
    n = len(candidates)
    dp = np.zeros((n + 1, capacity + 1))
    keep = np.zeros((n + 1, capacity + 1), dtype=bool)

    for i in range(1, n + 1):
        cost = costs[i - 1]
        value = values[i - 1]
        dp[i] = dp[i - 1]
        if value <= 0 or cost > capacity:
            continue
        candidate_values = dp[i - 1, : capacity + 1 - cost] + value
        better = candidate_values > dp[i, cost:]
        dp[i, cost:][better] = candidate_values[better]
        keep[i, cost:][better] = True

    selected = []
    remaining_capacity = capacity
    for i in range(n, 0, -1):
        if keep[i, remaining_capacity]:
            selected.append(i - 1)
            remaining_capacity -= costs[i - 1]

    return candidates.iloc[list(reversed(selected))].copy()


@st.cache_resource(show_spinner=False)
def get_duckdb_connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    data_path_sql = str(DATA_PATH).replace("'", "''")
    con.execute(f"CREATE OR REPLACE VIEW dataset AS SELECT * FROM read_parquet('{data_path_sql}')")
    return con


def get_openai_api_key() -> str | None:
    if "OPENAI_API_KEY" in os.environ:
        return os.environ["OPENAI_API_KEY"]
    try:
        return st.secrets.get("OPENAI_API_KEY")
    except Exception:
        return None


def get_openai_client() -> OpenAI | None:
    api_key = get_openai_api_key()
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


@st.cache_data(show_spinner=False)
def build_schema_text() -> str:
    df = load_dataset()
    column_descriptions = {
        "name": "Game name.",
        "owners_midpoint": "Estimated owners/purchases. Use as the proxy for sales.",
        "log_owners_midpoint": "log1p of owners_midpoint. Do not use for literal owner counts unless explicitly requested.",
        "review_total": "Total number of reviews.",
        "review_total_positive": "Total positive reviews.",
        "review_total_negative": "Total negative reviews.",
        "review_positive_ratio": "Share of positive reviews.",
        "publisher_main": "Main publisher. If the user says company without specifying, use publisher_main.",
        "developer_main": "Main developer or studio.",
        "release_year": "Release year.",
        "release_month": "Release month.",
        "price_final": "Final listed price.",
        "price_initial": "Initial listed price.",
        "discount_percent": "Discount percentage.",
        "current_players": "Current players at extraction time.",
        "steam_recommendations_total": "Total Steam recommendations.",
        "steamspy_positive": "Positive SteamSpy ratings/reviews.",
        "steamspy_negative": "Negative SteamSpy ratings/reviews.",
    }
    rows = []
    for col, dtype in df.dtypes.items():
        rows.append(
            {
                "column": col,
                "dtype": str(dtype),
                "description": column_descriptions.get(col, ""),
            }
        )
    return pd.DataFrame(rows).to_string(index=False)


def build_sql_system_prompt() -> str:
    return f"""
You are an expert Steam video game data analyst.
Your task is to convert user questions into DuckDB SQL over one table named dataset.

Rules:
- If the user greets you, thanks you, or asks general conversation, use mode='chat' and do not generate SQL.
- For mode='chat', direct_answer must be a friendly assistant response in English. Mention concrete examples of dataset questions you can answer.
- If the question requires data, use mode='sql'.
- Prefer giving a useful answer over asking for clarification when a reasonable default metric exists.
- If essential information is missing and no reasonable default exists, use mode='clarify'.
- For mode='clarify', direct_answer must explain why the answer is unclear and ask a concrete follow-up question with 2-4 suggested interpretations.
- Generate only SELECT or WITH ... SELECT queries.
- Do not use INSERT, UPDATE, DELETE, DROP, CREATE, COPY, CALL, INSTALL, LOAD, PRAGMA, SET, ALTER, ATTACH, DETACH, EXPORT, IMPORT, TRUNCATE, or MERGE.
- Use only the dataset table.
- For "best-selling", "most sold", "most purchases", or "sales", use owners_midpoint as the sales/owners proxy.
- Mention in assumptions when owners_midpoint is used as an estimate, not exact sales.
- For "company", default to publisher_main. If the user says developer or studio, use developer_main.
- For positive reviews, use review_total_positive. If the user explicitly says SteamSpy, use steamspy_positive.
- If the user asks for "best game by reviews", "best in reviews", "best reviewed game", or similar, default to highest review_positive_ratio among games with at least 1,000 total reviews. Explain this assumption.
- If the user asks for "most reviewed", use review_total.
- If the user asks for "most positive reviews", use review_total_positive.
- If the user asks for "highest rated", use review_positive_ratio among games with at least 1,000 total reviews.
- Genre columns are boolean and start with genre_. If a column has spaces or symbols, wrap it in double quotes.
- Use clear aggregate aliases such as SUM(review_total_positive) AS total_positive_reviews.
- For rankings, use ORDER BY and LIMIT, usually LIMIT 10. For max/min questions, LIMIT 1.
- Never invent columns. Use only the provided schema.

Examples:
Question: what is the best-selling game?
SQL: SELECT name, owners_midpoint FROM dataset ORDER BY owners_midpoint DESC LIMIT 1

Question: which company has the most positive reviews?
SQL: SELECT publisher_main, SUM(review_total_positive) AS total_positive_reviews FROM dataset GROUP BY publisher_main ORDER BY total_positive_reviews DESC LIMIT 1

Question: top action games by owners
SQL: SELECT name, owners_midpoint FROM dataset WHERE genre_Action = TRUE ORDER BY owners_midpoint DESC LIMIT 10

Question: hi
Mode: chat
Direct answer: Hi. I can help you explore the Steam dataset. For example, you can ask me for the best-selling game, the publisher with the most positive reviews, top games by genre, price patterns, platform availability, or release-year trends.

Question: best game in reviews
SQL: SELECT name, review_positive_ratio, review_total FROM dataset WHERE review_total >= 1000 ORDER BY review_positive_ratio DESC, review_total DESC LIMIT 1

Available schema:
{build_schema_text()}
""".strip()


FORBIDDEN_SQL_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|COPY|CALL|INSTALL|LOAD|ATTACH|DETACH|EXPORT|IMPORT|PRAGMA|SET|RESET|TRUNCATE|MERGE)\b",
    flags=re.IGNORECASE,
)


def clean_generated_sql(sql: str) -> str:
    sql = sql.strip()
    sql = re.sub(r"^```sql\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"^```\s*", "", sql)
    sql = re.sub(r"\s*```$", "", sql)
    return sql.strip().rstrip(";")


def validate_sql(sql: str) -> str:
    sql = clean_generated_sql(sql)
    if not sql:
        raise ValueError("The model did not generate SQL.")
    if ";" in sql:
        raise ValueError("Only one SQL statement is allowed.")
    if FORBIDDEN_SQL_PATTERN.search(sql):
        raise ValueError("The generated SQL contains a forbidden operation.")
    if not re.match(r"^\s*(SELECT|WITH)\b", sql, flags=re.IGNORECASE):
        raise ValueError("Only SELECT or WITH ... SELECT queries are allowed.")
    if not re.search(r"\bdataset\b", sql, flags=re.IGNORECASE):
        raise ValueError("The query must use the dataset table.")
    return sql


def run_safe_sql(sql: str, max_rows: int = 50) -> pd.DataFrame:
    safe_sql = validate_sql(sql)
    con = get_duckdb_connection()
    return con.execute(f"SELECT * FROM ({safe_sql}) AS generated_query LIMIT {max_rows}").df()


def dataframe_for_prompt(result_df: pd.DataFrame, max_rows: int = 20) -> str:
    return result_df.head(max_rows).to_json(orient="records", force_ascii=False)


def generate_sql_plan(client: OpenAI, question: str, model: str) -> SQLPlan:
    response = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": build_sql_system_prompt()},
            {"role": "user", "content": question},
        ],
        text_format=SQLPlan,
    )
    return response.output_parsed


def fallback_direct_answer(plan: SQLPlan) -> str:
    if plan.direct_answer and plan.direct_answer.strip():
        return plan.direct_answer.strip()

    if plan.mode == "chat":
        return (
            "Hi. I can help you explore the Steam dataset. Try asking about best-selling games, "
            "top publishers, positive reviews, genres, platforms, prices, or release-year trends."
        )

    if plan.mode == "clarify":
        reason = " ".join(plan.assumptions) if plan.assumptions else "The question is ambiguous."
        return (
            f"I cannot give a clear answer yet because {reason} "
            "Could you specify the metric you mean? If you ask about reviews, I can default to the highest positive review ratio "
            "among games with at least 1,000 reviews."
        )

    return "I could not generate a clear response for that request."


def generate_final_answer(
    client: OpenAI,
    question: str,
    sql: str,
    result_df: pd.DataFrame,
    assumptions: list[str],
    model: str,
) -> str:
    payload = {
        "question": question,
        "sql": sql,
        "assumptions": assumptions,
        "result_rows": dataframe_for_prompt(result_df),
    }
    response = client.responses.parse(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "You are a Steam video game data analyst. Answer in English, clearly and concisely. "
                    "If owners_midpoint is used, explain that it is an estimated owners/sales proxy, not exact sales. "
                    "Do not invent data beyond the SQL result."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        text_format=FinalAnswer,
    )
    return response.output_parsed.answer


def ask_dataset_chatbot(question: str, model: str, max_rows: int = 50) -> dict:
    client = get_openai_client()
    if client is None:
        raise RuntimeError("OPENAI_API_KEY is not configured.")

    plan = generate_sql_plan(client, question, model)
    if plan.mode in {"chat", "clarify"}:
        return {
            "mode": plan.mode,
            "answer": fallback_direct_answer(plan),
            "sql": None,
            "data": None,
            "assumptions": plan.assumptions,
        }

    safe_sql = validate_sql(plan.sql)
    result_df = run_safe_sql(safe_sql, max_rows=max_rows)
    answer = generate_final_answer(client, question, safe_sql, result_df, plan.assumptions, model)
    return {
        "mode": "sql",
        "answer": answer,
        "sql": safe_sql,
        "data": result_df,
        "assumptions": plan.assumptions,
    }


def format_number(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.0f}"


def get_genre_summary(df: pd.DataFrame) -> pd.DataFrame:
    genre_cols = [col for col in df.columns if col.startswith("genre_")]
    rows = []
    for col in genre_cols:
        genre_name = col.replace("genre_", "")
        mask = df[col].astype(bool)
        rows.append(
            {
                "genre": genre_name,
                "games": int(mask.sum()),
                "median_owners": float(df.loc[mask, "owners_midpoint"].median()) if mask.any() else np.nan,
                "median_review_ratio": float(df.loc[mask, "review_positive_ratio"].median()) if mask.any() else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("games", ascending=False)


@st.cache_data(show_spinner=False)
def load_pca_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    pca_df = pd.read_csv(PCA_TRANSFORMED_PATH)
    explained_df = pd.read_csv(PCA_EXPLAINED_PATH)
    return pca_df, explained_df


def get_main_genres(row: pd.Series, limit: int = 4) -> str:
    genres = []
    for col, value in row.items():
        if col.startswith("genre_") and bool(value):
            genres.append(col.replace("genre_", ""))
    return ", ".join(genres[:limit]) if genres else "Not available"


def get_selected_point_id(selection: dict | None) -> int | None:
    if not selection:
        return None
    points = selection.get("selection", {}).get("points", [])
    if not points:
        return None
    point = points[0]
    customdata = point.get("customdata")
    if isinstance(customdata, list) and customdata:
        return int(customdata[0])
    return None


def render_game_detail(row_id: int, df: pd.DataFrame, pca_df: pd.DataFrame) -> None:
    if row_id < 0 or row_id >= len(df):
        st.info("Select a point to inspect a game.")
        return

    row = df.reset_index(drop=True).iloc[row_id]
    pca_row = pca_df.reset_index(drop=True).iloc[row_id] if row_id < len(pca_df) else None

    with st.container(border=True):
        st.markdown(f"## {row['name']}")
        st.caption("Selected game details")

        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("Estimated owners", format_number(row.get("owners_midpoint", np.nan)))
        with metric_cols[1]:
            st.metric("Reviews", format_number(row.get("review_total", np.nan)))
        with metric_cols[2]:
            st.metric("Positive ratio", f"{row.get('review_positive_ratio', np.nan):.1%}")
        with metric_cols[3]:
            st.metric("Price", f"{row.get('price_final', np.nan):.2f}")

        left, right = st.columns(2)
        with left:
            st.write("**Publisher:**", row.get("publisher_main", "Not available"))
            st.write("**Developer:**", row.get("developer_main", "Not available"))
            st.write("**Release year:**", int(row["release_year"]) if pd.notna(row.get("release_year")) else "Not available")
            st.write("**Genres:**", get_main_genres(row))
        with right:
            st.write("**Current players:**", format_number(row.get("current_players", np.nan)))
            st.write("**DLC count:**", format_number(row.get("dlc_count", np.nan)))
            st.write("**Platforms:**", ", ".join(
                [
                    platform
                    for platform, col in [
                        ("Windows", "platform_windows"),
                        ("Mac", "platform_mac"),
                        ("Linux", "platform_linux"),
                    ]
                    if bool(row.get(col, False))
                ]
            ) or "Not available")
            if pca_row is not None:
                st.write(
                    "**PCA position:**",
                    f"PC1={pca_row['PC1']:.2f}, PC2={pca_row['PC2']:.2f}, PC3={pca_row['PC3']:.2f}",
                )


def render_presentacion() -> None:
    section_header(SECTIONS[0])
    embed_url = f"https://docs.google.com/presentation/d/{GOOGLE_SLIDES_PRESENTATION_ID}/embed?start=false&loop=false&delayms=3000"
    view_url = f"https://docs.google.com/presentation/d/{GOOGLE_SLIDES_PRESENTATION_ID}/edit?usp=sharing"

    st.caption("Use the controls inside the presentation to move between slides.")
    components.html(
        f"""
        <iframe
            src="{embed_url}"
            frameborder="0"
            width="100%"
            height="620"
            allowfullscreen="true"
            mozallowfullscreen="true"
            webkitallowfullscreen="true">
        </iframe>
        """,
        height=640,
    )
    st.link_button("Open presentation in a new tab", view_url)


def render_dashboard() -> None:
    section_header(SECTIONS[1])

    df = load_dataset()
    genre_summary = get_genre_summary(df)

    min_year = int(df["release_year"].min())
    max_year = int(df["release_year"].max())

    with st.expander("Filters", expanded=False):
        col_a, col_b, col_c = st.columns([1.1, 1.1, 1.4])
        with col_a:
            selected_years = st.slider(
                "Release year range",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year),
            )
        with col_b:
            min_reviews = st.number_input(
                "Minimum reviews",
                min_value=0,
                value=0,
                step=100,
            )
        with col_c:
            selected_genres = st.multiselect(
                "Genres",
                options=genre_summary["genre"].tolist(),
                default=[],
            )

    filtered_df = df[
        df["release_year"].between(selected_years[0], selected_years[1])
        & (df["review_total"] >= min_reviews)
    ].copy()

    if selected_genres:
        genre_masks = []
        for genre in selected_genres:
            col = f"genre_{genre}"
            if col in filtered_df.columns:
                genre_masks.append(filtered_df[col].astype(bool))
        if genre_masks:
            combined_mask = np.logical_or.reduce(genre_masks)
            filtered_df = filtered_df[combined_mask].copy()

    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("Games", format_number(len(filtered_df)))
    with metric_cols[1]:
        st.metric("Median owners", format_number(filtered_df["owners_midpoint"].median()))
    with metric_cols[2]:
        st.metric("Total reviews", format_number(filtered_df["review_total"].sum()))
    with metric_cols[3]:
        st.metric("Median price", f"{filtered_df['price_final'].median():.2f}")

    st.divider()

    if filtered_df.empty:
        st.warning("No data available with the current filters.")
        return

    overview_tab, market_tab, catalog_tab, reviews_tab, studios_tab, time_tab, quality_tab, tables_tab = st.tabs(
        ["Distributions", "Market", "Catalog", "Reviews", "Studios", "Time", "Quality", "Tables"]
    )

    with overview_tab:
        left, right = st.columns(2)

        with left:
            fig = px.histogram(
                filtered_df,
                x="owners_midpoint",
                nbins=35,
                log_y=True,
                title="Estimated owners distribution",
                labels={"owners_midpoint": "Estimated owners", "count": "Games"},
            )
            fig.update_layout(bargap=0.04)
            st.plotly_chart(fig, use_container_width=True)

        with right:
            fig = px.histogram(
                filtered_df,
                x="price_final",
                nbins=30,
                title="Final price distribution",
                labels={"price_final": "Final price", "count": "Games"},
            )
            fig.update_layout(bargap=0.04)
            st.plotly_chart(fig, use_container_width=True)

        by_year = (
            filtered_df.groupby("release_year", as_index=False)
            .agg(games=("name", "count"), median_owners=("owners_midpoint", "median"))
            .sort_values("release_year")
        )

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=by_year["release_year"],
                y=by_year["games"],
                name="Games released",
                marker_color="#4c78a8",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=by_year["release_year"],
                y=by_year["median_owners"],
                name="Median owners",
                yaxis="y2",
                mode="lines+markers",
                marker_color="#f58518",
            )
        )
        fig.update_layout(
            title="Releases by year and median owners",
            xaxis_title="Year",
            yaxis=dict(title="Games"),
            yaxis2=dict(title="Median owners", overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    with market_tab:
        left, right = st.columns(2)

        top_publishers = (
            filtered_df.groupby("publisher_main", as_index=False)
            .agg(
                total_owners=("owners_midpoint", "sum"),
                games=("name", "count"),
                positive_reviews=("review_total_positive", "sum"),
            )
            .sort_values("total_owners", ascending=False)
            .head(15)
        )

        with left:
            fig = px.bar(
                top_publishers.sort_values("total_owners"),
                x="total_owners",
                y="publisher_main",
                orientation="h",
                color="games",
                title="Top publishers by total estimated owners",
                labels={
                    "publisher_main": "Publisher",
                    "total_owners": "Total owners",
                    "games": "Games",
                },
            )
            fig.update_layout(coloraxis_colorbar_title="Games")
            st.plotly_chart(fig, use_container_width=True)

        with right:
            fig = px.treemap(
                top_publishers,
                path=["publisher_main"],
                values="positive_reviews",
                color="games",
                title="Publisher share by positive reviews",
                labels={"positive_reviews": "Positive reviews", "games": "Games"},
            )
            st.plotly_chart(fig, use_container_width=True)

        scatter_df = filtered_df.nlargest(min(350, len(filtered_df)), "review_total").copy()
        fig = px.scatter(
            scatter_df,
            x="review_total",
            y="owners_midpoint",
            size="current_players",
            color="review_positive_ratio",
            hover_name="name",
            log_x=True,
            log_y=True,
            title="Reviews, owners, and current players",
            labels={
                "review_total": "Total reviews",
                "owners_midpoint": "Estimated owners",
                "current_players": "Current players",
                "review_positive_ratio": "Positive ratio",
            },
        )
        fig.update_layout(
            coloraxis_colorbar_title="Positive ratio",
            annotations=[
                dict(
                    text="Bubble size represents current players.",
                    x=0,
                    y=-0.18,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    align="left",
                    font=dict(size=12, color="#5b6472"),
                )
            ],
            margin=dict(b=80),
        )
        st.plotly_chart(fig, use_container_width=True)

    with catalog_tab:
        left, right = st.columns(2)

        genre_filtered = get_genre_summary(filtered_df).head(15)
        with left:
            fig = px.bar(
                genre_filtered.sort_values("games"),
                x="games",
                y="genre",
                orientation="h",
                title="Most frequent genres",
                labels={"games": "Games", "genre": "Genre", "median_owners": "Median owners"},
            )
            fig.update_traces(marker_color="#4c78a8")
            st.plotly_chart(fig, use_container_width=True)

        platform_counts = pd.DataFrame(
            {
                "platform": ["Windows", "Mac", "Linux"],
                "games": [
                    int(filtered_df["platform_windows"].sum()),
                    int(filtered_df["platform_mac"].sum()),
                    int(filtered_df["platform_linux"].sum()),
                ],
            }
        )
        with right:
            fig = px.pie(
                platform_counts,
                names="platform",
                values="games",
                hole=0.42,
                title="Platform compatibility",
            )
            st.plotly_chart(fig, use_container_width=True)

        category_cols = [col for col in filtered_df.columns if col.startswith("category_")]
        category_counts = (
            filtered_df[category_cols]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
            .rename(columns={"index": "category", 0: "games"})
        )
        category_counts["category"] = category_counts["category"].str.replace("category_", "", regex=False)
        fig = px.bar(
            category_counts.sort_values("games"),
            x="games",
            y="category",
            orientation="h",
            title="Most common feature categories",
            labels={"games": "Games", "category": "Category"},
        )
        fig.update_traces(marker_color="#72b7b2")
        st.plotly_chart(fig, use_container_width=True)

    with reviews_tab:
        left, right = st.columns(2)

        review_bins_df = filtered_df.copy()
        review_bins_df["review_volume"] = pd.cut(
            review_bins_df["review_total"],
            bins=[-1, 100, 1_000, 10_000, 100_000, np.inf],
            labels=["0-100", "101-1K", "1K-10K", "10K-100K", "100K+"],
        )
        review_volume_summary = (
            review_bins_df.groupby("review_volume", observed=True)
            .agg(
                games=("name", "count"),
                median_positive_ratio=("review_positive_ratio", "median"),
                median_owners=("owners_midpoint", "median"),
            )
            .reset_index()
        )

        with left:
            fig = px.bar(
                review_volume_summary,
                x="review_volume",
                y="games",
                color="median_positive_ratio",
                color_continuous_scale="Viridis",
                title="Games by review volume bucket",
                labels={
                    "review_volume": "Review volume",
                    "games": "Games",
                    "median_positive_ratio": "Median positive ratio",
                },
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            fig = px.violin(
                review_bins_df,
                x="review_volume",
                y="review_positive_ratio",
                color="review_volume",
                box=True,
                points=False,
                title="Positive review ratio by review volume",
                labels={"review_volume": "Review volume", "review_positive_ratio": "Positive ratio"},
            )
            st.plotly_chart(fig, use_container_width=True)

        review_mix = filtered_df.nlargest(min(25, len(filtered_df)), "review_total").copy()
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                y=review_mix["name"],
                x=review_mix["review_total_positive"],
                orientation="h",
                name="Positive",
                marker_color="#2ca02c",
            )
        )
        fig.add_trace(
            go.Bar(
                y=review_mix["name"],
                x=review_mix["review_total_negative"],
                orientation="h",
                name="Negative",
                marker_color="#d62728",
            )
        )
        fig.update_layout(
            barmode="stack",
            title="Positive vs negative reviews for the most reviewed games",
            xaxis_title="Reviews",
            yaxis_title="Game",
            height=760,
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with studios_tab:
        left, right = st.columns(2)

        developer_summary = (
            filtered_df.groupby("developer_main", as_index=False)
            .agg(
                games=("name", "count"),
                total_owners=("owners_midpoint", "sum"),
                median_positive_ratio=("review_positive_ratio", "median"),
                total_positive_reviews=("review_total_positive", "sum"),
            )
            .query("games >= 2")
            .sort_values("total_owners", ascending=False)
            .head(20)
        )

        publisher_quality = (
            filtered_df.groupby("publisher_main", as_index=False)
            .agg(
                games=("name", "count"),
                median_owners=("owners_midpoint", "median"),
                total_positive_reviews=("review_total_positive", "sum"),
                median_positive_ratio=("review_positive_ratio", "median"),
            )
            .query("games >= 2")
            .sort_values("total_positive_reviews", ascending=False)
            .head(25)
        )

        with left:
            fig = px.scatter(
                developer_summary,
                x="games",
                y="total_owners",
                size="total_positive_reviews",
                color="median_positive_ratio",
                hover_name="developer_main",
                log_y=True,
                color_continuous_scale="Plasma",
                title="Developer portfolio size vs total owners",
                labels={
                    "games": "Games",
                    "total_owners": "Total estimated owners",
                    "total_positive_reviews": "Positive reviews",
                    "median_positive_ratio": "Median positive ratio",
                },
            )
            fig.update_layout(
                coloraxis_colorbar_title="Median positive ratio",
                annotations=[
                    dict(
                        text="Bubble size represents total positive reviews.",
                        x=0,
                        y=-0.2,
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        align="left",
                        font=dict(size=12, color="#5b6472"),
                    )
                ],
                margin=dict(b=85),
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            fig = px.bar(
                publisher_quality.sort_values("total_positive_reviews"),
                x="total_positive_reviews",
                y="publisher_main",
                orientation="h",
                color="median_positive_ratio",
                color_continuous_scale="Cividis",
                title="Top publishers by positive reviews",
                labels={
                    "publisher_main": "Publisher",
                    "total_positive_reviews": "Positive reviews",
                    "median_positive_ratio": "Median positive ratio",
                },
            )
            fig.update_layout(coloraxis_colorbar_title="Median positive ratio")
            st.plotly_chart(fig, use_container_width=True)

        top_developers_table = developer_summary[
            ["developer_main", "games", "total_owners", "total_positive_reviews", "median_positive_ratio"]
        ].rename(
            columns={
                "developer_main": "Developer",
                "games": "Games",
                "total_owners": "Total estimated owners",
                "total_positive_reviews": "Positive reviews",
                "median_positive_ratio": "Median positive ratio",
            }
        )
        st.dataframe(top_developers_table, use_container_width=True, hide_index=True)

    with time_tab:
        left, right = st.columns(2)

        by_year_quality = (
            filtered_df.groupby("release_year", as_index=False)
            .agg(
                games=("name", "count"),
                total_owners=("owners_midpoint", "sum"),
                median_reviews=("review_total", "median"),
                median_positive_ratio=("review_positive_ratio", "median"),
                median_price=("price_final", "median"),
            )
            .sort_values("release_year")
        )

        with left:
            fig = px.line(
                by_year_quality,
                x="release_year",
                y=["median_reviews", "median_price"],
                markers=True,
                title="Median reviews and price over time",
                labels={"release_year": "Release year", "value": "Median value", "variable": "Metric"},
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            fig = px.area(
                by_year_quality,
                x="release_year",
                y="total_owners",
                color_discrete_sequence=["#54a24b"],
                title="Estimated owners accumulated by release year",
                labels={"release_year": "Release year", "total_owners": "Total estimated owners"},
            )
            st.plotly_chart(fig, use_container_width=True)

        monthly = (
            filtered_df.groupby("release_month", as_index=False)
            .agg(
                games=("name", "count"),
                median_owners=("owners_midpoint", "median"),
                median_positive_ratio=("review_positive_ratio", "median"),
            )
            .sort_values("release_month")
        )
        monthly["release_month"] = monthly["release_month"].astype(int)
        fig = px.bar(
            monthly,
            x="release_month",
            y="games",
            color="median_owners",
            color_continuous_scale="Sunset",
            title="Seasonality: releases by month",
            labels={"release_month": "Release month", "games": "Games", "median_owners": "Median owners"},
        )
        fig.add_trace(
            go.Scatter(
                x=monthly["release_month"],
                y=monthly["median_positive_ratio"] * monthly["games"].max(),
                mode="lines+markers",
                name="Positive ratio scaled",
                line=dict(color="#111827", width=2),
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    with quality_tab:
        left, right = st.columns(2)

        with left:
            fig = px.box(
                filtered_df,
                x="price_category",
                y="review_positive_ratio",
                points="outliers",
                title="Positive ratio by price category",
                labels={"price_category": "Price category", "review_positive_ratio": "Positive ratio"},
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            fig = px.scatter(
                filtered_df,
                x="achievement_avg_percent",
                y="review_positive_ratio",
                size="review_total",
                color="price_category",
                hover_name="name",
                title="Achievement completion vs satisfaction",
                labels={
                    "achievement_avg_percent": "Average achievement unlock %",
                    "review_positive_ratio": "Positive ratio",
                    "review_total": "Reviews",
                    "price_category": "Price",
                },
            )
            fig.update_layout(
                legend_title_text="Price category",
                annotations=[
                    dict(
                        text="Bubble size represents total reviews.",
                        x=0,
                        y=-0.18,
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        align="left",
                        font=dict(size=12, color="#5b6472"),
                    )
                ],
                margin=dict(b=80),
            )
            st.plotly_chart(fig, use_container_width=True)

        corr_cols = [
            "owners_midpoint",
            "price_final",
            "review_total",
            "review_positive_ratio",
            "current_players",
            "steam_recommendations_total",
            "publisher_past_games_count",
            "developer_past_games_count",
        ]
        corr = filtered_df[corr_cols].corr(numeric_only=True)
        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Correlations between key variables",
        )
        st.plotly_chart(fig, use_container_width=True)

        waterfall_values = {
            "All games": len(filtered_df),
            ">= 1K reviews": int((filtered_df["review_total"] >= 1_000).sum()),
            ">= 80% positive": int((filtered_df["review_positive_ratio"] >= 0.80).sum()),
            ">= 1M owners": int((filtered_df["owners_midpoint"] >= 1_000_000).sum()),
        }
        waterfall_steps = list(waterfall_values.keys())
        waterfall_counts = list(waterfall_values.values())
        waterfall_deltas = [waterfall_counts[0]] + [
            waterfall_counts[i] - waterfall_counts[i - 1] for i in range(1, len(waterfall_counts))
        ]
        fig = go.Figure(
            go.Waterfall(
                x=waterfall_steps,
                y=waterfall_deltas,
                measure=["absolute", "relative", "relative", "relative"],
                text=waterfall_counts,
                textposition="outside",
                connector={"line": {"color": "#6b7280"}},
                increasing={"marker": {"color": "#4c78a8"}},
                decreasing={"marker": {"color": "#e45756"}},
            )
        )
        fig.update_layout(
            title="Quality and scale funnel",
            yaxis_title="Games",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tables_tab:
        left, right = st.columns(2)

        with left:
            top_games = filtered_df.nlargest(20, "owners_midpoint")[
                [
                    "name",
                    "owners_midpoint",
                    "review_total",
                    "review_positive_ratio",
                    "publisher_main",
                    "developer_main",
                    "release_year",
                ]
            ].rename(
                columns={
                    "name": "Game",
                    "owners_midpoint": "Estimated owners",
                    "review_total": "Reviews",
                    "review_positive_ratio": "Positive ratio",
                    "publisher_main": "Publisher",
                    "developer_main": "Developer",
                    "release_year": "Release year",
                }
            )
            st.subheader("Top games by estimated owners")
            st.dataframe(top_games, use_container_width=True, hide_index=True)

        with right:
            underrated = (
                filtered_df[filtered_df["review_total"] >= 1_000]
                .sort_values(["review_positive_ratio", "owners_midpoint"], ascending=[False, True])
                .head(20)
            )[
                [
                    "name",
                    "owners_midpoint",
                    "review_total",
                    "review_positive_ratio",
                    "price_final",
                    "publisher_main",
                ]
            ].rename(
                columns={
                    "name": "Game",
                    "owners_midpoint": "Estimated owners",
                    "review_total": "Reviews",
                    "review_positive_ratio": "Positive ratio",
                    "price_final": "Final price",
                    "publisher_main": "Publisher",
                }
            )
            st.subheader("High-rated lower-reach games")
            st.dataframe(underrated, use_container_width=True, hide_index=True)

        st.subheader("Publisher summary")
        publisher_table = (
            filtered_df.groupby("publisher_main", as_index=False)
            .agg(
                games=("name", "count"),
                total_owners=("owners_midpoint", "sum"),
                median_owners=("owners_midpoint", "median"),
                total_reviews=("review_total", "sum"),
                median_positive_ratio=("review_positive_ratio", "median"),
            )
            .sort_values("total_owners", ascending=False)
            .head(30)
            .rename(
                columns={
                    "publisher_main": "Publisher",
                    "games": "Games",
                    "total_owners": "Total estimated owners",
                    "median_owners": "Median owners",
                    "total_reviews": "Total reviews",
                    "median_positive_ratio": "Median positive ratio",
                }
            )
        )
        st.dataframe(publisher_table, use_container_width=True, hide_index=True)


def render_pca() -> None:
    section_header(SECTIONS[2])

    df = load_dataset()
    pca_df, explained_df = load_pca_outputs()

    explained_lookup = explained_df.set_index("component")["explained_variance_ratio"].to_dict()
    detail_cols = [
        "owners_midpoint",
        "review_total",
        "review_positive_ratio",
        "price_final",
        "publisher_main",
        "developer_main",
        "release_year",
    ]
    merged_pca = pca_df.reset_index(drop=True).copy()
    merged_pca["_row_id"] = np.arange(len(merged_pca))
    for col in detail_cols:
        if col in df.columns:
            merged_pca[col] = df.reset_index(drop=True)[col]

    st.info(
        "Interaction guide: hover to inspect points, use the mouse wheel to zoom, drag with the right mouse button to pan, "
        "and in the 3D view drag with the left mouse button to rotate. Use box/lasso selection from the Plotly toolbar to select points. "
        "If your browser does not trigger point selection, use the selector below the charts."
    )

    metric_cols = st.columns(2)
    with metric_cols[0]:
        st.metric(
            "2D explained variance",
            "15.6%",
            help="PC1 + PC2 explain approximately 0.15555 of the total variance.",
        )
    with metric_cols[1]:
        st.metric(
            "3D explained variance",
            "21.4%",
            help="PC1 + PC2 + PC3 explain approximately 0.214030 of the total variance.",
        )

    st.caption(
        "These percentages are cumulative explained variance: the 2D map uses PC1 and PC2, while the 3D map uses PC1, PC2, and PC3. "
        "They are useful for visualization, but they do not represent the full information contained in the dataset."
    )

    color_metric = st.selectbox(
        "Color points by",
        options=[
            "log_owners_midpoint",
            "owners_midpoint_literal",
            "review_total",
            "review_positive_ratio",
            "price_final",
            "release_year",
        ],
        format_func=lambda value: {
            "log_owners_midpoint": "Log owners",
            "owners_midpoint_literal": "Estimated owners",
            "review_total": "Total reviews",
            "review_positive_ratio": "Positive review ratio",
            "price_final": "Final price",
            "release_year": "Release year",
        }[value],
    )

    tab_2d, tab_3d = st.tabs(["PCA 2D", "PCA 3D"])
    with tab_2d:
        fig = px.scatter(
            merged_pca,
            x="PC1",
            y="PC2",
            color=color_metric,
            hover_name="name",
            custom_data=["_row_id", "name"],
            color_continuous_scale="Viridis",
            title="PCA 2D projection",
            labels={
                "PC1": f"PC1 ({explained_lookup.get('PC1', 0) * 100:.1f}% variance)",
                "PC2": f"PC2 ({explained_lookup.get('PC2', 0) * 100:.1f}% variance)",
                "log_owners_midpoint": "Log owners",
                "owners_midpoint_literal": "Estimated owners",
                "review_total": "Total reviews",
                "review_positive_ratio": "Positive ratio",
                "price_final": "Final price",
                "release_year": "Release year",
            },
        )
        fig.update_traces(marker=dict(size=9, opacity=0.78, line=dict(width=0.4, color="white")))
        fig.update_layout(
            height=720,
            dragmode="select",
            coloraxis_colorbar_title="Selected metric",
            margin=dict(l=10, r=10, t=70, b=10),
        )
        selection = st.plotly_chart(
            fig,
            use_container_width=True,
            key="pca_2d_chart",
            on_select="rerun",
            selection_mode=("points", "box", "lasso"),
            config={"scrollZoom": True, "displaylogo": False},
        )
        selected_id = get_selected_point_id(selection)
        if selected_id is not None:
            st.session_state["selected_pca_game_id"] = selected_id

    with tab_3d:
        fig = px.scatter_3d(
            merged_pca,
            x="PC1",
            y="PC2",
            z="PC3",
            color=color_metric,
            hover_name="name",
            custom_data=["_row_id", "name"],
            color_continuous_scale="Viridis",
            title="PCA 3D projection",
            labels={
                "PC1": f"PC1 ({explained_lookup.get('PC1', 0) * 100:.1f}% variance)",
                "PC2": f"PC2 ({explained_lookup.get('PC2', 0) * 100:.1f}% variance)",
                "PC3": f"PC3 ({explained_lookup.get('PC3', 0) * 100:.1f}% variance)",
                "log_owners_midpoint": "Log owners",
                "owners_midpoint_literal": "Estimated owners",
                "review_total": "Total reviews",
                "review_positive_ratio": "Positive ratio",
                "price_final": "Final price",
                "release_year": "Release year",
            },
        )
        fig.update_traces(marker=dict(size=4, opacity=0.74))
        fig.update_layout(
            height=760,
            coloraxis_colorbar_title="Selected metric",
            margin=dict(l=0, r=0, t=70, b=0),
            scene=dict(
                xaxis_title=f"PC1 ({explained_lookup.get('PC1', 0) * 100:.1f}%)",
                yaxis_title=f"PC2 ({explained_lookup.get('PC2', 0) * 100:.1f}%)",
                zaxis_title=f"PC3 ({explained_lookup.get('PC3', 0) * 100:.1f}%)",
            ),
        )
        selection = st.plotly_chart(
            fig,
            use_container_width=True,
            key="pca_3d_chart",
            on_select="rerun",
            selection_mode=("points",),
            config={"scrollZoom": True, "displaylogo": False},
        )
        selected_id = get_selected_point_id(selection)
        if selected_id is not None:
            st.session_state["selected_pca_game_id"] = selected_id

    st.divider()
    selected_game_id = st.session_state.get("selected_pca_game_id")
    if selected_game_id is None:
        st.info("Select a point in either PCA chart to open the game detail panel.")
    else:
        render_game_detail(selected_game_id, df, pca_df)


def render_modelo() -> None:
    section_header(SECTIONS[3])

    st.caption(
        "Enter one or more Steam game names. The app will resolve each name to a Steam appid, collect Steam and SteamSpy data, "
        "apply the preprocessing pipeline, run the saved model, and compare the prediction with the SteamSpy owner range."
    )

    if "applied_model_results" not in st.session_state:
        st.session_state["applied_model_results"] = pd.DataFrame()
    if "applied_model_found_games" not in st.session_state:
        st.session_state["applied_model_found_games"] = []

    left, right = st.columns([1, 1])
    with left:
        game_names_raw = st.text_area(
            "Game names",
            placeholder="Baldur's Gate 3\nSlay the Spire\nCyberpunk 2077",
            height=160,
            help="Write one game per line. You can add as many as you want.",
        )
    with right:
        force_refresh = st.toggle(
            "Force API refresh",
            value=False,
            help="If disabled, cached raw API responses are reused when available.",
        )
        st.write("**Pipeline steps**")
        st.write("1. Resolve Steam appid")
        st.write("2. Collect Steam and SteamSpy data")
        st.write("3. Apply post-release preprocessing")
        st.write("4. Run the saved model")
        st.write("5. Compare against SteamSpy owner range")

    game_names = [line.strip() for line in game_names_raw.splitlines() if line.strip()]

    col_a, col_b = st.columns([1, 4])
    with col_a:
        start = st.button("Start inference", type="primary", disabled=not game_names)
    with col_b:
        if game_names:
            st.caption(f"{len(game_names)} game(s) queued.")
        else:
            st.caption("No games queued yet.")

    if start:
        results = []
        found_games: list[str] = []
        progress = st.progress(0, text="Preparing inference...")
        found_placeholder = st.empty()

        with st.status("Running applied model pipeline...", expanded=True) as status:
            for idx, game_name in enumerate(game_names, start=1):
                progress.progress((idx - 1) / len(game_names), text=f"Inferring {game_name}...")
                status.write(f"Resolving and inferring: **{game_name}**")

                try:
                    summary = run_prediction(game_name, force=force_refresh)
                    summary = summary.copy()
                    summary.insert(0, "input_name", game_name)
                    results.append(summary)

                    resolved_name = str(summary.iloc[0]["name"])
                    appid = int(summary.iloc[0]["appid"])
                    found_games.append(f"{resolved_name} ({appid})")
                    found_placeholder.success("Found so far: " + ", ".join(found_games))
                    status.write(f"Completed: **{resolved_name}** (`appid={appid}`)")
                except Exception as exc:
                    error_row = pd.DataFrame(
                        [
                            {
                                "input_name": game_name,
                                "appid": np.nan,
                                "name": None,
                                "model_path": None,
                                "predicted_log_owners": np.nan,
                                "predicted_owners": np.nan,
                                "steamspy_owners_range": None,
                                "steamspy_owners_midpoint": np.nan,
                                "steamspy_log_owners_midpoint": np.nan,
                                "absolute_error_owners": np.nan,
                                "prediction_to_steamspy_ratio": np.nan,
                                "error": str(exc),
                            }
                        ]
                    )
                    results.append(error_row)
                    status.write(f"Failed: **{game_name}** - {exc}")

                progress.progress(idx / len(game_names), text=f"Processed {idx}/{len(game_names)} game(s).")

            status.update(label="Applied model inference finished.", state="complete")

        st.session_state["applied_model_results"] = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
        st.session_state["applied_model_found_games"] = found_games

    results_df = st.session_state.get("applied_model_results", pd.DataFrame())

    if not results_df.empty:
        st.divider()
        st.subheader("Inference results")

        display_df = results_df.copy()
        for col in ["predicted_owners", "steamspy_owners_midpoint", "absolute_error_owners"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].map(lambda value: f"{value:,.0f}" if pd.notna(value) else None)
        if "prediction_to_steamspy_ratio" in display_df.columns:
            display_df["prediction_to_steamspy_ratio"] = display_df["prediction_to_steamspy_ratio"].map(
                lambda value: f"{value:.2f}x" if pd.notna(value) else None
            )

        visible_cols = [
            "input_name",
            "appid",
            "name",
            "predicted_owners",
            "steamspy_owners_range",
            "absolute_error_owners",
            "prediction_to_steamspy_ratio",
            "error",
        ]
        visible_cols = [col for col in visible_cols if col in display_df.columns]
        display_df = display_df[visible_cols].rename(
            columns={
                "input_name": "Input name",
                "appid": "App ID",
                "name": "Resolved game",
                "predicted_owners": "Predicted owners",
                "steamspy_owners_range": "SteamSpy owner range",
                "absolute_error_owners": "Absolute error vs midpoint",
                "prediction_to_steamspy_ratio": "Prediction / midpoint",
                "error": "Error",
            }
        )
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        successful = results_df[results_df["predicted_owners"].notna()].copy()
        if not successful.empty:
            fig = px.bar(
                successful,
                x="name",
                y="predicted_owners",
                color="prediction_to_steamspy_ratio",
                color_continuous_scale="Viridis",
                title="Predicted owners by game",
                labels={
                    "name": "Game",
                    "predicted_owners": "Predicted owners",
                    "prediction_to_steamspy_ratio": "Prediction / SteamSpy midpoint",
                },
            )
            fig.update_layout(xaxis_tickangle=-30, coloraxis_colorbar_title="Ratio")
            st.plotly_chart(fig, use_container_width=True)

            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=successful["name"],
                    y=successful["predicted_owners"],
                    name="Predicted owners",
                    marker_color="#4c78a8",
                )
            )
            fig.add_trace(
                go.Bar(
                    x=successful["name"],
                    y=successful["steamspy_owners_midpoint"],
                    name="SteamSpy midpoint",
                    marker_color="#f58518",
                )
            )
            fig.update_layout(
                barmode="group",
                title="Prediction vs SteamSpy midpoint",
                xaxis_title="Game",
                yaxis_title="Owners",
                xaxis_tickangle=-30,
            )
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Raw prediction output", expanded=False):
            st.dataframe(results_df, use_container_width=True, hide_index=True)


def render_price_optimizer_workspace() -> None:
    df = load_dataset()

    try:
        model_path, _ = load_owner_model()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    st.caption(
        "This conceptual simulator uses the saved model as a demand anchor, then applies a deterministic synthetic price-response curve "
        "so the optimum stays close to the game's original price while still varying by title."
    )

    searchable_df = df.sort_values(["name", "release_year", "publisher_main"]).reset_index(drop=True)
    options = searchable_df.index.tolist()

    def game_label(idx: int) -> str:
        row = searchable_df.loc[idx]
        year = int(row["release_year"]) if pd.notna(row.get("release_year")) else "Unknown year"
        publisher = row.get("publisher_main", "Unknown publisher")
        return f"{row['name']} ({year}) - {publisher}"

    selected_idx = st.selectbox("Game", options=options, format_func=game_label)
    selected_row = searchable_df.loc[selected_idx]

    col_a, col_b = st.columns([1, 1.4])
    with col_a:
        include_default_grid = st.checkbox("Include default price grid", value=True)
    with col_b:
        custom_prices_raw = st.text_input(
            "Custom prices",
            placeholder="Example: 4.99, 14.99, 24.99",
            help="Separate prices with commas, spaces, or semicolons.",
        )

    with st.expander("Selected game baseline", expanded=False):
        baseline_cols = [
            col
            for col in [
                "name",
                "publisher_main",
                "developer_main",
                "release_year",
                "price_final",
                "price_initial",
                "discount_percent",
                "owners_midpoint",
                "review_total",
                "review_positive_ratio",
            ]
            if col in searchable_df.columns
        ]
        st.dataframe(selected_row[baseline_cols].to_frame("value"), use_container_width=True)

    if st.button("Run price optimization", type="primary", use_container_width=True):
        try:
            prices = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0] if include_default_grid else []
            prices.extend(parse_price_list(custom_prices_raw) if custom_prices_raw.strip() else [])
            prices = sorted(set(round(price, 2) for price in prices))
            if not prices:
                st.warning("Add at least one price or keep the default price grid enabled.")
                return

            with st.spinner(f"Testing {len(prices)} price scenarios for {selected_row['name']}..."):
                results = build_price_optimization_results(selected_row, prices, df)
        except ValueError as exc:
            st.error(f"Price input error: {exc}")
            return
        except Exception as exc:
            st.error(f"The optimization could not be completed: {exc}")
            return

        best = results.loc[results["expected_revenue"].idxmax()]
        st.success(f"Best simulated price for {selected_row['name']}: {format_money(best['price'])}")

        metric_a, metric_b, metric_c = st.columns(3)
        metric_a.metric("Optimal price", format_money(best["price"]))
        metric_b.metric("Simulated owners", format_count(best["predicted_owners"]))
        metric_c.metric("Expected revenue", format_money(best["expected_revenue"]))

        baseline_a, baseline_b = st.columns(2)
        baseline_a.metric("Original listed price", format_money(best["reference_price"]))
        baseline_b.metric("Synthetic curve center", format_money(best["synthetic_curve_center"]))

        display_results = results.rename(
            columns={
                "price": "Price",
                "price_category": "Price category",
                "predicted_log_owners": "Simulated log owners",
                "predicted_owners": "Simulated owners",
                "expected_revenue": "Expected revenue",
                "reference_price": "Original listed price",
                "real_owners_influence": "Real owners influence",
                "synthetic_curve_center": "Synthetic curve center",
            }
        ).copy()
        display_results["Price"] = display_results["Price"].map(format_money)
        display_results["Simulated owners"] = display_results["Simulated owners"].map(format_count)
        display_results["Expected revenue"] = display_results["Expected revenue"].map(format_money)
        display_results["Simulated log owners"] = display_results["Simulated log owners"].map(lambda value: f"{value:.3f}")
        display_results["Original listed price"] = display_results["Original listed price"].map(format_money)
        display_results["Real owners influence"] = display_results["Real owners influence"].map(lambda value: f"{value:+.2f}")
        display_results["Synthetic curve center"] = display_results["Synthetic curve center"].map(format_money)
        st.dataframe(display_results, use_container_width=True, hide_index=True)

        fig_revenue = px.line(
            results,
            x="price",
            y="expected_revenue",
            markers=True,
            title="Expected revenue by simulated price",
            labels={"price": "Price", "expected_revenue": "Expected revenue"},
            color_discrete_sequence=["#2f6f73"],
        )
        fig_revenue.add_vline(
            x=float(best["price"]),
            line_dash="dash",
            line_color="#f58518",
            annotation_text="Best price",
            annotation_position="top right",
        )
        fig_revenue.update_layout(yaxis_tickprefix="$", xaxis_tickprefix="$")
        st.plotly_chart(fig_revenue, use_container_width=True)

        fig_owners = px.bar(
            results,
            x="price",
            y="predicted_owners",
            color="price_category",
            title="Simulated owners by price",
            labels={
                "price": "Price",
                "predicted_owners": "Simulated owners",
                "price_category": "Price category",
            },
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig_owners.update_layout(xaxis_tickprefix="$", legend_title_text="Price category")
        st.plotly_chart(fig_owners, use_container_width=True)

        st.info(
            "Interpretation note: these are synthetic scenario results for presentation purposes. The model provides the baseline demand scale, "
            "the original listed price anchors the curve, and real owner counts provide a light adjustment. This is not a causal estimate."
        )

        st.caption(f"Model used: {model_path.relative_to(PROJECT_ROOT)}")


def render_publisher_investment_workspace() -> None:
    try:
        investment_df = build_publisher_investment_table()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"The investment table could not be prepared: {exc}")
        return

    st.caption(
        "This conceptual allocator samples a varied set of 15 candidate games, assigns investment costs by studio/franchise strength, "
        "and selects the portfolio with the highest expected publisher profit under the chosen budget."
    )

    col_a, col_b = st.columns([1, 1])
    with col_a:
        budget = st.number_input(
            "Investment budget",
            min_value=50_000,
            max_value=80_000_000,
            value=8_000_000,
            step=250_000,
            format="%d",
            help="Budget available for acquiring publishing/investment rights across the sampled opportunities.",
        )
    with col_b:
        sample_seed = st.number_input(
            "Deal flow seed",
            min_value=1,
            max_value=9999,
            value=42,
            step=1,
            help="Change this value to generate another reproducible set of 15 candidate games.",
        )

    candidates = sample_varied_investment_candidates(investment_df, int(sample_seed), n=15)

    st.subheader("Available candidate games")
    st.caption(
        "These are the 15 opportunities currently available in this simulated deal flow. Prediction-based return metrics are hidden "
        "until the optimization is run."
    )
    candidate_preview = candidates.rename(
        columns={
            "name": "Game",
            "developer_main": "Developer",
            "publisher_main": "Current publisher",
            "release_year": "Year",
            "investment_tier": "Investment tier",
            "fame_score": "Studio/IP score",
            "investment_cost": "Investment cost",
            "real_price": "Real price",
            "review_positive_ratio": "Positive review ratio",
        }
    ).copy()
    for col in ["Investment cost", "Real price"]:
        if col in candidate_preview.columns:
            candidate_preview[col] = candidate_preview[col].map(format_money)
    if "Studio/IP score" in candidate_preview.columns:
        candidate_preview["Studio/IP score"] = candidate_preview["Studio/IP score"].map(lambda value: f"{value:.2f}")
    if "Positive review ratio" in candidate_preview.columns:
        candidate_preview["Positive review ratio"] = candidate_preview["Positive review ratio"].map(lambda value: f"{value:.1%}")

    preview_cols = [
        "Game",
        "Developer",
        "Current publisher",
        "Year",
        "Investment tier",
        "Studio/IP score",
        "Investment cost",
        "Real price",
        "Positive review ratio",
    ]
    preview_cols = [col for col in preview_cols if col in candidate_preview.columns]
    st.dataframe(candidate_preview[preview_cols], use_container_width=True, hide_index=True)

    run_optimization = st.button("Run investment optimization", type="primary", use_container_width=True)
    if not run_optimization:
        st.info("Choose a budget and run the optimizer to reveal predicted returns, selected games, and portfolio charts.")
        with st.expander("How investment costs are assigned", expanded=False):
            st.write(
                "Investment costs are generated in memory from six tiers: Micro Indie, Indie, Rising Studio, Established Studio, "
                "Premium IP, and AAA / Top Franchise. The tier score is mostly driven by developer track record, publisher track record, "
                "real owner counts, review volume, and simple franchise markers in the game title. No source dataset is modified."
            )
        return

    selected = optimize_investment_portfolio(candidates, float(budget))

    total_cost = selected["investment_cost"].sum() if not selected.empty else 0
    total_profit = selected["expected_profit"].sum() if not selected.empty else 0
    portfolio_roi = total_profit / total_cost if total_cost > 0 else 0

    st.divider()
    st.subheader("Optimization results")

    metric_a, metric_b, metric_c, metric_d = st.columns(4)
    metric_a.metric("Selected games", f"{len(selected)} / {len(candidates)}")
    metric_b.metric("Budget used", format_money(total_cost), delta=f"{total_cost / budget:.1%}" if budget else None)
    metric_c.metric("Expected profit", format_money(total_profit))
    metric_d.metric("Expected ROI", f"{portfolio_roi:.1%}")
    st.caption("Expected ROI means expected net profit divided by investment cost.")

    if selected.empty:
        st.warning(
            "No positive-profit combination fits this budget. Increase the budget or change the deal flow seed to explore another sample."
        )

    candidates_display = candidates.copy()
    candidates_display["Selected"] = candidates_display["name"].isin(selected["name"]) if not selected.empty else False
    candidates_display = candidates_display.sort_values(["Selected", "expected_profit"], ascending=[False, False])

    table = candidates_display.rename(
        columns={
            "name": "Game",
            "developer_main": "Developer",
            "publisher_main": "Current publisher",
            "release_year": "Year",
            "investment_tier": "Investment tier",
            "fame_score": "Studio/IP score",
            "investment_cost": "Investment cost",
            "real_price": "Real price",
            "predicted_owners": "Predicted owners",
            "gross_revenue": "Gross revenue",
            "publisher_share": "Publisher share",
            "expected_profit": "Expected profit",
            "roi": "Expected ROI",
            "review_positive_ratio": "Positive review ratio",
        }
    )

    formatted = table.copy()
    for col in ["Investment cost", "Real price", "Gross revenue", "Publisher share", "Expected profit"]:
        if col in formatted.columns:
            formatted[col] = formatted[col].map(format_money)
    if "Predicted owners" in formatted.columns:
        formatted["Predicted owners"] = formatted["Predicted owners"].map(format_count)
    if "Studio/IP score" in formatted.columns:
        formatted["Studio/IP score"] = formatted["Studio/IP score"].map(lambda value: f"{value:.2f}")
    if "Expected ROI" in formatted.columns:
        formatted["Expected ROI"] = formatted["Expected ROI"].map(lambda value: f"{value:.1%}")
    if "Positive review ratio" in formatted.columns:
        formatted["Positive review ratio"] = formatted["Positive review ratio"].map(lambda value: f"{value:.1%}")

    visible_cols = [
        "Selected",
        "Game",
        "Developer",
        "Current publisher",
        "Year",
        "Investment tier",
        "Studio/IP score",
        "Investment cost",
        "Real price",
        "Predicted owners",
        "Publisher share",
        "Expected profit",
        "Expected ROI",
    ]
    visible_cols = [col for col in visible_cols if col in formatted.columns]
    st.dataframe(formatted[visible_cols], use_container_width=True, hide_index=True)

    chart_df = candidates_display.copy()
    chart_df["Selection"] = np.where(chart_df["Selected"], "Selected", "Not selected")
    fig = px.scatter(
        chart_df,
        x="investment_cost",
        y="expected_profit",
        size="predicted_owners",
        color="Selection",
        hover_name="name",
        hover_data={
            "developer_main": True,
            "investment_tier": True,
            "roi": ":.1%",
            "real_price": ":$.2f",
            "investment_cost": ":$,.0f",
            "expected_profit": ":$,.0f",
            "predicted_owners": ":,.0f",
            "Selection": False,
        },
        title="Candidate opportunities: investment cost vs expected profit",
        labels={
            "investment_cost": "Investment cost",
            "expected_profit": "Expected profit",
            "predicted_owners": "Predicted owners",
        },
        color_discrete_map={"Selected": "#2f6f73", "Not selected": "#b8bec8"},
    )
    fig.update_layout(xaxis_tickprefix="$", yaxis_tickprefix="$", legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

    tier_summary = (
        candidates_display.groupby(["investment_tier", "Selected"], observed=False)
        .agg(games=("name", "count"), total_cost=("investment_cost", "sum"), expected_profit=("expected_profit", "sum"))
        .reset_index()
    )
    fig_tiers = px.bar(
        tier_summary,
        x="investment_tier",
        y="games",
        color="Selected",
        title="Sample diversity by investment tier",
        labels={"investment_tier": "Investment tier", "games": "Games", "Selected": "Portfolio"},
        color_discrete_map={True: "#2f6f73", False: "#d4d8df"},
    )
    fig_tiers.update_layout(xaxis_tickangle=-20)
    st.plotly_chart(fig_tiers, use_container_width=True)

    with st.expander("How investment costs are assigned", expanded=False):
        st.write(
            "Investment costs are generated in memory from six tiers: Micro Indie, Indie, Rising Studio, Established Studio, "
            "Premium IP, and AAA / Top Franchise. The tier score is mostly driven by developer track record, publisher track record, "
            "real owner counts, review volume, and simple franchise markers in the game title. No source dataset is modified."
        )
        st.write(
            "The expected return assumes the publisher captures 20% of gross revenue. The optimizer then maximizes expected profit "
            "under the selected budget, using a knapsack-style allocation."
        )


def render_aplicaciones() -> None:
    section_header(SECTIONS[4])

    active_workspace = st.session_state.get("application_workspace")
    if active_workspace:
        workspace_titles = {
            "price_optimizer": "Price Optimizer",
            "investment_assistant": "Publisher Investment Assistant",
            "public_explorer": "Public Interest Explorer",
        }
        workspace_copy = {
            "price_optimizer": (
                "This workspace will simulate several price points for a selected game, run the prediction pipeline for each scenario, "
                "and estimate the revenue-maximizing price using predicted owners x price."
            ),
            "investment_assistant": (
                "This workspace will compare candidate games under a fixed budget, estimate expected upside, and recommend where a "
                "publisher or investor should allocate capital."
            ),
            "public_explorer": (
                "This workspace will provide a lightweight experience for players, journalists, and curious users who want to explore "
                "rankings, compare games, and understand what makes a title stand out in the Steam market."
            ),
        }

        if st.button("Back to Applications"):
            st.session_state.pop("application_workspace", None)
            st.rerun()

        st.subheader(workspace_titles[active_workspace])
        st.write(workspace_copy[active_workspace])
        if active_workspace == "price_optimizer":
            render_price_optimizer_workspace()
            return
        if active_workspace == "investment_assistant":
            render_publisher_investment_workspace()
            return

        placeholder(
            "Workspace coming next",
            "This area is reserved for the dedicated interactive workflow that will be built in the next step.",
        )
        return

    st.caption(
        "These application areas translate the model into practical product and business workflows. "
        "Each card will later open a dedicated workspace."
    )

    with st.container(border=True):
        st.subheader("Price Optimizer")
        st.write(
            "Simulate multiple price points for a game, estimate demand at each price, and identify the price "
            "that maximizes expected revenue using predicted owners x price."
        )
        if st.button("Open Price Optimizer", key="open_price_optimizer", use_container_width=True):
            st.session_state["application_workspace"] = "price_optimizer"
            st.rerun()

    with st.container(border=True):
        st.subheader("Publisher Investment Assistant")
        st.write(
            "Support portfolio decisions by comparing candidate games under a fixed budget. The tool will estimate "
            "expected upside and recommend where a publisher or investor should allocate capital."
        )
        if st.button("Open Investment Assistant", key="open_investment_assistant", use_container_width=True):
            st.session_state["application_workspace"] = "investment_assistant"
            st.rerun()

    with st.container(border=True):
        st.subheader("Public Interest Explorer")
        st.write(
            "A lightweight experience for players, journalists, and curious users who want to explore rankings, "
            "compare games, and understand what makes a title stand out in the Steam market."
        )
        st.link_button(
            "Open Public Explorer",
            "https://www.nintendo.com/es-es/",
            use_container_width=True,
        )


def render_chatbot() -> None:
    section_header(SECTIONS[5])

    if "chatbot_messages" not in st.session_state:
        st.session_state["chatbot_messages"] = [
            {
                "role": "assistant",
                "content": (
                    "Hi. Ask me about the Steam dataset: best-selling games, publishers, reviews, "
                    "genres, prices, release years, platforms, or any ranking you want to explore."
                ),
                "sql": None,
                "data": None,
                "assumptions": [],
            }
        ]

    with st.container(border=True):
        col_a, col_b, col_c = st.columns([1.2, 1, 1])
        with col_a:
            model = st.text_input("Model", value=os.getenv("OPENAI_MODEL", DEFAULT_CHAT_MODEL))
        with col_b:
            max_rows = st.slider("Max result rows", min_value=5, max_value=100, value=50, step=5)
        with col_c:
            show_debug = st.toggle("Show SQL and tables", value=True)

        client_available = get_openai_client() is not None

        if not client_available:
            st.warning(
                "OPENAI_API_KEY is not configured. Set it as an environment variable or in Streamlit secrets to enable the chatbot."
            )

    st.caption(
        "The assistant generates read-only SQL, the app validates it, DuckDB executes it on the parquet dataset, "
        "and the assistant summarizes the result."
    )

    for message in st.session_state["chatbot_messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if show_debug and message.get("sql"):
                with st.expander("Generated SQL", expanded=False):
                    st.code(message["sql"], language="sql")
            if show_debug and isinstance(message.get("data"), pd.DataFrame):
                with st.expander("Query result", expanded=False):
                    st.dataframe(message["data"], use_container_width=True, hide_index=True)
            if show_debug and message.get("assumptions"):
                with st.expander("Assumptions", expanded=False):
                    for assumption in message["assumptions"]:
                        st.write(f"- {assumption}")

    prompt = st.chat_input("Ask a question about the dataset...", disabled=not client_available)

    if prompt:
        user_message = {"role": "user", "content": prompt, "sql": None, "data": None, "assumptions": []}
        st.session_state["chatbot_messages"].append(user_message)

        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking and querying the dataset..."):
                try:
                    result = ask_dataset_chatbot(prompt, model=model, max_rows=max_rows)
                    assistant_message = {
                        "role": "assistant",
                        "content": result["answer"],
                        "sql": result["sql"],
                        "data": result["data"],
                        "assumptions": result["assumptions"],
                    }
                    st.write(result["answer"])

                    if show_debug and result.get("sql"):
                        with st.expander("Generated SQL", expanded=True):
                            st.code(result["sql"], language="sql")
                    if show_debug and isinstance(result.get("data"), pd.DataFrame):
                        with st.expander("Query result", expanded=True):
                            st.dataframe(result["data"], use_container_width=True, hide_index=True)
                    if show_debug and result.get("assumptions"):
                        with st.expander("Assumptions", expanded=False):
                            for assumption in result["assumptions"]:
                                st.write(f"- {assumption}")

                except Exception as exc:
                    assistant_message = {
                        "role": "assistant",
                        "content": f"Sorry, I could not answer that question. Error: {exc}",
                        "sql": None,
                        "data": None,
                        "assumptions": [],
                    }
                    st.error(assistant_message["content"])

        st.session_state["chatbot_messages"].append(assistant_message)

    if st.button("Clear chat history"):
        st.session_state.pop("chatbot_messages", None)
        st.rerun()


def main() -> None:
    inject_css()

    with st.sidebar:
        st.title("Steam Analytics")
        st.caption("Project navigation")

        selected_label = st.radio(
            "Sections",
            options=[section.label for section in SECTIONS],
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown(
            '<p class="section-note">Base app ready for connecting notebooks, models, charts, and the chatbot.</p>',
            unsafe_allow_html=True,
        )

    selected = next(section for section in SECTIONS if section.label == selected_label)

    if selected.key == "presentacion":
        render_presentacion()
    elif selected.key == "dashboard":
        render_dashboard()
    elif selected.key == "pca":
        render_pca()
    elif selected.key == "modelo":
        render_modelo()
    elif selected.key == "aplicaciones":
        render_aplicaciones()
    elif selected.key == "chatbot":
        render_chatbot()


if __name__ == "__main__":
    main()

