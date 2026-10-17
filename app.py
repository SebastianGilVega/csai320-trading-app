"""Quantitative Stock Trading Assistant - Streamlit dashboard backed by AWS S3 & Lex."""

import io
import os
import uuid
from datetime import datetime

import boto3
import numpy as np
import pandas as pd
import streamlit as st
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
from PIL import Image, UnidentifiedImageError

BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "csai320-final-project-sebastian-gil-vega")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
DATA_KEY_TEMPLATE = "data/processed/{ticker}_backtested_processed.csv"
PLOT_KEY_TEMPLATE = "plots/{ticker}_strategy_comparison.png"
AVAILABLE_TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "TSLA",
    "BRK-B",
    "LLY",
    "JPM",
    "V",
    "XOM",
    "UNH",
    "JNJ",
    "MA",
    "PG",
    "HD",
    "MRK",
    "ABBV",
    "CVX",
    "COST",
    "PEP",
    "ADBE",
    "WMT",
    "BAC",
    "NFLX",
    "MCD",
    "ACN",
    "CSCO",
    "TMO",
    "ABT",
    "DHR",
    "LIN",
    "DIS",
    "AMD",
    "VZ",
    "PFE",
    "INTC",
    "CMCSA",
    "PEP",
    "NKE",
    "QCOM",
    "IBM",
    "AMGN",
    "TXN",
    "HON",
    "SBUX",
    "CAT",
    "GE",
    "SPY",]
TRADING_DAYS = 252
LEX_BOT_ID = os.getenv("LEX_BOT_ID", "P5ZAO8XNQJ")
LEX_BOT_ALIAS_ID = os.getenv("LEX_BOT_ALIAS_ID", "TSTALIASID")
LEX_LOCALE_ID = os.getenv("LEX_LOCALE_ID", "en_US")

st.set_page_config(
    page_title="AI Quant Trading Advisor", page_icon="📈", layout="wide"
)


class S3FileError(Exception):
  """Raised when a file cannot be retrieved from S3."""


@st.cache_resource(show_spinner=False)
def get_boto3_session() -> boto3.Session:
  try:
    aws_secrets = st.secrets.get("aws", {})
  except Exception:
    aws_secrets = {}
  if aws_secrets.get("aws_access_key_id"):
    return boto3.Session(
        aws_access_key_id=aws_secrets["aws_access_key_id"],
        aws_secret_access_key=aws_secrets["aws_secret_access_key"],
        aws_session_token=aws_secrets.get("aws_session_token"),
        region_name=aws_secrets.get("region_name", AWS_REGION),
    )
  return boto3.Session(region_name=AWS_REGION)


@st.cache_resource(show_spinner=False)
def get_s3_client():
  return get_boto3_session().client("s3")


@st.cache_resource(show_spinner=False)
def get_lex_client():
  return get_boto3_session().client("lexv2-runtime")


def ask_lex(text: str, session_id: str) -> str:
  try:
    response = get_lex_client().recognize_text(
        botId=LEX_BOT_ID,
        botAliasId=LEX_BOT_ALIAS_ID,
        localeId=LEX_LOCALE_ID,
        sessionId=session_id,
        text=text,
    )
  except Exception as exc:
    return f"⚠️ AWS connection error with Lex: {exc}"

  replies = [
      m["content"] for m in response.get("messages", []) if m.get("content")
  ]
  return (
      "\n\n".join(replies)
      or "I didn't get a response from the bot. Try rephrasing."
  )


def _get_object_bytes(bucket: str, key: str) -> bytes:
  try:
    response = get_s3_client().get_object(Bucket=bucket, Key=key)
    return response["Body"].read()
  except Exception as exc:
    raise S3FileError(f"Could not retrieve file from S3: {key}") from exc


@st.cache_data(ttl=600, show_spinner="Loading data from S3...")
def load_csv_from_s3(bucket: str, key: str) -> pd.DataFrame:
  raw = _get_object_bytes(bucket, key)
  return pd.read_csv(io.BytesIO(raw))


@st.cache_data(ttl=600, show_spinner="Loading plot from S3...")
def load_image_from_s3(bucket: str, key: str) -> bytes:
  return _get_object_bytes(bucket, key)


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
  df = df.copy()
  date_col = (
      "Unnamed: 0"
      if "Unnamed: 0" in df.columns
      else ("Date" if "Date" in df.columns else df.columns[0])
  )
  df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
  df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)

  close_col = "Close" if "Close" in df.columns else df.columns[1]
  return df, {
      "close": close_col,
      "returns": "Returns",
      "strategy": "Strategy_Return",
  }


def compute_metrics(returns: pd.Series, risk_free_rate: float) -> dict:
  r = returns.dropna()
  if r.empty:
    return {
        k: np.nan
        for k in ("total", "cagr", "vol", "sharpe", "mdd", "win_rate")
    }
  equity = (1 + r).cumprod()
  years = max(len(r) / TRADING_DAYS, 1e-9)
  vol = r.std() * np.sqrt(TRADING_DAYS)
  excess = r.mean() * TRADING_DAYS - risk_free_rate
  return {
      "total": equity.iloc[-1] - 1,
      "cagr": equity.iloc[-1] ** (1 / years) - 1,
      "vol": vol,
      "sharpe": excess / vol if vol > 0 else np.nan,
      "mdd": (equity / equity.cummax() - 1).min(),
      "win_rate": (r > 0).sum() / max((r != 0).sum(), 1),
  }


def fmt_pct(x):
  return "N/A" if pd.isna(x) else f"{x:.2%}"


def fmt_num(x):
  return "N/A" if pd.isna(x) else f"{x:.2f}"


# Initialize Session State for Ticker Portfolio / Chat
if "portfolio_tickers" not in st.session_state:
  st.session_state.portfolio_tickers = ["AAPL", "GOOGL"]
if "active_ticker" not in st.session_state:
  st.session_state.active_ticker = "AAPL"

# Sidebar Controls
with st.sidebar:
  st.title("⚙️ Portfolio Manager")

  selected_ticker = st.selectbox(
      "Active Analyzed Stock",
      st.session_state.portfolio_tickers,
      index=st.session_state.portfolio_tickers.index(
          st.session_state.active_ticker
      )
      if st.session_state.active_ticker in st.session_state.portfolio_tickers
      else 0,
  )
  st.session_state.active_ticker = selected_ticker

  st.markdown("---")
  st.subheader("Add Asset to Portfolio")
  unselected_tickers = [
      t for t in AVAILABLE_TICKERS if t not in st.session_state.portfolio_tickers
  ]
  if unselected_tickers:
    new_ticker_input = st.selectbox("Select available stock", unselected_tickers)
    if st.button("➕ Add to Portfolio"):
      if new_ticker_input not in st.session_state.portfolio_tickers:
        st.session_state.portfolio_tickers.append(new_ticker_input)
        st.session_state.active_ticker = new_ticker_input
        st.rerun()
  else:
    st.caption("All available tickers are in your portfolio!")

  st.markdown("---")
  initial_capital = st.number_input("Initial capital ($)", value=10_000, step=1_000)
  position_size = st.slider("Position size (% of capital)", 1, 100, 25) / 100
  stop_loss = st.slider("Stop-loss (%)", 1.0, 20.0, 5.0, step=0.5) / 100
  risk_free_rate = (
      st.slider("Risk-free rate (%)", 0.0, 10.0, 4.0, step=0.25) / 100
  )

  if st.button("🔄 Refresh Cache"):
    st.cache_data.clear()
    st.rerun()

# Main Header
st.title("📈 AI-Powered Quantitative Trading Assistant")
st.markdown(
    "Interactive Machine Learning Strategy Dashboard & Digital Financial"
    f" Advisor for **{st.session_state.active_ticker}**."
)

# Load Data for Active Ticker
data_key = DATA_KEY_TEMPLATE.format(ticker=st.session_state.active_ticker)
plot_key = PLOT_KEY_TEMPLATE.format(ticker=st.session_state.active_ticker)

try:
  raw_df = load_csv_from_s3(BUCKET_NAME, data_key)
  df, cols = prepare_data(raw_df)
except S3FileError as err:
  st.error(
      f"❌ Data for **{st.session_state.active_ticker}** not found in S3."
      " Please run your Colab ML notebook for this ticker first."
  )
  st.stop()

close = df[cols["close"]]
market_ret = df[cols["returns"]]
strategy_ret = df[cols["strategy"]]

# Metrics Row
market_m = compute_metrics(market_ret, risk_free_rate)
strategy_m = compute_metrics(strategy_ret, risk_free_rate)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest Close", f"${close.iloc[-1]:,.2f}")
c2.metric(
    "ML Strategy Return",
    fmt_pct(strategy_m["total"]),
    f"{(strategy_m['total'] - market_m['total']):+.2%} vs Market",
)
c3.metric(
    "Strategy Sharpe Ratio",
    fmt_num(strategy_m["sharpe"]),
    f"{(strategy_m['sharpe'] - market_m['sharpe']):+.2f} vs Market",
)
c4.metric("Max Drawdown", fmt_pct(strategy_m["mdd"]))

st.divider()

# Layout Tabs: Dashboard & AI Assistant Chat
tab_dashboard, tab_chat = st.tabs(
    ["📊 Strategy & Risk Dashboard", "💬 AI Financial Advisor Chat"]
)

with tab_dashboard:
  col_l, col_r = st.columns(2)
  with col_l:
    st.subheader("Machine Learning Strategy vs Market")
    cum_df = pd.DataFrame(
        {
            "Buy & Hold": (1 + market_ret.fillna(0)).cumprod(),
            "ML Strategy": (1 + strategy_ret.fillna(0)).cumprod(),
        }
    )
    st.line_chart(cum_df, height=350)

  with col_r:
    st.subheader("Strategy Comparison Plot from S3")
    try:
      img_bytes = load_image_from_s3(BUCKET_NAME, plot_key)
      st.image(img_bytes, width="stretch")
    except Exception:
      st.warning("Strategy comparison plot image not found in S3 bucket.")

  st.subheader("Performance Summary Table")
  summary_df = pd.DataFrame(
      {"Buy & Hold Market": market_m, "ML Random Forest Strategy": strategy_m}
  ).rename(
      index={
          "total": "Total Return",
          "cagr": "CAGR",
          "vol": "Annual Volatility",
          "sharpe": "Sharpe Ratio",
          "mdd": "Max Drawdown",
          "win_rate": "Win Rate",
      }
  )
  st.dataframe(summary_df.map(fmt_num), width="stretch")

with tab_chat:
  st.subheader("Conversational AI Advisor (Amazon Lex)")
  st.markdown(
      "Chat with your digital advisor regarding portfolio holdings, future"
      " predictions, buy/sell recommendations, and risk (VaR)."
  )

  if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
  if "lex_session_id" not in st.session_state:
    st.session_state.lex_session_id = f"streamlit-{uuid.uuid4().hex}"

  history = st.container(height=400)
  with history:
    for message in st.session_state.chat_messages:
      with st.chat_message(message["role"]):
        st.markdown(message["content"])

  if prompt := st.chat_input(
      "e.g., How is GOOGL doing? / What is the prediction for AAPL tomorrow?"
  ):
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with history:
      with st.chat_message("user"):
        st.markdown(prompt)
      with st.chat_message("assistant"):
        with st.spinner("Analyzing ML models & portfolio data..."):
          reply = ask_lex(prompt, st.session_state.lex_session_id)
        st.markdown(reply)
    st.session_state.chat_messages.append(
        {"role": "assistant", "content": reply}
    )

  if st.button("🗑️ Reset Conversation"):
    st.session_state.chat_messages = []
    st.session_state.lex_session_id = f"streamlit-{uuid.uuid4().hex}"
    st.rerun()

st.caption("For educational and portfolio management purposes only.")