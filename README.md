# Quantitative Stock Trading Assistant & AI Financial Advisor

An interactive machine learning dashboard and conversational financial advisor powered by Streamlit, AWS S3, and Amazon Lex.

## Features
- **ML Strategy Performance**: Evaluates Random Forest trading models against market benchmarks (Buy & Hold).
- **Key Quantitative Metrics**: Calculates Total Return, CAGR, Volatility, Sharpe Ratio, Maximum Drawdown, and Win Rate.
- **Visual Analytics**: Interactive performance charts and ML strategy comparison visuals loaded directly from AWS S3.
- **Conversational AI Advisor**: Real-time integration with Amazon Lex for portfolio insights, trade sentiment, and risk analysis.
- **Dynamic Portfolio Manager**: Switch between multiple tracked assets (AAPL, GOOGL, MSFT, NVDA, and more).

## Tech Stack
- **Frontend / Dashboard**: Streamlit
- **Cloud Storage**: AWS S3
- **Conversational Bot**: Amazon Lex V2
- **Data & Backtesting**: Pandas, NumPy, Boto3

## Setup & Deployment

### 1. Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Configure AWS Secrets
Create `.streamlit/secrets.toml` with your AWS credentials:
```toml
[aws]
aws_access_key_id = "YOUR_ACCESS_KEY"
aws_secret_access_key = "YOUR_SECRET_KEY"
aws_session_token = "YOUR_SESSION_TOKEN"
region_name = "us-east-1"
```

### 3. Run Locally
```bash
streamlit run app.py
```
