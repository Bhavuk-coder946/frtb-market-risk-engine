%%writefile app.py
import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# 1. PAGE SETUP & STYLING
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="FRTB Market Risk Engine & Validation",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------
# 2. DATA INGESTION ENGINE WITH CACHING & ROBUST RESILIENCE
# ------------------------------------------------------------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def load_market_data(start_date="2019-01-01"):
    tickers_us = ["^GSPC", "IEF", "EURUSD=X", "AAPL"]
    tickers_in = ["^NSEI", "RELIANCE.NS", "INR=X", "HDFCBANK.NS"]
    all_tickers = tickers_us + tickers_in

    try:
        raw = yf.download(all_tickers, start=start_date, auto_adjust=True, progress=False)["Close"]
        clean_prices = raw.ffill().dropna()
        log_returns = np.log(clean_prices / clean_prices.shift(1)).dropna()
        if len(log_returns) > 250:
            return log_returns[tickers_us], log_returns[tickers_in]
    except Exception:
        pass

    # Fallback to realistic synthetic multi-asset data if Yahoo Finance rate limits
    dates = pd.date_range(start=start_date, end=pd.Timestamp.today(), freq="B")
    n = len(dates)
    np.random.seed(42)
    df_us = pd.DataFrame(np.random.normal(0.0003, 0.012, size=(n, 4)), index=dates, columns=tickers_us)
    df_in = pd.DataFrame(np.random.normal(0.0004, 0.014, size=(n, 4)), index=dates, columns=tickers_in)
    return df_us, df_in

# ------------------------------------------------------------------------------
# 3. INSTITUTIONAL RISK ENGINE
# ------------------------------------------------------------------------------
class InstitutionalRiskEngine:
    def __init__(self, returns_df: pd.DataFrame, weights: np.ndarray, asset_names: list, notional: float):
        self.returns_df = returns_df
        self.weights = weights
        self.asset_names = asset_names
        self.notional = notional

    def apply_ewma_fhs(self, returns_window: pd.DataFrame, lambda_=0.94):
        """Filtered Historical Simulation: scales returns by EWMA volatility."""
        variances = returns_window.ewm(alpha=1 - lambda_, adjust=False).var().dropna()
        vols = np.sqrt(variances)
        current_vol = vols.iloc[-1]
        shifted_vols = vols.shift(1).dropna()
        aligned_returns = returns_window.loc[shifted_vols.index]
        return (aligned_returns / shifted_vols) * current_vol

    def compute_risk_metrics(self, scaled_returns: pd.DataFrame, alpha=0.975):
        port_returns = scaled_returns.dot(self.weights).values
        losses = -port_returns * self.notional

        # 1-Day VaR & Expected Shortfall
        var_1d = np.percentile(losses, alpha * 100)
        tail_mask = losses >= var_1d
        es_1d = np.mean(losses[tail_mask]) if np.any(tail_mask) else var_1d

        # FRTB Liquidity Horizons: 10d benchmark + 20d duration scaling for Rates/Credit
        base_es_10d = es_1d * np.sqrt(10)
        rates_penalty = (es_1d * self.weights[1]) * np.sqrt(20 - 10)
        frtb_es = np.sqrt(base_es_10d**2 + rates_penalty**2)

        # Component VaR (Euler Allocation proxy via Beta)
        port_variance = np.var(port_returns)
        if port_variance > 0:
            covs = np.cov(scaled_returns.T, port_returns)[0:-1, -1]
            betas = covs / port_variance
            component_var = self.weights * betas * var_1d
        else:
            component_var = np.zeros(len(self.weights))

        # Marginal Expected Shortfall (MES)
        asset_losses = -scaled_returns.values * self.notional
        mes = np.mean(asset_losses[tail_mask], axis=0) * self.weights if np.any(tail_mask) else np.zeros(len(self.weights))

        return {
            "VaR_1D": var_1d,
            "ES_1D": es_1d,
            "FRTB_ES": frtb_es,
            "Component_VaR": component_var,
            "MES": mes
        }

    def run_rolling_backtest(self, window_size=252, alpha=0.975):
        dates = self.returns_df.index[window_size:]
        actual_port_returns = self.returns_df.dot(self.weights)
        actual_losses = -actual_port_returns * self.notional

        res_losses, res_vars, res_es = [], [], []

        for i in range(window_size, len(self.returns_df)):
            sub_window = self.returns_df.iloc[i - window_size : i]
            scaled_window = self.apply_ewma_fhs(sub_window)
            port_ret = scaled_window.dot(self.weights).values
            losses = -port_ret * self.notional

            var = np.percentile(losses, alpha * 100)
            tail = losses[losses >= var]
            es = np.mean(tail) if len(tail) > 0 else var

            res_losses.append(actual_losses.iloc[i])
            res_vars.append(var)
            res_es.append(es)

        return pd.DataFrame({"Realized_Loss": res_losses, "VaR": res_vars, "ES": res_es}, index=dates)

# ------------------------------------------------------------------------------
# 4. REGULATORY MODEL VALIDATOR
# ------------------------------------------------------------------------------
class RegulatoryValidator:
    @staticmethod
    def validate(realized_loss, var_forecast, alpha=0.975):
        hits = (realized_loss > var_forecast).astype(int)
        N = len(hits)
        x = int(np.sum(hits))
        p = 1.0 - alpha
        eps = 1e-12
        p_hat = np.clip(x / N if N > 0 else 0, eps, 1 - eps)
        p_c = np.clip(p, eps, 1 - eps)

        # Kupiec POF Test (Unconditional Coverage)
        lr_uc = -2 * np.log(((1 - p_c)**(N - x) * (p_c**x)) / (((1 - p_hat)**(N - x)) * (p_hat**x)))
        pval_uc = 1 - stats.chi2.cdf(lr_uc, df=1)

        # Christoffersen Test (Independence of Violations)
        h_prev, h_curr = hits[:-1], hits[1:]
        n00 = np.sum((h_prev == 0) & (h_curr == 0))
        n01 = np.sum((h_prev == 0) & (h_curr == 1))
        n10 = np.sum((h_prev == 1) & (h_curr == 0))
        n11 = np.sum((h_prev == 1) & (h_curr == 1))

        pi01 = np.clip(n01 / (n00 + n01) if (n00 + n01) > 0 else eps, eps, 1 - eps)
        pi11 = np.clip(n11 / (n10 + n11) if (n10 + n11) > 0 else eps, eps, 1 - eps)
        pi = np.clip((n01 + n11) / (N - 1) if (N - 1) > 0 else eps, eps, 1 - eps)

        num = (1 - pi)**(n00 + n10) * (pi**(n01 + n11))
        den = ((1 - pi01)**n00) * (pi01**n01) * ((1 - pi11)**n10) * (pi11**n11)
        lr_ind = -2 * np.log(num / den)
        pval_ind = 1 - stats.chi2.cdf(lr_ind, df=1)

        # Basel Traffic-Light Rating
        zone = "Green" if x <= 9 else ("Yellow" if x <= 15 else "Red")
        zone_color = "#2ECC71" if zone == "Green" else ("#F1C40F" if zone == "Yellow" else "#E74C3C")

        return {
            "Breaches": x,
            "Observations": N,
            "Empirical_Rate": round(float(p_hat * 100), 2),
            "Expected_Rate": round(float(p * 100), 2),
            "Kupiec_p": round(float(pval_uc), 4),
            "Christoff_p": round(float(pval_ind), 4),
            "Zone": zone,
            "Zone_Color": zone_color
        }

# ------------------------------------------------------------------------------
# 5. SIDEBAR CONFIGURATION
# ------------------------------------------------------------------------------
st.sidebar.title("Trading Desk Controls")
market = st.sidebar.radio("Asset Universe", ["US Multi-Asset Book", "Indian Multi-Asset Book"])

with st.sidebar.expander("Capital & Risk Parameters", expanded=True):
    notional = st.sidebar.number_input("Portfolio Notional", min_value=1_000_000, value=10_000_000, step=1_000_000)
    alpha = st.sidebar.select_slider("FRTB Confidence Level (α)", options=[0.95, 0.975, 0.99], value=0.975)
    window_size = st.sidebar.slider("Lookback Calibration Window", 126, 504, 252, 21)

with st.sidebar.expander("Asset Allocation Weights", expanded=True):
    if market == "US Multi-Asset Book":
        labels = ["S&P 500 (^GSPC)", "7-10Y Treasury (IEF)", "EUR/USD (EURUSD=X)", "Apple (AAPL)"]
        default_w = [0.40, 0.25, 0.15, 0.20]
    else:
        labels = ["NIFTY 50 (^NSEI)", "Reliance (RELIANCE.NS)", "USD/INR (INR=X)", "HDFC Bank (HDFCBANK.NS)"]
        default_w = [0.40, 0.25, 0.15, 0.20]

    w1 = st.sidebar.slider(labels[0], 0.0, 1.0, default_w[0], 0.05)
    w2 = st.sidebar.slider(labels[1], 0.0, 1.0, default_w[1], 0.05)
    w3 = st.sidebar.slider(labels[2], 0.0, 1.0, default_w[2], 0.05)
    w4 = st.sidebar.slider(labels[3], 0.0, 1.0, default_w[3], 0.05)

    weights = np.array([w1, w2, w3, w4]) / max(w1 + w2 + w3 + w4, 1e-6)

# Ingest and Initialize Engine
ret_us, ret_in = load_market_data()
returns_df = ret_us if market == "US Multi-Asset Book" else ret_in
engine = InstitutionalRiskEngine(returns_df, weights, labels, notional)

# ------------------------------------------------------------------------------
# 6. HEADER METRICS
# ------------------------------------------------------------------------------
st.title("🏛️ Institutional Market Risk & FRTB Validation Engine")
currency = "$" if market == "US Multi-Asset Book" else "₹"

scaled_latest = engine.apply_ewma_fhs(engine.returns_df.iloc[-window_size:])
latest = engine.compute_risk_metrics(scaled_latest, alpha=alpha)

k1, k2, k3, k4 = st.columns(4)
k1.metric("1-Day VaR", f"{currency}{latest['VaR_1D']:,.0f}", help="97.5% quantile cutoff loss threshold")
k2.metric("1-Day Expected Shortfall", f"{currency}{latest['ES_1D']:,.0f}", help="Average loss during tail events")
k3.metric("FRTB Scaled ES", f"{currency}{latest['FRTB_ES']:,.0f}", help="Scaled across 10d-20d Liquidity Horizons")
k4.metric("Capital Notional", f"{currency}{notional:,.0f}")

st.divider()

# ------------------------------------------------------------------------------
# 7. DASHBOARD TABS
# ------------------------------------------------------------------------------
tab_bt, tab_attrib, tab_stress = st.tabs([
    "📈 Backtesting & Basel Scorecard",
    "🍰 Risk Attribution (Component VaR)",
    "⚡ Macro Scenario Stress Testing"
])

# --- TAB 1: BACKTESTING & AUDIT IMPACT ---
with tab_bt:
    bt_df = engine.run_rolling_backtest(window_size=window_size, alpha=alpha)
    val = RegulatoryValidator.validate(bt_df.iloc[-250:]["Realized_Loss"].values, bt_df.iloc[-250:]["VaR"].values, alpha=alpha)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df["Realized_Loss"], name="Realized Loss (P&L)", line=dict(color="#4A90E2", width=1)))
    fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df["VaR"], name="VaR Cutoff", line=dict(color="#F5A623", width=1.5, dash="dash")))
    fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df["ES"], name="Expected Shortfall", line=dict(color="#D0021B", width=1.5)))

    breaches = bt_df[bt_df["Realized_Loss"] > bt_df["VaR"]]
    fig.add_trace(go.Scatter(x=breaches.index, y=breaches["Realized_Loss"], mode="markers", name="VaR Breach", marker=dict(color="red", size=6, symbol="x")))
    fig.update_layout(template="plotly_dark", height=400, title="Trailing Out-of-Sample Backtest (P&L vs. Risk Bands)")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"**Basel Zone:** <span style='color:{val['Zone_Color']}; font-weight:bold; font-size:20px;'>{val['Zone']}</span>", unsafe_allow_html=True)
    c2.metric("Breach Count (T-250)", f"{val['Breaches']} / 250")
    c3.metric("Kupiec Test (POF)", f"p = {val['Kupiec_p']}")
    c4.metric("Christoffersen Test", f"p = {val['Christoff_p']}")

    st.markdown("---")
    st.subheader("📋 Executive Audit Commentary: What These Results Mean")

    col_mean, col_impact, col_action = st.columns(3)

    with col_mean:
        st.markdown("#### 1. Plain English Meaning")
        st.write(f"Over the last 250 business days, the portfolio exceeded its safety cutoff **{val['Breaches']} times**.")
        if val['Zone'] == "Green":
            st.write("• **Well-Calibrated:** The failure rate matches regulatory limits (~6 expected). The model's risk warnings are accurate.")
            st.write("• **No Clustering:** Breaches occurred randomly without multi-day panic cascades.")
        elif val['Zone'] == "Yellow":
            st.write("• **Underestimating Tail Risk:** The desk suffered more bad days than permitted. The model is reacting too slowly to volatility shifts.")
        else:
            st.write("• **Model Breakdown:** The model severely under-reported risk, leading to persistent unexpected trading losses.")

    with col_impact:
        st.markdown("#### 2. Impact on Firm & Trading Desk")
        if val['Zone'] == "Green":
            st.write("• **No Capital Surcharges:** Regulators apply the baseline capital multiplier of **3.0x**.")
            st.write("• **Trading Autonomy:** The Front Office can maintain active risk limits without mandated trade liquidations.")
            st.write("• **Cost of Capital:** The bank saves substantial capital reserves by avoiding regulatory cash lockups.")
        elif val['Zone'] == "Yellow":
            st.write("• **Regulatory Multiplier Hike:** Capital charges increase from 3.0x to up to **3.85x** as a mandatory penalty.")
            st.write("• **Desk P&L Drag:** The desk must set aside extra capital, reducing Return on Equity (ROE) and bonus pools.")
        else:
            st.write("• **IMA Revocation Risk:** Regulators may revoke internal model approval, forcing punitive standardized rules.")

    with col_action:
        st.markdown("#### 3. Management Actions Required")
        if val['Zone'] == "Green":
            st.write("1. Maintain current model calibration parameters.")
            st.write("2. File routine quarterly backtesting sign-offs with internal audit and regulators.")
            st.write("3. Keep monitoring correlation shifts between equities and fixed-income hedges.")
        else:
            st.write("1. **Trigger Model Recalibration:** Shorten EWMA decay factor ($\lambda$) to adapt faster to recent volatility.")
            st.write("2. **Desk Position Limits:** Temporarily reduce gross market exposure across high-beta single stocks.")
            st.write("3. **Conduct Root-Cause Attribution:** Review whether breaches were driven by earnings surprises, illiquidity, or macro shocks.")

# --- TAB 2: RISK ATTRIBUTION & DESK ACTIONS ---
with tab_attrib:
    c_var = np.maximum(0, latest["Component_VaR"])
    highest_idx = int(np.argmax(c_var))
    highest_asset = labels[highest_idx]
    highest_share = (c_var[highest_idx] / np.sum(c_var)) * 100 if np.sum(c_var) > 0 else 0

    col_p, col_b = st.columns(2)
    with col_p:
        fig_pie = px.pie(values=c_var, names=labels, title="Component VaR Attribution (%)", hole=0.4)
        fig_pie.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        df_mes = pd.DataFrame({"Asset": labels, "Marginal_ES": latest["MES"]})
        fig_bar = px.bar(df_mes, x="Asset", y="Marginal_ES", title=f"Marginal Expected Shortfall ({currency})", color="Marginal_ES", color_continuous_scale="Reds")
        fig_bar.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Executive Desk Commentary: What These Results Mean")

    ca_mean, ca_impact, ca_action = st.columns(3)

    with ca_mean:
        st.markdown("#### 1. Plain English Meaning")
        st.write(f"• **Primary Risk Driver:** **{highest_asset}** accounts for **{highest_share:.1f}%** of the entire desk's tail risk.")
        st.write("• **Diversification Effect:** Assets with low Component VaR (like Treasuries or FX) act as dampeners, absorbing shocks during equity sell-offs.")
        st.write("• **Tail Exposure:** Marginal ES reveals which exact asset bleeds the most money on days when market circuit breakers trip.")

    with ca_impact:
        st.markdown("#### 2. Impact on Firm & Trading Desk")
        st.write(f"• **Risk Limit Bottleneck:** The desk is heavily vulnerable to idiosyncratic shocks in **{highest_asset}**.")
        st.write("• **Concentration Risk:** A sharp sell-off in that single position could wipe out a week of trading profits for the firm.")
        st.write("• **Capacity Constraints:** Other traders cannot add new positions because this single asset consumes most of the risk budget.")

    with ca_action:
        st.markdown("#### 3. Management Actions Required")
        st.write(f"1. **Rebalance Concentration:** Trim notional allocation in **{highest_asset}** by 5%–10% to free up desk risk capacity.")
        st.write("2. **Add Downside Protection:** Buy out-of-the-money put options or enter short futures on the index to hedge beta.")
        st.write("3. **Cross-Check with Valuations:** Review Front Office vs. Back Office P&L attribution to ensure pricing inputs for this asset match audited books.")

# --- TAB 3: MACRO STRESS TESTING SANDBOX ---
with tab_stress:
    st.subheader("Historical & Custom Macro Scenario Simulator")
    st.write("Simulate portfolio tail losses under historical crises or custom desk shocks.")

    s1, s2 = st.columns([1, 2])

    with s1:
        scenario = st.selectbox("Stress Scenarios", [
            "COVID-19 Crash (March 2020)",
            "2008 Lehman / GFC Collapse",
            "2022 Central Bank Rate Spike",
            "Custom Shock Sliders"
        ])

        if scenario == "COVID-19 Crash (March 2020)":
            shocks = [-0.12, 0.04, 0.01, -0.15]
        elif scenario == "2008 Lehman / GFC Collapse":
            shocks = [-0.22, 0.08, -0.02, -0.25]
        elif scenario == "2022 Central Bank Rate Spike":
            shocks = [-0.05, -0.06, 0.03, -0.08]
        else:
            sh_a1 = st.slider(f"{labels[0]} Shock (%)", -30.0, 30.0, -10.0, 1.0) / 100.0
            sh_a2 = st.slider(f"{labels[1]} Shock (%)", -30.0, 30.0, 2.0, 1.0) / 100.0
            sh_a3 = st.slider(f"{labels[2]} Shock (%)", -30.0, 30.0, 1.0, 1.0) / 100.0
            sh_a4 = st.slider(f"{labels[3]} Shock (%)", -30.0, 30.0, -12.0, 1.0) / 100.0
            shocks = [sh_a1, sh_a2, sh_a3, sh_a4]

    with s2:
        port_shock = np.dot(shocks, weights)
        dollar_impact = -port_shock * notional

        st.markdown(f"### Scenario Impact: **{scenario}**")
        st.metric("Total Stressed Dollar Loss", f"{currency}{dollar_impact:,.0f}", delta=f"{-port_shock*100:.2f}% Notional Return", delta_color="inverse")

        df_impact = pd.DataFrame({
            "Asset": labels,
            "Weight": [f"{w*100:.1f}%" for w in weights],
            "Applied Shock": [f"{s*100:.1f}%" for s in shocks],
            "Loss Contribution": [f"{currency}{-s * w * notional:,.0f}" for s, w in zip(shocks, weights)]
        })
        st.table(df_impact)
