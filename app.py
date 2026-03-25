import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import date, timedelta
import holidays

# ── Página ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Calculadora Black–Scholes BR",
    layout="wide",
    page_icon="📊",
)

# ── CSS — estilo Calculadora de Prazos ───────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main {
    background: #f0f4f0 !important;
    color: #1a1a1a !important;
    font-family: 'Roboto', sans-serif !important;
}

.block-container {
    padding: 1.2rem 1.6rem 2rem 1.6rem !important;
    max-width: 1280px !important;
}

[data-testid="stSidebar"] { display: none !important; }

/* ── Section header bar ── */
.sec-bar {
    display: flex;
    align-items: center;
    gap: 10px;
    background: linear-gradient(90deg, #2e9e4f 0%, #4cbb6e 100%);
    color: #fff !important;
    font-family: 'Roboto', sans-serif;
    font-size: 16px;
    font-weight: 700;
    padding: 10px 18px;
    border-radius: 8px;
    margin-bottom: 14px;
    margin-top: 6px;
    letter-spacing: 0.2px;
    box-shadow: 0 2px 8px rgba(46,158,79,0.18);
}
.sec-bar .ico { font-size: 20px; line-height: 1; }

/* ── Result info row ── */
.result-row {
    display: flex;
    flex-wrap: wrap;
    gap: 28px;
    font-size: 14px;
    color: #1a1a1a;
    padding: 6px 0 10px 0;
    border-bottom: 1px solid #e8f0e8;
    margin-bottom: 10px;
}
.result-row span { font-weight: 400; color: #444; }
.result-row b    { font-weight: 600; color: #1a1a1a; }

/* ── Status boxes ── */
.box-ok {
    background: #e8f8ed;
    border: 1px solid #74cc91;
    border-radius: 8px;
    padding: 12px 18px;
    margin: 6px 0 10px 0;
    font-size: 15px;
    font-weight: 500;
    color: #1a5e2e;
}
.box-warn {
    background: #fff8e1;
    border: 1px solid #f4c842;
    border-radius: 8px;
    padding: 12px 18px;
    margin: 6px 0;
    font-size: 14px;
    color: #7a5500;
}
.box-info {
    background: #eaf3fb;
    border: 1px solid #90c4e8;
    border-radius: 8px;
    padding: 12px 18px;
    margin: 6px 0;
    font-size: 14px;
    color: #1a4a6e;
}

/* ── Gregas grid ── */
.gk-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 10px;
    margin: 4px 0 8px 0;
}
.gk-card {
    background: #f6faf6;
    border: 1px solid #c8e0c8;
    border-radius: 8px;
    padding: 10px 12px;
    text-align: center;
}
.gk-label {
    font-size: 10px;
    font-weight: 600;
    color: #4a7a4a;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 4px;
}
.gk-value {
    font-size: 17px;
    font-weight: 700;
    color: #1a5e2e;
}

/* ── Vol cards ── */
.vol-card {
    background: #f6faf6;
    border: 1px solid #c8e0c8;
    border-radius: 8px;
    padding: 10px 14px;
    text-align: center;
    height: 100%;
}
.vol-label {
    font-size: 10px;
    font-weight: 600;
    color: #4a7a4a;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 4px;
}
.vol-value {
    font-size: 20px;
    font-weight: 700;
    color: #1a5e2e;
}

/* ── Risk grid ── */
.risk-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 10px;
    margin: 6px 0;
}
.risk-card {
    background: #f6faf6;
    border-left: 4px solid #2e9e4f;
    border-radius: 0 8px 8px 0;
    padding: 10px 12px;
}
.risk-label {
    font-size: 10px;
    font-weight: 600;
    color: #4a7a4a;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 3px;
}
.risk-value {
    font-size: 18px;
    font-weight: 700;
    color: #1a1a1a;
}
.risk-help {
    font-size: 10px;
    color: #888;
    margin-top: 2px;
}

/* ── Streamlit widget overrides ── */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stDateInput"] input {
    background: #fff !important;
    border: 1px solid #c8d8c8 !important;
    border-radius: 6px !important;
    color: #1a1a1a !important;
    font-family: 'Roboto', sans-serif !important;
    font-size: 14px !important;
    padding: 7px 10px !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus {
    border-color: #2e9e4f !important;
    box-shadow: 0 0 0 2px rgba(46,158,79,0.15) !important;
}
[data-testid="stSelectbox"] > div > div {
    background: #fff !important;
    border: 1px solid #c8d8c8 !important;
    border-radius: 6px !important;
    color: #1a1a1a !important;
    font-family: 'Roboto', sans-serif !important;
    font-size: 14px !important;
}
label, [data-testid="stWidgetLabel"] p {
    font-family: 'Roboto', sans-serif !important;
    font-size: 12px !important;
    color: #444 !important;
    font-weight: 500 !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    margin-bottom: 3px !important;
}

/* Metric fallback */
[data-testid="stMetric"] {
    background: #f6faf6 !important;
    border: 1px solid #c8e0c8 !important;
    border-radius: 8px !important;
    padding: 10px 12px !important;
}
[data-testid="stMetricLabel"] p {
    font-size: 10px !important;
    font-weight: 600 !important;
    color: #4a7a4a !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}
[data-testid="stMetricValue"] {
    font-size: 17px !important;
    font-weight: 700 !important;
    color: #1a5e2e !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #2e9e4f 0%, #3db360 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Roboto', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 7px 16px !important;
    transition: filter 0.15s ease, box-shadow 0.15s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    filter: brightness(1.08) !important;
    box-shadow: 0 3px 10px rgba(46,158,79,0.28) !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 8px !important;
    border: 1px solid #c8e0c8 !important;
    overflow: hidden !important;
}

/* hr */
hr {
    border: none !important;
    border-top: 1px solid #dde8dd !important;
    margin: 8px 0 !important;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  FUNÇÕES
# ═══════════════════════════════════════════════════════════════════════════

def get_stock_price(ticker: str):
    try:
        h = yf.Ticker(ticker).history(period="1d")
        return float(h['Close'].iloc[-1]) if not h.empty else None
    except Exception:
        return None


def get_historical_prices(ticker: str, period: str = "1y"):
    try:
        return yf.Ticker(ticker).history(period=period)
    except Exception:
        return pd.DataFrame()


def calculate_annualized_volatility(returns: pd.Series, trading_days: int = 252):
    if returns.empty:
        return np.nan
    return returns.std(ddof=1) * np.sqrt(trading_days)


def calculate_garch11(
    returns: pd.Series,
    n_init:   int   = 50,
    alpha:    float = 0.05,
    beta:     float = 0.90,
    gamma:    float = 0.05,
    reducer:  float = 0.80,
):
    """
    GARCH(1,1) com janela de inicialização configurável.
      n_init  — obs. para estimar σ² inicial (aquecimento)
      alpha   — peso ARCH  (ε²_{t-1})
      beta    — persistência GARCH (σ²_{t-1})
      gamma   — reservado (GJR-GARCH / extensões)
      reducer — multiplicador aplicado ao vol anual final
    ω = σ²_LP × (1 − α − β), com σ²_LP da janela de aquecimento.
    """
    if returns.empty or len(returns) < n_init + 2:
        return np.nan, np.nan

    res    = returns - returns.mean()
    var_lp = res.iloc[:n_init].var(ddof=1)
    omega  = max(var_lp * (1.0 - alpha - beta), 1e-10)

    sigma2    = np.empty(len(res))
    sigma2[0] = var_lp
    for t in range(1, len(res)):
        sigma2[t] = omega + alpha * (res.iloc[t - 1] ** 2) + beta * sigma2[t - 1]

    sigma_annual = np.sqrt(max(sigma2[-1], 1e-12)) * np.sqrt(252)
    return sigma_annual, sigma_annual * reducer


def calculate_business_days(start: date, end: date) -> int:
    br_h = holidays.Brazil(years=list(range(start.year, end.year + 1)))
    bd, cur = 0, start
    while cur < end:
        if cur.weekday() < 5 and cur not in br_h:
            bd += 1
        cur += timedelta(days=1)
    return bd


def black_scholes(S, K, T_years, r, sigma, opt='Call'):
    if T_years <= 0:
        intr = max(0.0, S - K) if opt == 'Call' else max(0.0, K - S)
        return {'Preço Teórico': intr, 'Delta': 0.0, 'Gamma': 0.0,
                'Vega': 0.0, 'Theta': 0.0, 'Rho': 0.0}

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T_years) / (sigma * np.sqrt(T_years))
    d2 = d1 - sigma * np.sqrt(T_years)

    if opt == 'Call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T_years) * norm.cdf(d2)
        delta = norm.cdf(d1)
        rho   = K * T_years * np.exp(-r * T_years) * norm.cdf(d2)
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T_years))
                 - r * K * np.exp(-r * T_years) * norm.cdf(d2))
    else:
        price = K * np.exp(-r * T_years) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1.0
        rho   = -K * T_years * np.exp(-r * T_years) * norm.cdf(-d2)
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T_years))
                 + r * K * np.exp(-r * T_years) * norm.cdf(-d2))

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T_years))
    vega  = S * norm.pdf(d1) * np.sqrt(T_years)

    return {
        'Preço Teórico': price,
        'Delta':  delta,
        'Gamma':  gamma,
        'Vega':   vega  / 100.0,
        'Theta':  theta / 252.0,
        'Rho':    rho   / 100.0,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  ESTADO
# ═══════════════════════════════════════════════════════════════════════════

if 'spot_price' not in st.session_state:
    st.session_state.spot_price = 0.0
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []


# ═══════════════════════════════════════════════════════════════════════════
#  ① HEADER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="sec-bar" style="font-size:20px; padding:14px 20px; margin-top:0;">
  <span class="ico">📊</span> Calculadora Black–Scholes BR
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  ② ATIVO BASE & PARÂMETROS
# ═══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="sec-bar"><span class="ico">📋</span> Ativo Base &amp; Parâmetros</div>',
            unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns([2, 1.2, 2, 2, 2])

with c1:
    ticker_input = st.text_input("Ticker (ex: PETR4.SA, VALE3.SA)",
                                  value="BOVA11.SA", key="ticker_input").upper()

# ── Auto-fetch: roda sempre que o ticker muda (comparando com último ticker buscado) ──
if st.session_state.get('last_fetched_ticker') != ticker_input:
    p_auto = get_stock_price(ticker_input)
    if p_auto:
        st.session_state.spot_price        = p_auto
        st.session_state['spot_input']     = p_auto   # força atualização do widget
        st.session_state['last_fetched_ticker'] = ticker_input

# with c2:
#     st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
#     if st.button("↻ Atualizar Preço", key="btn_price"):
#         p = get_stock_price(ticker_input)
#         if p:
#             st.session_state.spot_price    = p
#             st.session_state['spot_input'] = p        # atualiza o campo diretamente
#             st.session_state['last_fetched_ticker'] = ticker_input
#             st.rerun()
#         else:
#             st.warning("Não foi possível obter o preço. Informe manualmente.")

with c3:
    spot_price = st.number_input(
        "Preço Atual do Ativo (S)",
        value=float(st.session_state.get('spot_input', st.session_state.spot_price or 0.01)),
        format="%.2f", step=0.01, key="spot_input"
    )

with c4:
    rf_pct = st.number_input("Taxa Livre de Risco a.a. (%)",
                              value=14.75, step=0.25, format="%.2f", key="rf_input")
    risk_free_rate = rf_pct / 100.0

with c5:
    vol_pct = st.number_input("Volatilidade Implícita a.a. (%)",
                               value=15.0, step=0.5, format="%.2f", key="vol_input")
    volatility = vol_pct / 100.0

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════
#  ③ PARÂMETROS DA OPÇÃO
# ═══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="sec-bar"><span class="ico">⚙️</span> Parâmetros da Opção</div>',
            unsafe_allow_html=True)

p1, p2, p3, p4, p5 = st.columns([1.5, 2, 2, 1.5, 1.5])

with p1:
    option_type = st.selectbox("Tipo", ["Call", "Put", "Ativo"], key="opt_type")

with p2:
    if option_type in ["Call", "Put"]:
        strike_price = st.number_input(
            "Strike (K)", value=spot_price if spot_price > 0 else 10.0,
            step=0.01, format="%.2f", key="strike_input"
        )
    else:
        strike_price = None
        st.number_input("Strike (K)", value=0.0, disabled=True, key="strike_dis")

with p3:
    today = date.today()
    expiration_date = st.date_input("Data de Vencimento",
                                    value=today + timedelta(days=30),
                                    min_value=today, key="venc_input")

business_days    = calculate_business_days(today, expiration_date)
time_to_maturity = business_days / 252.0   

with p4:
    st.metric("Dias Úteis até o Vencimento", business_days)

with p5:
    quantity = st.number_input("Quantidade",
                               value=1, step=1, key="qty_input",
                               help="Positivo = compra  |  Negativo = venda")

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════
#  ④ RESULTADO — PREÇO TEÓRICO & GREGAS
# ═══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="sec-bar"><span class="ico">📈</span> Resultado — Preço Teórico &amp; Gregas</div>',
            unsafe_allow_html=True)

if option_type in ["Call", "Put"] and spot_price > 0 and strike_price is not None:
    sim = black_scholes(spot_price, strike_price, time_to_maturity,
                        risk_free_rate, volatility, option_type)

    if option_type == "Call":
        ratio = abs(spot_price - strike_price) / strike_price
        moneyness = "ITM 🟢" if spot_price > strike_price else ("ATM ⚪" if ratio < 0.01 else "OTM 🔴")
    else:
        ratio = abs(spot_price - strike_price) / strike_price
        moneyness = "ITM 🟢" if spot_price < strike_price else ("ATM ⚪" if ratio < 0.01 else "OTM 🔴")

    st.markdown(f"""
    <div class="result-row">
      <div><span>Ativo:</span> <b>{ticker_input}</b></div>
      <div><span>Tipo:</span> <b>{option_type}</b></div>
      <div><span>Strike (K):</span> <b>R$ {strike_price:.2f}</b></div>
      <div><span>Spot (S):</span> <b>R$ {spot_price:.2f}</b></div>
      <div><span>Dias úteis:</span> <b>{business_days}</b></div>
      <div><span>Vencimento:</span> <b>{expiration_date.strftime('%d/%m/%Y')}</b></div>
      <div><span>Vol. implícita:</span> <b>{vol_pct:.1f}%</b></div>
      <div><span>Taxa:</span> <b>{rf_pct:.2f}%</b></div>
    </div>
    <div class="box-ok">
      ✅ &nbsp;<b>Preço Teórico: R$ {sim['Preço Teórico']:.4f}</b>
      &nbsp;·&nbsp; Moneyness: {moneyness}
    </div>
    <div class="gk-grid">
      <div class="gk-card">
        <div class="gk-label">Preço Teórico</div>
        <div class="gk-value">R$ {sim['Preço Teórico']:.4f}</div>
      </div>
      <div class="gk-card">
        <div class="gk-label">Delta</div>
        <div class="gk-value">{sim['Delta']:.4f}</div>
      </div>
      <div class="gk-card">
        <div class="gk-label">Gamma</div>
        <div class="gk-value">{sim['Gamma']:.4f}</div>
      </div>
      <div class="gk-card">
        <div class="gk-label">Vega / 1%</div>
        <div class="gk-value">{sim['Vega']:.4f}</div>
      </div>
      <div class="gk-card">
        <div class="gk-label">Theta / d.u.</div>
        <div class="gk-value">{sim['Theta']:.4f}</div>
      </div>
      <div class="gk-card">
        <div class="gk-label">Rho / 1%</div>
        <div class="gk-value">{sim['Rho']:.4f}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

elif option_type == "Ativo" and spot_price > 0:
    st.markdown(f"""
    <div class="result-row">
      <div><span>Ativo:</span> <b>{ticker_input}</b></div>
      <div><span>Tipo:</span> <b>Ativo objeto</b></div>
      <div><span>Preço:</span> <b>R$ {spot_price:.2f}</b></div>
    </div>
    <div class="box-ok">✅ &nbsp;<b>Ativo objeto selecionado — Delta = 1,0000</b></div>
    <div class="gk-grid">
      <div class="gk-card"><div class="gk-label">Preço</div><div class="gk-value">R$ {spot_price:.2f}</div></div>
      <div class="gk-card"><div class="gk-label">Delta</div><div class="gk-value">1.0000</div></div>
      <div class="gk-card"><div class="gk-label">Gamma</div><div class="gk-value">0.0000</div></div>
      <div class="gk-card"><div class="gk-label">Vega / 1%</div><div class="gk-value">0.0000</div></div>
      <div class="gk-card"><div class="gk-label">Theta / d.u.</div><div class="gk-value">0.0000</div></div>
      <div class="gk-card"><div class="gk-label">Rho / 1%</div><div class="gk-value">0.0000</div></div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown('<div class="box-info">ℹ️ &nbsp;Informe o preço do ativo, tipo e strike para calcular.</div>',
                unsafe_allow_html=True)

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════
#  ⑤ VOLATILIDADE HISTÓRICA + GARCH + BOTÕES
# ═══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="sec-bar"><span class="ico">📉</span> Volatilidade Histórica &amp; GARCH(1,1)</div>',
            unsafe_allow_html=True)

GARCH_N_INIT  = 50
GARCH_ALPHA   = 0.05
GARCH_BETA    = 0.90
GARCH_GAMMA   = 0.05
GARCH_REDUCER = 0.80

hist_data  = get_historical_prices(ticker_input, period="1y")
vol_annual = garch_raw = garch_red = None

if not hist_data.empty and len(hist_data) > GARCH_N_INIT + 5:
    daily_returns = hist_data['Close'].pct_change().dropna()
    vol_annual    = calculate_annualized_volatility(daily_returns, 252)
    garch_raw, garch_red = calculate_garch11(
        daily_returns,
        n_init=GARCH_N_INIT, alpha=GARCH_ALPHA,
        beta=GARCH_BETA,     gamma=GARCH_GAMMA,
        reducer=GARCH_REDUCER,
    )

vc1, vc2, vc3, vb1, vb2, vb3 = st.columns([2, 2, 2, 1.4, 1.4, 1.4])

with vc1:
    v1 = f"{vol_annual * 100:.2f}%" if vol_annual is not None and not np.isnan(vol_annual) else "—"
    st.markdown(f"""
    <div class="vol-card">
      <div class="vol-label">Vol. Histórica a.a.</div>
      <div class="vol-value">{v1}</div>
    </div>""", unsafe_allow_html=True)

with vc2:
    v2 = f"{garch_raw * 100:.2f}%" if garch_raw is not None and not np.isnan(garch_raw) else "—"
    st.markdown(f"""
    <div class="vol-card">
      <div class="vol-label">GARCH(1,1) &nbsp;<small style='font-weight:400;color:#6a9e6a'>α={GARCH_ALPHA} β={GARCH_BETA} n={GARCH_N_INIT}</small></div>
      <div class="vol-value">{v2}</div>
    </div>""", unsafe_allow_html=True)

with vc3:
    v3 = f"{garch_red * 100:.2f}%" if garch_red is not None and not np.isnan(garch_red) else "—"
    st.markdown(f"""
    <div class="vol-card">
      <div class="vol-label">GARCH × {int(GARCH_REDUCER*100)}% &nbsp;<small style='font-weight:400;color:#6a9e6a'>(−{int((1-GARCH_REDUCER)*100)}%)</small></div>
      <div class="vol-value">{v3}</div>
    </div>""", unsafe_allow_html=True)

with vb1:
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    add_clicked = st.button("＋ Adicionar", key="btn_add")

with vb2:
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    update_clicked = st.button("↺ Atualizar", key="btn_upd")

with vb3:
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    clear_clicked = st.button("✕ Limpar", key="btn_clr")


# ── Lógica dos botões ─────────────────────────────────────────────────────

if add_clicked:
    if option_type == "Ativo":
        item = {
            "Tipo": "Ativo", "Strike": np.nan, "Vencimento": None,
            "Qtd": float(quantity), "IV": np.nan, "Dias Úteis": np.nan,
            "Preço Teórico Unit.": float(spot_price),
            "Delta Unit.": 1.0, "Gamma Unit.": 0.0,
            "Vega Unit.": 0.0, "Theta Unit.": 0.0, "Rho Unit.": 0.0,
            "Delta Total": 1.0 * quantity, "Gamma Total": 0.0,
            "Vega Total": 0.0, "Theta Total": 0.0, "Rho Total": 0.0,
        }
    elif spot_price > 0 and strike_price is not None:
        res = black_scholes(spot_price, strike_price, time_to_maturity,
                            risk_free_rate, volatility, option_type)
        item = {
            "Tipo": option_type, "Strike": float(strike_price),
            "Vencimento": expiration_date, "Qtd": float(quantity),
            "IV": float(volatility), "Dias Úteis": int(business_days),
            "Preço Teórico Unit.": float(res['Preço Teórico']),
            "Delta Unit.":  float(res['Delta']),
            "Gamma Unit.":  float(res['Gamma']),
            "Vega Unit.":   float(res['Vega']),
            "Theta Unit.":  float(res['Theta']),
            "Rho Unit.":    float(res['Rho']),
            "Delta Total":  float(res['Delta'])  * float(quantity),
            "Gamma Total":  float(res['Gamma'])  * float(quantity),
            "Vega Total":   float(res['Vega'])   * float(quantity),
            "Theta Total":  float(res['Theta'])  * float(quantity),
            "Rho Total":    float(res['Rho'])    * float(quantity),
        }
    else:
        item = None
        st.markdown('<div class="box-warn">⚠️ Informe preço e strike para adicionar.</div>',
                    unsafe_allow_html=True)
    if item is not None:
        st.session_state.portfolio.append(item)
        st.success("Posição adicionada à estrutura.")

if update_clicked:
    if not st.session_state.portfolio:
        st.markdown('<div class="box-info">ℹ️ Estrutura vazia.</div>', unsafe_allow_html=True)
    else:
        updated, today2 = 0, date.today()
        for i, itm in enumerate(st.session_state.portfolio):
            tipo = itm.get('Tipo')
            qtd  = float(itm.get('Qtd', 0.0))
            if tipo == 'Ativo':
                st.session_state.portfolio[i]['Preço Teórico Unit.'] = float(spot_price)
                st.session_state.portfolio[i]['Delta Total'] = 1.0 * qtd
                updated += 1
            else:
                K, venc = itm.get('Strike'), itm.get('Vencimento')
                if venc is None or (isinstance(K, float) and np.isnan(K)):
                    continue
                dias = calculate_business_days(today2, venc)
                res  = black_scholes(float(spot_price), float(K), dias / 252.0,
                                     risk_free_rate, volatility, tipo)
                st.session_state.portfolio[i].update({
                    'Preço Teórico Unit.': float(res['Preço Teórico']),
                    'Delta Unit.': float(res['Delta']), 'Gamma Unit.': float(res['Gamma']),
                    'Vega Unit.':  float(res['Vega']),  'Theta Unit.': float(res['Theta']),
                    'Rho Unit.':   float(res['Rho']),   'Dias Úteis':  int(dias),
                    'Delta Total': float(res['Delta']) * qtd,
                    'Gamma Total': float(res['Gamma']) * qtd,
                    'Vega Total':  float(res['Vega'])  * qtd,
                    'Theta Total': float(res['Theta']) * qtd,
                    'Rho Total':   float(res['Rho'])   * qtd,
                })
                updated += 1
        st.success(f"{updated} posição(ões) recalculada(s).")

if clear_clicked:
    st.session_state.portfolio = []
    st.rerun()

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════
#  ⑥ ESTRUTURA DE OPÇÕES
# ═══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="sec-bar"><span class="ico">🗂️</span> Estrutura de Opções</div>',
            unsafe_allow_html=True)

if st.session_state.portfolio:
    df_p = pd.DataFrame(st.session_state.portfolio)
    display_cols = ["Tipo", "Strike", "Vencimento", "Qtd",
                    "Preço Teórico Unit.", "Delta Unit.", "Gamma Unit.",
                    "Theta Unit.", "Delta Total"]
    fmt = {
        "Strike": "{:.2f}", "Preço Teórico Unit.": "{:.4f}",
        "Delta Unit.": "{:.4f}", "Gamma Unit.": "{:.4f}",
        "Theta Unit.": "{:.4f}", "Delta Total": "{:.4f}",
    }
    st.dataframe(df_p[display_cols].style.format(fmt), use_container_width=True)

    st.markdown("**Gerenciar posições:**")
    for idx, itm in enumerate(st.session_state.portfolio):
        mc = st.columns([5, 1, 1])
        with mc[0]:
            sk = itm['Strike'] if not (isinstance(itm['Strike'], float) and np.isnan(itm['Strike'])) else 'N/A'
            vd = itm['Vencimento'] if itm.get('Vencimento') else 'N/A'
            st.markdown(
                f"<span style='color:#2e9e4f;font-weight:700'>#{idx+1}</span> &nbsp;"
                f"<b>{itm.get('Tipo')}</b> &nbsp;·&nbsp; "
                f"<span style='font-size:13px;color:#555'>"
                f"Qtd: {int(itm.get('Qtd',0))} &nbsp;|&nbsp; K: {sk} &nbsp;|&nbsp; Venc: {vd}</span>",
                unsafe_allow_html=True
            )
        with mc[1]:
            if st.button("Editar", key=f"edit_{idx}"):
                st.session_state['edit_idx'] = idx
        with mc[2]:
            if st.button("Excluir", key=f"del_{idx}"):
                st.session_state.portfolio.pop(idx)
                st.rerun()

    if st.session_state.get('edit_idx') is not None:
        ei = st.session_state['edit_idx']
        if 0 <= ei < len(st.session_state.portfolio):
            itm = st.session_state.portfolio[ei]
            st.markdown(
                f'<div class="sec-bar" style="font-size:14px;padding:8px 16px;">'
                f'<span class="ico">✏️</span> Editando posição #{ei+1} — {itm.get("Tipo")}</div>',
                unsafe_allow_html=True)
            ec1, ec2, ec3 = st.columns(3)
            with ec1:
                new_qty = st.number_input("Quantidade", value=float(itm.get('Qtd', 0.0)),
                                          step=1.0, key=f"eq_{ei}")
            with ec2:
                if itm.get('Tipo') in ['Call', 'Put']:
                    new_k = st.number_input("Strike", value=float(itm.get('Strike', spot_price)),
                                            step=0.01, format="%.2f", key=f"ek_{ei}")
                else:
                    new_k = itm.get('Strike')
            with ec3:
                if itm.get('Tipo') in ['Call', 'Put']:
                    cv = itm.get('Vencimento')
                    dv = cv if isinstance(cv, date) else date.today()
                    new_v = st.date_input("Vencimento", value=dv,
                                          min_value=date.today(), key=f"ev_{ei}")
                else:
                    new_v = itm.get('Vencimento')

            sc, cc = st.columns(2)
            with sc:
                if st.button("Salvar alterações", key=f"save_{ei}"):
                    st.session_state.portfolio[ei]['Qtd'] = float(new_qty)
                    if itm.get('Tipo') in ['Call', 'Put']:
                        st.session_state.portfolio[ei].update({'Strike': float(new_k), 'Vencimento': new_v})
                        d2 = calculate_business_days(date.today(), new_v)
                        r2 = black_scholes(float(spot_price), float(new_k), d2 / 252.0,
                                           risk_free_rate, volatility, itm.get('Tipo'))
                        st.session_state.portfolio[ei].update({
                            'Preço Teórico Unit.': float(r2['Preço Teórico']),
                            'Delta Unit.': float(r2['Delta']), 'Gamma Unit.': float(r2['Gamma']),
                            'Vega Unit.':  float(r2['Vega']),  'Theta Unit.': float(r2['Theta']),
                            'Rho Unit.':   float(r2['Rho']),   'Dias Úteis':  int(d2),
                            'Delta Total': float(r2['Delta']) * float(new_qty),
                            'Gamma Total': float(r2['Gamma']) * float(new_qty),
                            'Vega Total':  float(r2['Vega'])  * float(new_qty),
                            'Theta Total': float(r2['Theta']) * float(new_qty),
                            'Rho Total':   float(r2['Rho'])   * float(new_qty),
                        })
                    else:
                        st.session_state.portfolio[ei]['Delta Total'] = 1.0 * float(new_qty)
                    del st.session_state['edit_idx']
                    st.rerun()
            with cc:
                if st.button("Cancelar", key=f"cancel_{ei}"):
                    del st.session_state['edit_idx']
                    st.rerun()

    # ── Resumo de risco ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="sec-bar"><span class="ico">⚖️</span> Resumo de Risco — Gregas Totais</div>',
                unsafe_allow_html=True)

    df_full = pd.DataFrame(st.session_state.portfolio)
    td = float(df_full["Delta Total"].sum())
    tg = float(df_full["Gamma Total"].sum())
    tv = float(df_full["Vega Total"].sum())
    tt = float(df_full["Theta Total"].sum())
    tr = float(df_full["Rho Total"].sum())

    st.markdown(f"""
    <div class="result-row" style="border:none; padding-bottom:8px;">
      <div><span>Posições na estrutura:</span> <b>{len(st.session_state.portfolio)}</b></div>
    </div>
    <div class="risk-grid">
      <div class="risk-card">
        <div class="risk-label">Δ Delta Total</div>
        <div class="risk-value">{td:.4f}</div>
        <div class="risk-help">Var. p/ +R$1 no ativo</div>
      </div>
      <div class="risk-card">
        <div class="risk-label">Γ Gamma Total</div>
        <div class="risk-value">{tg:.4f}</div>
        <div class="risk-help">Aceleração do delta</div>
      </div>
      <div class="risk-card">
        <div class="risk-label">Θ Theta / d.u.</div>
        <div class="risk-value">{tt:.4f}</div>
        <div class="risk-help">Decaimento diário útil</div>
      </div>
      <div class="risk-card">
        <div class="risk-label">ν Vega / 1%</div>
        <div class="risk-value">{tv:.4f}</div>
        <div class="risk-help">Exposição a +1% na vol</div>
      </div>
      <div class="risk-card">
        <div class="risk-label">ρ Rho / 1%</div>
        <div class="risk-value">{tr:.4f}</div>
        <div class="risk-help">Exposição a +1% na taxa</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown(
        '<div class="box-info">ℹ️ &nbsp;Adicione opções ou o ativo para montar sua estrutura '
        'e visualizar o risco agregado.</div>',
        unsafe_allow_html=True
    )

# ── Notas ─────────────────────────────────────────────────────────────────
with st.expander("📝  Notas e metodologia"):
    st.markdown(f"""
**Black–Scholes (1973)** — modelo de precificação para opções europeias sem dividendos.

- **Dias úteis**: excluem fins de semana e feriados nacionais brasileiros (lib `holidays`).
- **Base**: 252 dias úteis/ano.
- **Theta** por dia útil; **Vega** e **Rho** por variação de 1%.
- **GARCH(1,1)**: α = {GARCH_ALPHA} · β = {GARCH_BETA} · γ = {GARCH_GAMMA} · janela init = {GARCH_N_INIT} obs.
  ω = σ²_LP × (1 − α − β). O redutor de {int(GARCH_REDUCER*100)}% gera estimativa conservadora para uso como vol de entrada.
- Dados via **Yahoo Finance** (`yfinance`). Verifique a disponibilidade do ticker.
    """)