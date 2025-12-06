
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import date, timedelta
import holidays

# --- Configuração da Página ---
st.set_page_config(page_title="Calculadora Black–Scholes", layout="wide")

# --- Funções Auxiliares ---

def get_stock_price(ticker: str):
    """Busca o preço atual do ativo no Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period="1d")
        if not history.empty:
            return float(history['Close'].iloc[-1])
        return None
    except Exception:
        return None

def calculate_business_days(start_date: date, end_date: date):
    """Calcula dias úteis entre duas datas (exclui fins de semana e feriados brasileiros)."""
    br_holidays = holidays.Brazil(years=list(range(start_date.year, end_date.year + 1)))
    business_days = 0
    current_date = start_date
    while current_date < end_date:
        if current_date.weekday() < 5 and current_date not in br_holidays:
            business_days += 1
        current_date += timedelta(days=1)
    return business_days

def black_scholes(S, K, T_years, r_annual, sigma_annual, option_type='Call'):
    """
    Calcula preço teórico e gregas pelo modelo Black–Scholes.
    - S: preço do ativo
    - K: strike
    - T_years: tempo até vencimento em anos (dias úteis/250)
    - r_annual: taxa de juros anual (úteis) em decimal
    - sigma_annual: volatilidade anual em decimal
    - option_type: 'Call' ou 'Put'
    Observações de formato BR:
    - Theta reportado por dia útil (divide por 250).
    - Vega e Rho reportados por variação de 1% (divide por 100).
    """
    # Tratamento para vencimento imediato
    if T_years <= 0:
        intrinsic = max(0.0, S - K) if option_type == 'Call' else max(0.0, K - S)
        return {
            'Preço Teórico': intrinsic,
            'Delta': 0.0, 'Gamma': 0.0, 'Vega': 0.0, 'Theta': 0.0, 'Rho': 0.0
        }

    d1 = (np.log(S / K) + (r_annual + 0.5 * sigma_annual ** 2) * T_years) / (sigma_annual * np.sqrt(T_years))
    d2 = d1 - sigma_annual * np.sqrt(T_years)

    if option_type == 'Call':
        price = S * norm.cdf(d1) - K * np.exp(-r_annual * T_years) * norm.cdf(d2)
        delta = norm.cdf(d1)
        rho = K * T_years * np.exp(-r_annual * T_years) * norm.cdf(d2)
        theta = (
            - (S * norm.pdf(d1) * sigma_annual) / (2 * np.sqrt(T_years))
            - r_annual * K * np.exp(-r_annual * T_years) * norm.cdf(d2)
        )
    else:  # Put
        price = K * np.exp(-r_annual * T_years) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1.0
        rho = -K * T_years * np.exp(-r_annual * T_years) * norm.cdf(-d2)
        theta = (
            - (S * norm.pdf(d1) * sigma_annual) / (2 * np.sqrt(T_years))
            + r_annual * K * np.exp(-r_annual * T_years) * norm.cdf(-d2)
        )

    gamma = norm.pdf(d1) / (S * sigma_annual * np.sqrt(T_years))
    vega = S * norm.pdf(d1) * np.sqrt(T_years)

    # Ajustes de formato BR
    theta_daily = theta / 250.0          # por dia útil
    vega_per_1pct = vega / 100.0         # por 1% de vol
    rho_per_1pct = rho / 100.0           # por 1% de taxa

    return {
        'Preço Teórico': price,
        'Delta': delta,
        'Gamma': gamma,
        'Vega': vega_per_1pct,
        'Theta': theta_daily,
        'Rho': rho_per_1pct
    }

# --- Estado Inicial ---
if 'spot_price' not in st.session_state:
    st.session_state.spot_price = 0.0
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []

# --- Interface do Usuário ---

st.title("📊 Calculadora Black–Scholes e Estruturas (BR)")

# 1. Configuração do Ativo Base
st.sidebar.header("1. Ativo base")
ticker_input = st.sidebar.text_input("Ticker (ex: PETR4.SA, VALE3.SA)", value="BOVA11.SA").upper()

# Botão para atualizar preço via Yahoo
if st.sidebar.button("Atualizar preço (Yahoo Finance)"):
    fetched_price = get_stock_price(ticker_input)
    if fetched_price is not None:
        st.session_state.spot_price = fetched_price
        st.sidebar.success(f"Preço obtido: R$ {fetched_price:.2f}")
    else:
        st.sidebar.error("Não foi possível obter o preço. Verifique o ticker.")

# Input manual (override)
spot_price = st.sidebar.number_input(
    "Preço atual do ativo (S)",
    value=float(st.session_state.spot_price),
    format="%.2f",
    step=0.01,
    help="Por padrão vem do Yahoo, mas você pode editar."
)

# 2. Parâmetros de Mercado
st.sidebar.header("2. Parâmetros de mercado")
risk_free_rate_pct = st.sidebar.number_input(
    "Taxa de juros anual (%) [dias úteis ~250]",
    value=15.0, step=0.25, format="%.2f"
)
risk_free_rate = risk_free_rate_pct / 100.0

volatility_pct = st.sidebar.number_input(
    "Volatilidade implícita anual (%)",
    value=15.0, step=0.50, format="%.2f"
)
volatility = volatility_pct / 100.0

# 3. Inputs de opção
st.subheader("Montar estrutura / calcular opção")
col1, col2, col3, col4 = st.columns(4)

with col1:
    option_type = st.selectbox("Tipo", ["Call", "Put", "Ativo"])
    quantity = st.number_input(
        "Quantidade (pode ser negativa)",
        value=1, step=1,
        help="Positivo = compra, Negativo = venda."
    )

with col2:
    # Strike só para opções
    if option_type in ["Call", "Put"]:
        strike_price = st.number_input("Strike (K)", value=spot_price if spot_price > 0 else 10.0, step=0.01, format="%.2f")
    else:
        strike_price = None
        st.caption("Strike não aplicável para o ativo.")

with col3:
    today = date.today()
    expiration_date = st.date_input("Vencimento", value=today + timedelta(days=30), min_value=today)

# Cálculo de dias úteis e tempo em anos (úteis/250)
business_days = calculate_business_days(today, expiration_date)
time_to_maturity = business_days / 250.0

with col4:
    st.metric(label="Dias úteis até o vencimento", value=business_days)

st.markdown("---")

# Exibir gregas unitárias com base nos inputs atuais
st.subheader("Gregas e preço teórico (unitário)")
if option_type in ["Call", "Put"] and spot_price > 0 and strike_price is not None:
    sim = black_scholes(spot_price, strike_price, time_to_maturity, risk_free_rate, volatility, option_type)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Preço Teórico", f"R$ {sim['Preço Teórico']:.2f}")
    c2.metric("Delta", f"{sim['Delta']:.2f}")
    c3.metric("Gamma", f"{sim['Gamma']:.2f}")
    c4.metric("Theta (dia útil)", f"{sim['Theta']:.2f}")
    c5.metric("Vega (por 1%)", f"{sim['Vega']:.2f}")
    c6.metric("Rho (por 1%)", f"{sim['Rho']:.2f}")
elif option_type == "Ativo" and spot_price > 0:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Preço (Ativo)", f"R$ {spot_price:.2f}")
    # Ativo objeto: Delta = 1 para qtd > 0; -1 para qtd < 0. Demais gregas = 0.
    delta_unit = 1.0
    c2.metric("Delta", f"{delta_unit:.2f}")
    c3.metric("Gamma", "0.00")
    c4.metric("Theta (dia útil)", "0.00")
    c5.metric("Vega (por 1%)", "0.00")
    c6.metric("Rho (por 1%)", "0.00")
else:
    st.info("Informe preço, tipo e strike (para opções) para ver as gregas.")

# Botões para estrutura
add_col, update_col, clear_col = st.columns([1, 1, 3])

if add_col.button("Adicionar à estrutura"):
    if option_type == "Ativo":
        # Delta unit depende do sinal da quantidade
        delta_unit = 1.0
        item = {
            "Tipo": "Ativo",
            "Strike": np.nan,
            "Vencimento": None,
            "Qtd": float(quantity),
            "IV": np.nan,
            "Dias Úteis": np.nan,
            "Preço Teórico Unit.": float(spot_price),
            "Delta Unit.": float(delta_unit),
            "Gamma Unit.": 0.0,
            "Vega Unit.": 0.0,
            "Theta Unit.": 0.0,
            "Rho Unit.": 0.0,
            # Totais
            "Delta Total": delta_unit * quantity,
            "Gamma Total": 0.0,
            "Vega Total": 0.0,
            "Theta Total": 0.0,
            "Rho Total": 0.0,
        }
    else:
        res = black_scholes(spot_price, strike_price, time_to_maturity, risk_free_rate, volatility, option_type)
        item = {
            "Tipo": option_type,
            "Strike": float(strike_price),
            "Vencimento": expiration_date,
            "Qtd": float(quantity),
            "IV": float(volatility),
            "Dias Úteis": int(business_days),
            "Preço Teórico Unit.": float(res['Preço Teórico']),
            "Delta Unit.": float(res['Delta']),
            "Gamma Unit.": float(res['Gamma']),
            "Vega Unit.": float(res['Vega']),
            "Theta Unit.": float(res['Theta']),
            "Rho Unit.": float(res['Rho']),
            # Totais
            "Delta Total": float(res['Delta']) * float(quantity),
            "Gamma Total": float(res['Gamma']) * float(quantity),
            "Vega Total": float(res['Vega']) * float(quantity),
            "Theta Total": float(res['Theta']) * float(quantity),
            "Rho Total": float(res['Rho']) * float(quantity),
        }

    st.session_state.portfolio.append(item)
    st.success("Item adicionado à estrutura!")

# Botão para atualizar/recalcular toda a estrutura com os inputs atuais
if update_col.button("Atualizar estrutura"):
    if not st.session_state.portfolio:
        st.info("Nada para atualizar — a estrutura está vazia.")
    else:
        updated = 0
        today = date.today()
        for i, itm in enumerate(st.session_state.portfolio):
            tipo = itm.get('Tipo')
            qtd = float(itm.get('Qtd', 0.0))
            if tipo == 'Ativo':
                # atualizar preço do ativo e delta total
                st.session_state.portfolio[i]['Preço Teórico Unit.'] = float(spot_price)
                st.session_state.portfolio[i]['Delta Unit.'] = 1.0
                st.session_state.portfolio[i]['Delta Total'] = 1.0 * qtd
                # demais gregas permanecem 0
                updated += 1
            else:
                # opção: recalc com strike e vencimento armazenados
                K = itm.get('Strike')
                venc = itm.get('Vencimento')
                if venc is None or (isinstance(K, float) and np.isnan(K)):
                    # não é possível recalcular
                    continue
                # recalcular dias úteis e tempo até vencimento
                dias = calculate_business_days(today, venc)
                T = dias / 250.0
                res = black_scholes(float(spot_price), float(K), T, risk_free_rate, volatility, tipo)
                st.session_state.portfolio[i]['Preço Teórico Unit.'] = float(res['Preço Teórico'])
                st.session_state.portfolio[i]['Delta Unit.'] = float(res['Delta'])
                st.session_state.portfolio[i]['Gamma Unit.'] = float(res['Gamma'])
                st.session_state.portfolio[i]['Vega Unit.'] = float(res['Vega'])
                st.session_state.portfolio[i]['Theta Unit.'] = float(res['Theta'])
                st.session_state.portfolio[i]['Rho Unit.'] = float(res['Rho'])
                st.session_state.portfolio[i]['Dias Úteis'] = int(dias)
                # Totais
                st.session_state.portfolio[i]['Delta Total'] = float(res['Delta']) * qtd
                st.session_state.portfolio[i]['Gamma Total'] = float(res['Gamma']) * qtd
                st.session_state.portfolio[i]['Vega Total'] = float(res['Vega']) * qtd
                st.session_state.portfolio[i]['Theta Total'] = float(res['Theta']) * qtd
                st.session_state.portfolio[i]['Rho Total'] = float(res['Rho']) * qtd
                updated += 1

        st.success(f"Estrutura atualizada ({updated} posições recalculadas).")

if clear_col.button("Limpar estrutura"):
    st.session_state.portfolio = []
    st.rerun()

# --- Exibição da Estrutura ---
if st.session_state.portfolio:
    st.write("""
        <style>
        .small-font-table td, .small-font-table th {font-size: 10px !important;}
        .element-container .metric-value {font-size: 10px !important;}
        </style>
    """, unsafe_allow_html=True)
    st.write("### Estrutura atual")
    df_portfolio = pd.DataFrame(st.session_state.portfolio)

    display_cols = [
        "Tipo", "Strike", "Vencimento", "Qtd",
        "Preço Teórico Unit.", "Delta Unit.", "Gamma Unit.", "Theta Unit.",
        "Delta Total"
    ]

    # Formatação apenas em colunas numéricas para evitar erro de formato
    fmt = {
        "Strike": "{:.2f}",
        "Preço Teórico Unit.": "{:.2f}",
        "Delta Unit.": "{:.2f}",
        "Gamma Unit.": "{:.2f}",
        "Theta Unit.": "{:.2f}",
        "Delta Total": "{:.2f}",
    }

    st.dataframe(
        df_portfolio[display_cols].style.format(fmt).set_table_attributes('class="small-font-table"'),
        use_container_width=True
    )

    # --- Gerenciar posições: Editar / Excluir ---
    st.markdown("#### Gerenciar posições")
    for idx, itm in enumerate(st.session_state.portfolio):
        tipo = itm.get('Tipo')
        cols = st.columns([3, 1, 1])
        with cols[0]:
            strike_display = itm['Strike'] if not (isinstance(itm['Strike'], float) and np.isnan(itm['Strike'])) else 'N/A'
            venc_display = itm['Vencimento'] if itm.get('Vencimento') is not None else 'N/A'
            st.write(f"**{idx+1}. {tipo}** — Qtd: {int(itm.get('Qtd',0))} | Strike: {strike_display} | Venc: {venc_display}")
        with cols[1]:
            if st.button('Editar', key=f'edit_btn_{idx}'):
                st.session_state['edit_idx'] = idx
        with cols[2]:
            if st.button('Excluir', key=f'del_btn_{idx}'):
                st.session_state.portfolio.pop(idx)
                st.success(f"Posição {idx+1} excluída.")
                st.rerun()

    # Formulário de edição quando um índice está em edição
    if st.session_state.get('edit_idx') is not None:
        edit_i = st.session_state['edit_idx']
        # protect against stale index
        if 0 <= edit_i < len(st.session_state.portfolio):
            itm = st.session_state.portfolio[edit_i]
            st.markdown(f"### Editando posição {edit_i+1} ({itm.get('Tipo')})")
            ecol1, ecol2, ecol3 = st.columns(3)
            with ecol1:
                new_qty = st.number_input('Quantidade', value=float(itm.get('Qtd',0.0)), step=1.0, key=f'qty_edit_{edit_i}')
            with ecol2:
                if itm.get('Tipo') in ['Call','Put']:
                    new_strike = st.number_input('Strike', value=float(itm.get('Strike', spot_price)), step=0.01, format='%.2f', key=f'strike_edit_{edit_i}')
                else:
                    new_strike = itm.get('Strike')
            with ecol3:
                if itm.get('Tipo') in ['Call','Put']:
                    # convert stored vencimento to date if necessary
                    current_venc = itm.get('Vencimento')
                    try:
                        default_venc = current_venc if isinstance(current_venc, date) else (current_venc)
                    except Exception:
                        default_venc = date.today()
                    new_venc = st.date_input('Vencimento', value=default_venc, min_value=date.today(), key=f'venc_edit_{edit_i}')
                else:
                    new_venc = itm.get('Vencimento')

            save_col, cancel_col = st.columns([1,1])
            with save_col:
                if st.button('Salvar alterações', key=f'save_edit_{edit_i}'):
                    # apply changes
                    st.session_state.portfolio[edit_i]['Qtd'] = float(new_qty)
                    if itm.get('Tipo') in ['Call','Put']:
                        st.session_state.portfolio[edit_i]['Strike'] = float(new_strike)
                        st.session_state.portfolio[edit_i]['Vencimento'] = new_venc
                        # recalc days and greeks
                        dias = calculate_business_days(date.today(), new_venc)
                        T = dias / 250.0
                        res = black_scholes(float(spot_price), float(new_strike), T, risk_free_rate, volatility, itm.get('Tipo'))
                        st.session_state.portfolio[edit_i]['Preço Teórico Unit.'] = float(res['Preço Teórico'])
                        st.session_state.portfolio[edit_i]['Delta Unit.'] = float(res['Delta'])
                        st.session_state.portfolio[edit_i]['Gamma Unit.'] = float(res['Gamma'])
                        st.session_state.portfolio[edit_i]['Vega Unit.'] = float(res['Vega'])
                        st.session_state.portfolio[edit_i]['Theta Unit.'] = float(res['Theta'])
                        st.session_state.portfolio[edit_i]['Rho Unit.'] = float(res['Rho'])
                        st.session_state.portfolio[edit_i]['Dias Úteis'] = int(dias)
                        # totals
                        st.session_state.portfolio[edit_i]['Delta Total'] = float(res['Delta']) * float(new_qty)
                        st.session_state.portfolio[edit_i]['Gamma Total'] = float(res['Gamma']) * float(new_qty)
                        st.session_state.portfolio[edit_i]['Vega Total'] = float(res['Vega']) * float(new_qty)
                        st.session_state.portfolio[edit_i]['Theta Total'] = float(res['Theta']) * float(new_qty)
                        st.session_state.portfolio[edit_i]['Rho Total'] = float(res['Rho']) * float(new_qty)
                    else:
                        # ativo: only quantity impacts totals
                        st.session_state.portfolio[edit_i]['Delta Total'] = 1.0 * float(new_qty)
                    # exit edit mode
                    del st.session_state['edit_idx']
                    st.success(f"Posição {edit_i+1} atualizada.")
                    st.rerun()
            with cancel_col:
                if st.button('Cancelar', key=f'cancel_edit_{edit_i}'):
                    del st.session_state['edit_idx']
                    st.rerun()

    # --- Resumo de Risco da Estrutura ---
    st.write("### Resumo de risco (gregas totais)")
    total_delta = float(df_portfolio["Delta Total"].sum())
    total_gamma = float(df_portfolio["Gamma Total"].sum())
    total_vega = float(df_portfolio["Vega Total"].sum())
    total_theta = float(df_portfolio["Theta Total"].sum())
    total_rho = float(df_portfolio["Rho Total"].sum())

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Delta Total (R$)", f"{total_delta:.2f}", help="Variação estimada para +R$1,00 no ativo.")
    m2.metric("Gamma Total", f"{total_gamma:.2f}", help="Aceleração do delta.")
    m3.metric("Theta Total (por dia útil)", f"{total_theta:.2f}", help="Decaimento diário (útil).")
    m4.metric("Vega Total (por 1%)", f"{total_vega:.2f}", help="Exposição à variação de 1% na vol.")
    m5.metric("Rho Total (por 1%)", f"{total_rho:.2f}", help="Exposição à variação de 1% na taxa.")

else:
    st.info("Adicione opções ou o ativo para montar sua estrutura e ver o delta total.")
