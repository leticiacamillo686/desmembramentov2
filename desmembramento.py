
import math
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional

import pandas as pd
import streamlit as st

# ================== Utils ==================

def parse_br_date(s: str) -> Optional[date]:
    """Parse 'DD/MM/YYYY' to date. Returns None if invalid."""
    if not isinstance(s, str):
        return None
    s = s.strip()
    if not s:
        return None
    try:
        d, m, y = s.split("/")
        return date(int(y), int(m), int(d))
    except Exception:
        return None

def months_between_calendar_months(start_date: date, end_date: date) -> int:
    """
    Months between two dates considering only year/month (ignores days).
    Equivalent to difference between (YYYY, MM). Non-negative.
    """
    if start_date is None or end_date is None:
        return 0
    return max(0, (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month))

def adjusted_price(original: float, months: int, annual_rate: float) -> float:
    """Compound the original price by annual_rate over 'months' months."""
    if months <= 0 or annual_rate == 0:
        return float(original)
    factor = (1.0 + annual_rate) ** (months / 12.0)
    return float(original) * factor

def fmt_brl(x: float) -> str:
    return f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def add_months(d: date, months: int) -> date:
    """Add (or subtract) months preserving the day when possible. Here day is always 7, 18 or 27 -> always valid."""
    m = d.month - 1 + months
    y = d.year + m // 12
    m = m % 12 + 1
    # For safety, clamp day to month length
    from calendar import monthrange
    last_day = monthrange(y, m)[1]
    day = min(d.day, last_day)
    return date(y, m, day)

def last_boleto_info(cancel_date: date, due_day: int) -> Dict[str, date]:
    """
    Given a cancellation date and a due day (7, 18, 27),
    returns the service period [start, end] and the last 'boleto' (due) date,
    considering post-paid billing: we charge AFTER the service period ends.
    """
    D = int(due_day)
    # Determine the start of the current service period that includes cancel_date
    if cancel_date.day >= D:
        period_start = date(cancel_date.year, cancel_date.month, D)
    else:
        # previous month D
        period_start = add_months(date(cancel_date.year, cancel_date.month, D), -1)

    # The period ends the day before the next period starts
    next_due_date = add_months(period_start, 1)  # same day D in next month
    period_end = next_due_date - timedelta(days=1)
    # The last boleto will be exactly on next_due_date (post-paid)
    return {
        "period_start": period_start,
        "period_end": period_end,
        "last_boleto": next_due_date,
    }

# Allocation respecting floors (e.g., hospedagem mínima)
def allocate_with_floors(total: float, items: List[Dict[str, Any]]) -> List[float]:
    """
    Distribute 'total' across items proportionally to 'weight', but enforce per-item 'floor'.
    items: [{ 'weight': float, 'floor': float }, ...]
    Returns allocations list with same order.
    Assumes total >= sum(floors). If any weight is zero, those items only receive floor.
    """
    n = len(items)
    floors = [max(0.0, float(it.get("floor", 0.0))) for it in items]
    weights = [max(0.0, float(it.get("weight", 0.0))) for it in items]
    min_required = sum(floors)
    total = max(total, min_required)

    # Initialize allocations at floors
    alloc = floors[:]
    remaining = total - sum(alloc)

    # If no remaining or all weights are zero, return floors (since that's the minimum feasible)
    if remaining <= 1e-9 or sum(weights) <= 0:
        return alloc

    # Proportional allocation on the remaining amount
    weight_sum = sum(weights)
    for i in range(n):
        if weights[i] > 0:
            alloc[i] += remaining * (weights[i] / weight_sum)

    # Normalize tiny FP residuals
    diff = total - sum(alloc)
    if abs(diff) > 1e-6:
        # add/subtract residual to first positive-weight item
        for i in range(n):
            if weights[i] > 0:
                alloc[i] += diff
                break

    return alloc

# ================== Streamlit App ==================

st.set_page_config(page_title="Desmembramento de Mensalidade", page_icon="🧮", layout="wide")
st.title("🧮 Desmembramento de Mensalidade de Clientes")

with st.sidebar:
    st.header("Passo 1 — Dados do cliente")
    mensalidade_atual = st.number_input("Mensalidade atual (Omie) — R$", min_value=0.0, step=0.01, value=160.00, format="%.2f")

    # DD/MM/AAAA text field for cancellation date
    default_cancel = date.today().strftime("%d/%m/%Y")
    cancel_str = st.text_input("Data de solicitação do cancelamento (DD/MM/AAAA)", value=default_cancel, help="Ex.: 13/08/2025")
    parsed_cancel = parse_br_date(cancel_str)
    if not parsed_cancel:
        st.error("Informe a data de cancelamento em **DD/MM/AAAA** (ex.: 13/08/2025).")
        st.stop()

    vencimento = st.selectbox("Dia de vencimento do boleto", options=[7, 18, 27], index=2, help="O boleto é pós-pago. Ex.: venc. 27/08 cobre 27/07–26/08.")

    st.header("Parâmetros de cálculo")
    reajuste_anual = st.number_input("Reajuste médio anual (ex.: 10% = 0,10)", min_value=0.0, step=0.01, value=0.10, format="%.2f")
    margem_negociacao = st.number_input("Margem de negociação sobre a parte negociável (ex.: 30% = 0,30)", min_value=0.0, step=0.01, value=0.30, format="%.2f")

    st.subheader("Pisos de hospedagem (R$)")
    piso_basic = st.number_input("Hospedagem Basic", min_value=0.0, step=1.0, value=70.0, format="%.2f")
    piso_standard = st.number_input("Hospedagem Standard", min_value=0.0, step=1.0, value=90.0, format="%.2f")
    piso_pro = st.number_input("Hospedagem Pro", min_value=0.0, step=1.0, value=120.0, format="%.2f")

# Last boleto information using parsed_cancel
info_boleto = last_boleto_info(parsed_cancel, int(vencimento))
colx, coly, colz = st.columns(3)
with colx:
    st.metric("📅 Último boleto", info_boleto["last_boleto"].strftime("%d/%m/%Y"))
with coly:
    st.metric("Período de serviço (início)", info_boleto["period_start"].strftime("%d/%m/%Y"))
with colz:
    st.metric("Período de serviço (fim)", info_boleto["period_end"].strftime("%d/%m/%Y"))
st.caption("O boleto é **pós-pago**: a data acima cobra exatamente o período indicado (do início ao dia anterior).")

st.markdown("---")
st.subheader("Passo 2 — Informar serviços do cliente")
st.markdown("Adicione **apenas** os serviços que o cliente possui. Datas no formato **DD/MM/AAAA**.")

default_rows = pd.DataFrame([
    {"Serviço": "E-mail", "Tipo": "Email", "Qtd": 1, "Preço_original": 11.00, "Contratação (DD/MM/AAAA)": "01/01/2020", "Hospedagem": ""},
    {"Serviço": "Site", "Tipo": "Site", "Qtd": 1, "Preço_original": 59.90, "Contratação (DD/MM/AAAA)": "01/01/2020", "Hospedagem": "Basic"},
])

edited = st.data_editor(
    default_rows,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Serviço": st.column_config.TextColumn("Serviço"),
        "Tipo": st.column_config.SelectboxColumn("Tipo", options=["Email", "Site", "Ferramenta"]),
        "Qtd": st.column_config.NumberColumn("Qtd", min_value=1, step=1),
        "Preço_original": st.column_config.NumberColumn("Preço original (R$)", min_value=0.0, step=0.01, format="%.2f"),
        "Contratação (DD/MM/AAAA)": st.column_config.TextColumn("Contratação (DD/MM/AAAA)", help="Ex.: 15/01/2020"),
        "Hospedagem": st.column_config.SelectboxColumn("Hospedagem", options=["", "Basic", "Standard", "Pro"]),
    },
    hide_index=True,
    key="editor_servicos"
)

st.markdown("---")
if st.button("Passo 3 — Calcular desmembramento", type="primary"):
    # ======= Construção dos serviços com reajuste =======
    annual = float(reajuste_anual)
    results = []
    non_neg_total = 0.0
    invalid_rows = []

    for idx, row in edited.iterrows():
        try:
            servico = str(row.get("Serviço", f"Serviço {idx+1}"))
            tipo = str(row.get("Tipo", "Ferramenta")).strip()
            qtd = int(row.get("Qtd", 1) or 1)
            preco_orig = float(row.get("Preço_original", 0.0) or 0.0)
            contrat_str = str(row.get("Contratação (DD/MM/AAAA)", "")).strip()
            contrat_date = parse_br_date(contrat_str)
            if not contrat_date:
                invalid_rows.append(idx + 1)  # 1-based for user
                continue
            hosped = str(row.get("Hospedagem", "")).strip()
        except Exception:
            invalid_rows.append(idx + 1)
            continue

        months = months_between_calendar_months(contrat_date, parsed_cancel)
        unit_aj = adjusted_price(preco_orig, months, annual)
        sub_aj = unit_aj * qtd

        # Não negociável? (Email)
        nao_neg = (tipo.lower() == "email")

        # Piso (só para Site)
        piso = 0.0
        if tipo.lower() == "site":
            if hosped.lower() == "basic":
                piso = piso_basic * qtd
            elif hosped.lower() == "standard":
                piso = piso_standard * qtd
            elif hosped.lower() == "pro":
                piso = piso_pro * qtd

        results.append({
            "Serviço": servico,
            "Tipo": tipo,
            "Qtd": qtd,
            "Preço original (R$)": preco_orig,
            "Contratação (DD/MM/AAAA)": contrat_date.strftime("%d/%m/%Y"),
            "Hospedagem": hosped,
            "Meses desde contratação": months,
            "Valor reajustado unit. (R$)": unit_aj,
            "Subtotal reajustado (R$)": sub_aj,
            "Não negociável?": "Sim" if nao_neg else "Não",
            "Piso (se aplicável) (R$)": piso,
            "Alocado final (R$)": 0.0,
            "Preço final unit. (R$)": 0.0,
        })

    if invalid_rows:
        st.error(f"As linhas {', '.join(map(str, invalid_rows))} têm **data de contratação inválida**. Use DD/MM/AAAA.")
        st.stop()

    # ======= Partes negociável x não negociável =======
    # Parte não negociável: soma dos Emails reajustados
    for r in results:
        if r["Não negociável?"] == "Sim":
            non_neg_total += r["Subtotal reajustado (R$)"]

    negotiable_base_current = mensalidade_atual - non_neg_total
    negotiable_base_current = max(0.0, negotiable_base_current)

    max_discount_value = margem_negociacao * negotiable_base_current
    target_negociable_subtotal = negotiable_base_current - max_discount_value

    # ======= Pisos (sites) =======
    floors = []
    weights = []
    negotiable_indices = []
    for i, r in enumerate(results):
        if r["Não negociável?"] == "Não":
            negotiable_indices.append(i)
            floors.append(float(r["Piso (se aplicável) (R$)"]))
            weights.append(max(0.0, float(r["Subtotal reajustado (R$)"])))

    sum_floors_neg = sum(floors)
    # alvo negociável nunca pode ficar abaixo dos pisos e nem acima da base negociável
    target_negociable_subtotal = min(max(target_negociable_subtotal, sum_floors_neg), negotiable_base_current)

    warnings = []
    if negotiable_base_current < sum_floors_neg - 1e-9:
        warnings.append(
            f"A mensalidade atual (R$ {mensalidade_atual:.2f}) é **inferior** à soma dos pisos dos serviços negociáveis "
            f"(R$ {sum_floors_neg:.2f}) somada à parte não negociável (R$ {non_neg_total:.2f}). "
            "Não é possível aplicar o desconto desejado sem violar pisos."
        )

    # ======= Alocação =======
    allocations = []
    if len(negotiable_indices) > 0:
        items = [{"weight": weights[k], "floor": floors[k]} for k in range(len(negotiable_indices))]
        allocations = allocate_with_floors(target_negociable_subtotal, items)

    # Preencher retorno
    for pos, ires in enumerate(negotiable_indices):
        results[ires]["Alocado final (R$)"] = allocations[pos]

    for i, r in enumerate(results):
        if r["Não negociável?"] == "Sim":
            results[i]["Alocado final (R$)"] = r["Subtotal reajustado (R$)"]

    for i, r in enumerate(results):
        qtd = max(1, int(r["Qtd"]))
        final_sub = float(r["Alocado final (R$)"])
        results[i]["Preço final unit. (R$)"] = final_sub / qtd if qtd > 0 else final_sub

    df_res = pd.DataFrame(results)

    # Totals e indicadores
    final_total = df_res["Alocado final (R$)"].sum()
    achieved_discount = 0.0
    if negotiable_base_current > 1e-9:
        achieved_discount = (negotiable_base_current - target_negociable_subtotal) / negotiable_base_current

    st.markdown("### Resultado do desmembramento")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mensalidade atual (Omie)", fmt_brl(mensalidade_atual))
        st.metric("Parte não negociável", fmt_brl(non_neg_total))
    with col2:
        st.metric("Base negociável (atual - não negociável)", fmt_brl(negotiable_base_current))
        st.metric("Máx. desconto (margem)", fmt_brl(max_discount_value))
    with col3:
        st.metric("Negociável sugerido (após desconto/pisos)", fmt_brl(target_negociable_subtotal))
        st.metric("Mensalidade sugerida (final)", fmt_brl(final_total))

    st.markdown(f"**Desconto efetivo sobre a parte negociável:** {achieved_discount*100:.2f}%")
    if warnings:
        for w in warnings:
            st.warning(w)

    st.markdown("### Detalhamento por serviço")
    st.dataframe(
        df_res[[
            "Serviço",
            "Tipo",
            "Qtd",
            "Hospedagem",
            "Preço original (R$)",
            "Contratação (DD/MM/AAAA)",
            "Meses desde contratação",
            "Valor reajustado unit. (R$)",
            "Subtotal reajustado (R$)",
            "Não negociável?",
            "Piso (se aplicável) (R$)",
            "Alocado final (R$)",
            "Preço final unit. (R$)",
        ]].rename(columns={
            "Preço original (R$)": "Preço original",
            "Valor reajustado unit. (R$)": "Reajustado unit.",
            "Subtotal reajustado (R$)": "Subtotal reajustado",
            "Piso (se aplicável) (R$)": "Piso",
            "Alocado final (R$)": "Alocado final",
            "Preço final unit. (R$)": "Preço final unit.",
        }),
        use_container_width=True
    )

    csv = df_res.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Baixar detalhamento (CSV)", data=csv, file_name="desmembramento_detalhamento.csv", mime="text/csv")

st.markdown("---")
with st.expander("📌 Regras e formato de datas"):
    st.markdown("""
- **Todas as datas** devem ser informadas e são exibidas no formato **DD/MM/AAAA** (ex.: 13/08/2025).
- Para a **Contratação**, a diferença de meses é calculada levando em conta apenas **ano/mês** (o dia é desconsiderado), mantendo a mesma lógica anterior.
- O **último boleto** é calculado com base no **vencimento** (7, 18 ou 27) e no **pós-pagamento** (cobra-se após o fim do período).
""")

st.caption("Versão Streamlit — datas 100% em DD/MM/AAAA (entrada e exibição).")
