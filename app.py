# app.py — Pricing Importaciones (MJ) v2.0
# Cambios v2.0:
# 1) Landed CLP unit SIN IVA (separa IVA importación y IVA servicios)
# 2) Fórmula de pricing por defecto: (Costo * 1,7) + IVA  => Costo * 1.7 * (1+IVA)
# 3) Precios CLP sin decimales
# 4) Redondeos siempre terminan en "900" (…900, …1900, …2900, etc.)
# 5) Resumen final: costo total carga + venta total por escenario + margen % con y sin IVA

import io
import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Pricing Importaciones — Music Jungle (v2.0)", layout="wide")

# -----------------------------
# Constantes / Defaults
# -----------------------------
FORMATS = ["CD", "Cassette", "LP simple", "LP doble", "Box"]
ROTATIONS = ["Alta", "Media", "Baja"]
MARKETS = ["Price-taker", "Price-maker"]

DEFAULT_RATIOS = {
    "CD": 0.2,
    "Cassette": 0.25,
    "LP simple": 1.0,
    "LP doble": 1.8,
    "Box": 3.2,
}

# -----------------------------
# Helpers
# -----------------------------
def normalize_money(x):
    """Parse '14,95' or '14.95' or '1.234,56' etc. -> float."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "":
        return np.nan
    s = re.sub(r"[^\d,.\-]", "", s)
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    else:
        if "," in s:
            s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except:
        return np.nan

def parse_pasted_table(text: str) -> pd.DataFrame:
    """
    Acepta TSV (recomendado) o CSV.
    Columnas esperadas:
      Origen, Producto, Formato, Qty, Moneda, Costo unit, Rotación, Mercado
    Mínimas:
      Producto, Formato, Qty, Moneda, Costo unit
    """
    text = (text or "").strip()
    if not text:
        return pd.DataFrame()

    # TSV primero
    try:
        df = pd.read_csv(io.StringIO(text), sep="\t", dtype=str)
        if df.shape[1] <= 1:
            raise ValueError("Not TSV")
    except Exception:
        df = pd.read_csv(io.StringIO(text), sep=",", dtype=str)

    df.columns = [c.strip() for c in df.columns]

    # Map nombres comunes
    colmap = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ["origen", "origin"]:
            colmap[c] = "Origen"
        elif cl in ["producto", "product", "artist-title", "artist - title", "title"]:
            colmap[c] = "Producto"
        elif cl in ["formato", "format", "configuration"]:
            colmap[c] = "Formato"
        elif cl in ["qty", "cantidad", "quantity"]:
            colmap[c] = "Qty"
        elif cl in ["moneda", "currency"]:
            colmap[c] = "Moneda"
        elif cl in ["costo unit", "costo unitario", "unit cost", "net price", "nett.price", "net_price"]:
            colmap[c] = "Costo unit"
        elif cl in ["rotación", "rotacion", "rotation"]:
            colmap[c] = "Rotación"
        elif cl in ["mercado", "market"]:
            colmap[c] = "Mercado"

    df = df.rename(columns=colmap)

    # Asegurar columnas
    for needed in ["Producto", "Formato", "Qty", "Moneda", "Costo unit"]:
        if needed not in df.columns:
            df[needed] = ""

    if "Origen" not in df.columns:
        df["Origen"] = "Importado"
    if "Rotación" not in df.columns:
        df["Rotación"] = ""
    if "Mercado" not in df.columns:
        df["Mercado"] = ""

    df["Origen"] = df["Origen"].fillna("Importado").replace("", "Importado")
    df["Producto"] = df["Producto"].fillna("").astype(str).str.strip()
    df["Formato"] = df["Formato"].fillna("").astype(str).str.strip()
    df["Moneda"] = df["Moneda"].fillna("").astype(str).str.strip().str.upper()

    df["Qty"] = df["Qty"].apply(normalize_money).fillna(0).astype(int)
    df["Costo unit"] = df["Costo unit"].apply(normalize_money)

    def std_fmt(x):
        s = str(x).strip().upper()
        if s in ["LP", "1LP", "VINYL", "VINILO"]:
            return "LP simple"
        if s in ["2-LP", "2LP", "2 LP", "2XLP", "DOUBLE LP", "3-LP", "3LP", "3 LP", "3XLP", "TRIPLE LP"]:
            return "LP doble"
        if s in ["CD", "1CD", "DISC"]:
            return "CD"
        if s in ["2-CD", "2CD", "2 CD", "2XCD", "DOUBLE CD", "3-CD", "3CD", "3 CD", "3XCD"]:
            return "CD"
        if s in ["CASSETTE", "TAPE"]:
            return "Cassette"
        if s in ["BOX", "BOXSET"]:
            return "Box"
        if x in FORMATS:
            return x
        return x if x else "LP simple"

    df["Formato"] = df["Formato"].apply(std_fmt)
    df["Rotación"] = df["Rotación"].astype(str).str.strip()
    df["Mercado"] = df["Mercado"].astype(str).str.strip()
    return df

def fx_to_clp(currency: str, amount: float, usdclp: float, eurclp: float) -> float:
    if pd.isna(amount):
        return np.nan
    c = (currency or "").upper().strip()
    if c == "USD":
        return float(amount) * float(usdclp)
    if c == "EUR":
        return float(amount) * float(eurclp)
    return float(amount)

def compute_ratio(fmt: str, ratio_map: dict) -> float:
    return float(ratio_map.get(fmt, 1.0))

def round_to_900_up(x: float) -> int:
    """
    Redondea hacia arriba a valores terminados en 900:
      15.200 -> 15.900
      15.900 -> 15.900
      15.901 -> 16.900
    """
    if x is None or np.isnan(x) or x <= 0:
        return 0
    step = 1000
    base = 900
    # ceil((x - base)/step)
    n = int(np.ceil((float(x) - base) / step))
    return int(base + n * step)

def matrix_adjustment(origen: str, ratio: float, rot: str, market: str) -> float:
    """
    Ajustes % según matriz MJ (simple).
    """
    origen = (origen or "").strip()
    rot = (rot or "").strip()
    market = (market or "").strip()

    if origen == "Importado":
        if ratio <= 1 and rot == "Alta" and market == "Price-taker":
            return -0.10
        if ratio <= 1 and rot == "Alta" and market == "Price-maker":
            return -0.05
        if ratio >= 3:
            return 0.12
        if ratio >= 1.8 and rot != "Alta":
            return 0.08
        return 0.00

    if origen == "Local":
        if rot == "Alta" and market == "Price-taker":
            return -0.05
        if rot == "Baja" and market == "Price-maker":
            return 0.08
        return 0.00

    return 0.00

# -----------------------------
# UI
# -----------------------------
st.title("Pricing Importaciones — Music Jungle (v2.0)")
st.caption("Pega la tabla de productos + ingresa cargos (shipping / aduana / IVA / servicios). "
           "Calcula landed SIN IVA y precios sugeridos con redondeo a …900.")

with st.sidebar:
    st.header("1) Parámetros")
    colfx1, colfx2 = st.columns(2)
    usdclp = colfx1.number_input("USD → CLP", min_value=1.0, value=950.0, step=1.0)
    eurclp = colfx2.number_input("EUR → CLP", min_value=1.0, value=1150.0, step=1.0)
    iva_cl = st.number_input("IVA Chile", min_value=0.0, max_value=0.30, value=0.19, step=0.01)

    st.divider()
    st.header("2) Fórmula de pricing")
    formula = st.selectbox(
        "Fórmula base",
        [
            "MJ 1,7 + IVA (default)",
            "Importado 2,2 (IVA incl.)",
            "GM objetivo sobre landed (sin IVA)",
            "Comparar (elegir el mayor)",
        ],
        index=0
    )
    mult_mj = st.number_input("Multiplicador MJ (sobre costo sin IVA)", min_value=0.5, value=1.7, step=0.05)
    mult_import = st.number_input("Multiplicador 2,2 (sobre costo origen)", min_value=0.5, value=2.2, step=0.05)
    gm_target = st.number_input("GM objetivo (si aplica)", min_value=0.10, max_value=0.90, value=0.45, step=0.01)

    st.divider()
    st.header("3) Prorrateo costos compartidos")
    alloc_method = st.selectbox("Método", ["Híbrido 50/50", "Peso", "Valor", "Cantidad"], index=0)

    st.divider()
    st.header("4) Ratios (peso operativo)")
    ratio_map = {}
    for k in FORMATS:
        ratio_map[k] = st.number_input(f"Ratio {k}", min_value=0.01, value=float(DEFAULT_RATIOS[k]), step=0.05)

st.subheader("Pega la tabla de productos")
st.caption("TSV recomendado: Origen, Producto, Formato, Qty, Moneda, Costo unit, Rotación, Mercado. "
           "También sirve con columnas mínimas: Producto, Formato, Qty, Moneda, Costo unit.")

default_example = (
    "Origen\tProducto\tFormato\tQty\tMoneda\tCosto unit\tRotación\tMercado\n"
    "Importado\tAqua – Aquarium\tLP simple\t2\tEUR\t14,95\tMedia\tPrice-taker\n"
    "Importado\tJamiroquai – High Times: Tour Edition\tLP doble\t3\tEUR\t27,50\tAlta\tPrice-taker\n"
    "Importado\tPixies – Surfer Rosa / Come On Pilgrim\tCD\t2\tEUR\t9,50\tAlta\tPrice-taker\n"
)

pasted = st.text_area("Tabla pegada", height=180, value=default_example)
df = parse_pasted_table(pasted)

st.subheader("Cargos / costos adicionales")
colc1, colc2, colc3, colc4 = st.columns(4)
shipping_currency = colc1.selectbox("Moneda shipping internacional", ["EUR", "USD"], index=0)
shipping_amount = colc2.number_input("Shipping internacional (monto)", min_value=0.0, value=0.0, step=1.0)

# Costos locales (CLP)
iva_import = colc3.number_input("IVA importación (CLP)", min_value=0.0, value=0.0, step=1000.0)
derechos = colc4.number_input("Derechos / arancel (CLP)", min_value=0.0, value=0.0, step=1000.0)

colc5, colc6, colc7, colc8 = st.columns(4)
servicios_aduana = colc5.number_input("Servicios aduana/agente (CLP)", min_value=0.0, value=0.0, step=1000.0)
iva_servicios = colc6.number_input("IVA servicios (CLP)", min_value=0.0, value=0.0, step=1000.0)
transporte_local = colc7.number_input("Transporte local (CLP)", min_value=0.0, value=0.0, step=1000.0)
otros = colc8.number_input("Otros (CLP)", min_value=0.0, value=0.0, step=1000.0)

st.divider()

if df.empty:
    st.warning("Pega una tabla válida para comenzar.")
    st.stop()

# -----------------------------
# Cálculos
# -----------------------------
dfc = df.copy()

# Costo origen CLP
dfc["Costo unit CLP (origen)"] = dfc.apply(lambda r: fx_to_clp(r["Moneda"], r["Costo unit"], usdclp, eurclp), axis=1)
dfc["FOB CLP total"] = dfc["Costo unit CLP (origen)"] * dfc["Qty"]

# Peso operativo
dfc["Ratio"] = dfc["Formato"].apply(lambda x: compute_ratio(x, ratio_map))
dfc["Peso unidades"] = dfc["Ratio"] * dfc["Qty"]

total_qty = int(dfc["Qty"].sum())
total_fob = float(dfc["FOB CLP total"].sum())
total_peso_u = float(dfc["Peso unidades"].sum())

# Shipping internacional en CLP
shipping_clp = float(shipping_amount) * (eurclp if shipping_currency == "EUR" else usdclp)

# Separación IVA (landed sin IVA)
shared_non_iva = float(shipping_clp + derechos + servicios_aduana + transporte_local + otros)
shared_iva = float(iva_import + iva_servicios)

# Shares prorrateo
if alloc_method == "Peso":
    denom = total_peso_u if total_peso_u else 1.0
    dfc["Share"] = dfc["Peso unidades"] / denom
elif alloc_method == "Valor":
    denom = total_fob if total_fob else 1.0
    dfc["Share"] = dfc["FOB CLP total"] / denom
elif alloc_method == "Cantidad":
    denom = total_qty if total_qty else 1.0
    dfc["Share"] = dfc["Qty"] / denom
else:  # Híbrido 50/50
    denom_w = total_peso_u if total_peso_u else 1.0
    denom_v = total_fob if total_fob else 1.0
    share_w = dfc["Peso unidades"] / denom_w
    share_v = dfc["FOB CLP total"] / denom_v
    dfc["Share"] = 0.5 * share_w + 0.5 * share_v

# Asignaciones
dfc["Asignado compartidos sin IVA"] = dfc["Share"] * shared_non_iva
dfc["Asignado IVA import/serv"] = dfc["Share"] * shared_iva

# Landed SIN IVA (requisito #1)
dfc["Landed CLP total (sin IVA)"] = dfc["FOB CLP total"] + dfc["Asignado compartidos sin IVA"]
dfc["Landed CLP unit (sin IVA)"] = np.where(dfc["Qty"] > 0, dfc["Landed CLP total (sin IVA)"] / dfc["Qty"], 0.0)

# Landed CON IVA (para análisis de margen caja)
dfc["Landed CLP total (con IVA)"] = dfc["Landed CLP total (sin IVA)"] + dfc["Asignado IVA import/serv"]
dfc["Landed CLP unit (con IVA)"] = np.where(dfc["Qty"] > 0, dfc["Landed CLP total (con IVA)"] / dfc["Qty"], 0.0)

# -------- Pricing escenarios --------
# Ajuste matriz (%)
dfc["Ajuste matriz %"] = dfc.apply(
    lambda r: matrix_adjustment(r.get("Origen", ""), float(r["Ratio"]), r.get("Rotación", ""), r.get("Mercado", "")),
    axis=1
)

# Escenario A: MJ 1,7 + IVA sobre costo SIN IVA (default #2)
dfc["Precio MJ (1,7+IVA)"] = dfc["Landed CLP unit (sin IVA)"] * float(mult_mj) * (1.0 + float(iva_cl))
dfc["Precio MJ (1,7+IVA) ajustado"] = dfc["Precio MJ (1,7+IVA)"] * (1.0 + dfc["Ajuste matriz %"])

# Escenario B: 2,2 sobre costo ORIGEN (como antes)
dfc["Precio 2,2 (origen)"] = dfc["Costo unit CLP (origen)"] * float(mult_import)
dfc["Precio 2,2 (origen) ajustado"] = dfc["Precio 2,2 (origen)"] * (1.0 + dfc["Ajuste matriz %"])

# Escenario C: GM objetivo sobre landed SIN IVA
dfc["Precio p/GM objetivo (sin IVA)"] = np.where(
    dfc["Landed CLP unit (sin IVA)"] > 0,
    dfc["Landed CLP unit (sin IVA)"] / (1.0 - float(gm_target)),
    0.0
)

# Elegir recomendado según fórmula seleccionada
if formula == "MJ 1,7 + IVA (default)":
    dfc["Precio recomendado"] = dfc["Precio MJ (1,7+IVA) ajustado"]
elif formula == "Importado 2,2 (IVA incl.)":
    dfc["Precio recomendado"] = dfc["Precio 2,2 (origen) ajustado"]
elif formula == "GM objetivo sobre landed (sin IVA)":
    dfc["Precio recomendado"] = dfc["Precio p/GM objetivo (sin IVA)"]
else:
    dfc["Precio recomendado"] = np.maximum(
        dfc["Precio MJ (1,7+IVA) ajustado"],
        np.maximum(dfc["Precio 2,2 (origen) ajustado"], dfc["Precio p/GM objetivo (sin IVA)"])
    )

# Redondeo a ...900 + sin decimales (#3 y #4)
dfc["Precio recomendado (…900)"] = dfc["Precio recomendado"].apply(round_to_900_up).astype(int)

# Versiones redondeadas de cada escenario (para resumen #5)
dfc["Precio MJ (…900)"] = dfc["Precio MJ (1,7+IVA) ajustado"].apply(round_to_900_up).astype(int)
dfc["Precio 2,2 (…900)"] = dfc["Precio 2,2 (origen) ajustado"].apply(round_to_900_up).astype(int)
dfc["Precio GM (…900)"] = dfc["Precio p/GM objetivo (sin IVA)"].apply(round_to_900_up).astype(int)

# GM real sobre el precio redondeado (sin IVA y con IVA)
dfc["GM real % (sin IVA)"] = np.where(
    dfc["Precio recomendado (…900)"] > 0,
    (dfc["Precio recomendado (…900)"] - dfc["Landed CLP unit (sin IVA)"]) / dfc["Precio recomendado (…900)"],
    0.0
)
dfc["GM real % (con IVA)"] = np.where(
    dfc["Precio recomendado (…900)"] > 0,
    (dfc["Precio recomendado (…900)"] - dfc["Landed CLP unit (con IVA)"]) / dfc["Precio recomendado (…900)"],
    0.0
)

# -----------------------------
# Salida principal
# -----------------------------
st.subheader("Resumen de importación")

landed_total_sin_iva = float(dfc["Landed CLP total (sin IVA)"].sum())
landed_total_con_iva = float(dfc["Landed CLP total (con IVA)"].sum())
total_shared = float(shared_non_iva + shared_iva)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("FOB total (CLP)", f"{int(round(total_fob)):,}".replace(",", "."))
c2.metric("Costos compartidos (CLP)", f"{int(round(total_shared)):,}".replace(",", "."))
c3.metric("Landed total SIN IVA (CLP)", f"{int(round(landed_total_sin_iva)):,}".replace(",", "."))
c4.metric("Landed total CON IVA (CLP)", f"{int(round(landed_total_con_iva)):,}".replace(",", "."))
avg_gm = dfc["GM real % (sin IVA)"].replace([np.inf, -np.inf], np.nan).dropna()
c5.metric("GM promedio (sin IVA)", f"{(avg_gm.mean()*100 if len(avg_gm) else 0):.1f}%")

st.subheader("Detalle por producto (landed + precio sugerido)")

# Mostrar sin decimales en CLP (#3)
view = dfc.copy()
for col in [
    "Costo unit CLP (origen)",
    "FOB CLP total",
    "Landed CLP unit (sin IVA)",
    "Landed CLP unit (con IVA)",
]:
    view[col] = view[col].round(0).astype(int)

display_cols = [
    "Producto", "Formato", "Qty", "Moneda", "Costo unit",
    "Costo unit CLP (origen)", "Landed CLP unit (sin IVA)", "Landed CLP unit (con IVA)",
    "Ajuste matriz %", "Precio MJ (…900)", "Precio 2,2 (…900)", "Precio GM (…900)",
    "Precio recomendado (…900)",
    "GM real % (sin IVA)", "GM real % (con IVA)",
    "Rotación", "Mercado"
]

st.dataframe(
    view[display_cols],
    use_container_width=True,
    hide_index=True
)

# -----------------------------
# Resumen final (requisito #5)
# -----------------------------
st.divider()
st.subheader("Resumen de la carga por escenario (venta total + margen)")

def totals_for(price_col_900: str):
    sales_total = int(np.sum(dfc[price_col_900].astype(int) * dfc["Qty"].astype(int)))
    gm_sin = (sales_total - landed_total_sin_iva) / sales_total if sales_total > 0 else 0.0
    gm_con = (sales_total - landed_total_con_iva) / sales_total if sales_total > 0 else 0.0
    return sales_total, gm_sin, gm_con

rows = []
scenarios = [
    ("MJ 1,7 + IVA (…900)", "Precio MJ (…900)"),
    ("2,2 sobre origen (…900)", "Precio 2,2 (…900)"),
    ("GM objetivo (…900)", "Precio GM (…900)"),
    ("Recomendado actual (…900)", "Precio recomendado (…900)"),
]

for label, colp in scenarios:
    sales_total, gm_sin, gm_con = totals_for(colp)
    rows.append({
        "Escenario": label,
        "Venta total (CLP)": sales_total,
        "Margen % (sin IVA)": round(gm_sin * 100, 1),
        "Margen % (con IVA)": round(gm_con * 100, 1),
    })

summary_df = pd.DataFrame(rows)
st.dataframe(summary_df, use_container_width=True, hide_index=True)

# Mostrar también costo total de la carga (sin/con IVA)
c6, c7 = st.columns(2)
c6.metric("Costo total carga (Landed SIN IVA)", f"{int(round(landed_total_sin_iva)):,}".replace(",", "."))
c7.metric("Costo total carga (Landed CON IVA)", f"{int(round(landed_total_con_iva)):,}".replace(",", "."))

# -----------------------------
# Exportar
# -----------------------------
st.subheader("Exportar")

export_cols = [
    "Producto", "Formato", "Qty", "Moneda", "Costo unit",
    "Costo unit CLP (origen)", "Landed CLP unit (sin IVA)", "Landed CLP unit (con IVA)",
    "Precio MJ (…900)", "Precio 2,2 (…900)", "Precio GM (…900)",
    "Precio recomendado (…900)",
    "GM real % (sin IVA)", "GM real % (con IVA)",
    "Rotación", "Mercado"
]

export_df = dfc.copy()
# redondear/clpear a enteros donde corresponde
export_df["Costo unit CLP (origen)"] = export_df["Costo unit CLP (origen)"].round(0).astype(int)
export_df["Landed CLP unit (sin IVA)"] = export_df["Landed CLP unit (sin IVA)"].round(0).astype(int)
export_df["Landed CLP unit (con IVA)"] = export_df["Landed CLP unit (con IVA)"].round(0).astype(int)
export_df["GM real % (sin IVA)"] = export_df["GM real % (sin IVA)"].round(4)
export_df["GM real % (con IVA)"] = export_df["GM real % (con IVA)"].round(4)

out_csv = export_df[export_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    "Descargar CSV",
    data=out_csv,
    file_name="pricing_mj_importacion_v2.csv",
    mime="text/csv"
)

st.caption("Notas: Landed SIN IVA excluye IVA importación y el IVA de servicios (recuperables como crédito). "
           "El resumen muestra márgenes vs landed SIN IVA y vs landed CON IVA (flujo de caja).")
