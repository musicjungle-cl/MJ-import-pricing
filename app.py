# app.py
# Streamlit app — Pricing Inteligente MJ (Importaciones)
# Permite: pegar tabla de productos + ingresar cargos (shipping + aduana/IVA/servicios) + elegir fórmula
# Devuelve: landed cost + precios sugeridos por SKU + brechas vs precio base (2.2)

import io
import re
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Pricing Importaciones — Music Jungle", layout="wide")

# -----------------------------
# Helpers
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

def normalize_money(x):
    """Accepts '14,95' or '14.95' or '1.234,56' etc."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "":
        return np.nan
    # remove currency symbols and spaces
    s = re.sub(r"[^\d,.\-]", "", s)
    # if both separators exist, assume thousands + decimal
    if "," in s and "." in s:
        # decide decimal separator as last occurrence
        if s.rfind(",") > s.rfind("."):
            # '.' thousands, ',' decimal
            s = s.replace(".", "").replace(",", ".")
        else:
            # ',' thousands, '.' decimal
            s = s.replace(",", "")
    else:
        # only comma: treat as decimal
        if "," in s:
            s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except:
        return np.nan

def parse_pasted_table(text: str) -> pd.DataFrame:
    """
    Expects TSV (tab-separated) like:
    Origen\tProducto\tFormato\tQty\tMoneda\tCosto unit\tRotación\tMercado
    or minimal:
    Producto\tFormato\tQty\tMoneda\tCosto unit
    """
    text = (text or "").strip()
    if not text:
        return pd.DataFrame()

    # Try TSV first
    try:
        df = pd.read_csv(io.StringIO(text), sep="\t", dtype=str)
        if df.shape[1] <= 1:
            raise ValueError("Not TSV")
    except Exception:
        # fallback: CSV with commas
        df = pd.read_csv(io.StringIO(text), sep=",", dtype=str)

    # Trim columns
    df.columns = [c.strip() for c in df.columns]

    # Map common column names to expected schema
    colmap = {}
    for c in df.columns:
        cl = c.lower()
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

    # Ensure required columns exist
    for needed in ["Producto", "Formato", "Qty", "Moneda", "Costo unit"]:
        if needed not in df.columns:
            df[needed] = ""

    if "Origen" not in df.columns:
        df["Origen"] = "Importado"
    if "Rotación" not in df.columns:
        df["Rotación"] = ""
    if "Mercado" not in df.columns:
        df["Mercado"] = ""

    # Normalize
    df["Origen"] = df["Origen"].fillna("Importado").replace("", "Importado")
    df["Producto"] = df["Producto"].fillna("").astype(str).str.strip()
    df["Formato"] = df["Formato"].fillna("").astype(str).str.strip()
    df["Moneda"] = df["Moneda"].fillna("").astype(str).str.strip().str.upper()
    df["Qty"] = df["Qty"].apply(normalize_money).fillna(0).astype(int)
    df["Costo unit"] = df["Costo unit"].apply(normalize_money)

    # Standardize formats: accept LP / 2-LP / 2LP / CD / 2-CD etc.
    def std_fmt(x):
        s = str(x).strip().upper()
        if s in ["LP", "1LP", "VINYL", "VINILO"]:
            return "LP simple"
        if s in ["2-LP", "2LP", "2 LP", "2XLP", "DOUBLE LP"]:
            return "LP doble"
        if s in ["CD", "1CD"]:
            return "CD"
        if s in ["2-CD", "2CD", "2 CD", "2XCD", "DOUBLE CD"]:
            return "CD"  # operativamente lo tratamos como CD liviano
        if s in ["CASSETTE", "TAPE"]:
            return "Cassette"
        if s in ["BOX", "BOXSET"]:
            return "Box"
        # already in target?
        if x in FORMATS:
            return x
        return x if x else "LP simple"

    df["Formato"] = df["Formato"].apply(std_fmt)

    # Clean rotations/markets
    df["Rotación"] = df["Rotación"].astype(str).str.strip()
    df["Mercado"] = df["Mercado"].astype(str).str.strip()

    return df

def compute_ratio(fmt: str, ratio_map: dict) -> float:
    return float(ratio_map.get(fmt, 1.0))

def round_up(price_clp: float, step: int) -> float:
    if price_clp is None or np.isnan(price_clp) or price_clp <= 0:
        return 0.0
    return float(int(np.ceil(price_clp / step) * step))

def matrix_adjustment(origen: str, ratio: float, rot: str, market: str) -> float:
    """
    Ajustes basados en la matriz que definimos.
    Retorna % (ej: -0.10, +0.08)
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

    # Local (si quisieras usarlo acá también)
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
st.title("Pricing Importaciones — Music Jungle")
st.caption("Pega la tabla de productos + ingresa cargos (shipping / aduana / IVA / servicios) y obtén landed cost y precios sugeridos.")

with st.sidebar:
    st.header("1) Parámetros")
    colfx1, colfx2 = st.columns(2)
    usdclp = colfx1.number_input("USD → CLP", min_value=1.0, value=950.0, step=1.0)
    eurclp = colfx2.number_input("EUR → CLP", min_value=1.0, value=1150.0, step=1.0)

    iva_cl = st.number_input("IVA Chile", min_value=0.0, max_value=0.30, value=0.19, step=0.01)
    rounding_step = st.number_input("Redondeo (CLP)", min_value=1, value=900, step=100)

    st.divider()
    st.header("2) Fórmula de pricing")
    pricing_mode = st.selectbox(
        "Modo de recomendación",
        [
            "Seguir fórmula base (2,2) + Matriz (ajuste)",
            "Proteger margen (GM objetivo sobre landed)",
            "Comparar ambos (y recomendar el mayor)",
        ],
        index=2
    )
    mult_import = st.number_input("Multiplicador importado (IVA incl.)", min_value=0.5, value=2.2, step=0.05)
    gm_target = st.number_input("GM objetivo (si aplica)", min_value=0.10, max_value=0.90, value=0.45, step=0.01)

    st.divider()
    st.header("3) Prorrateo costos compartidos")
    alloc_method = st.selectbox("Método", ["Híbrido 50/50", "Peso", "Valor", "Cantidad"], index=0)

    st.divider()
    st.header("4) Ratios (peso operativo)")
    st.caption("Ajusta solo si quieres afinar tu prorrateo.")
    ratio_map = {}
    for k in FORMATS:
        ratio_map[k] = st.number_input(f"Ratio {k}", min_value=0.01, value=float(DEFAULT_RATIOS[k]), step=0.05)

st.subheader("Pega la tabla de productos")
st.caption("Formato recomendado (TSV): Origen, Producto, Formato, Qty, Moneda, Costo unit, Rotación, Mercado. "
           "También funciona con columnas mínimas: Producto, Formato, Qty, Moneda, Costo unit.")

default_example = (
    "Origen\tProducto\tFormato\tQty\tMoneda\tCosto unit\tRotación\tMercado\n"
    "Importado\tAir – Virgin Suicides Redux\tLP simple\t2\tEUR\t18,50\tMedia\tPrice-taker\n"
    "Importado\tLeftfield – Leftism\tLP doble\t2\tEUR\t27,50\tAlta\tPrice-taker\n"
    "Importado\tBeastie Boys – Ill Communication\tCD\t2\tEUR\t5,45\tAlta\tPrice-taker\n"
)

pasted = st.text_area("Tabla pegada", height=180, value=default_example)

df = parse_pasted_table(pasted)

st.subheader("Cargos / costos adicionales")
colc1, colc2, colc3, colc4 = st.columns(4)

shipping_currency = colc1.selectbox("Moneda shipping internacional", ["EUR", "USD"], index=0)
shipping_amount = colc2.number_input("Shipping internacional (monto)", min_value=0.0, value=0.0, step=1.0)

# Costos locales CLP (aduana, IVA importación, servicios, etc.)
iva_import = colc3.number_input("IVA importación (CLP)", min_value=0.0, value=0.0, step=1000.0)
derechos = colc4.number_input("Derechos / arancel (CLP)", min_value=0.0, value=0.0, step=1000.0)

colc5, colc6, colc7, colc8 = st.columns(4)
servicios_aduana = colc5.number_input("Servicios aduana/agente (CLP)", min_value=0.0, value=0.0, step=1000.0)
iva_servicios = colc6.number_input("IVA servicios (CLP)", min_value=0.0, value=0.0, step=1000.0)
transporte_local = colc7.number_input("Transporte local (CLP)", min_value=0.0, value=0.0, step=1000.0)
otros = colc8.number_input("Otros (CLP)", min_value=0.0, value=0.0, step=1000.0)

# -----------------------------
# Compute
# -----------------------------
if df.empty:
    st.warning("Pega una tabla válida para comenzar.")
    st.stop()

# Convert origin unit cost to CLP
def fx_to_clp(currency: str, amount: float) -> float:
    if pd.isna(amount):
        return np.nan
    c = (currency or "").upper().strip()
    if c == "USD":
        return float(amount) * float(usdclp)
    if c == "EUR":
        return float(amount) * float(eurclp)
    # fallback assume CLP already
    return float(amount)

df_calc = df.copy()

df_calc["Costo unit CLP"] = df_calc.apply(lambda r: fx_to_clp(r["Moneda"], r["Costo unit"]), axis=1)
df_calc["FOB CLP total"] = df_calc["Costo unit CLP"] * df_calc["Qty"]
df_calc["Ratio"] = df_calc["Formato"].apply(lambda x: compute_ratio(x, ratio_map))
df_calc["Peso unidades"] = df_calc["Ratio"] * df_calc["Qty"]

total_qty = df_calc["Qty"].sum()
total_fob = df_calc["FOB CLP total"].sum()
total_peso_u = df_calc["Peso unidades"].sum()

# Shared costs
shipping_clp = (shipping_amount * (eurclp if shipping_currency == "EUR" else usdclp))
shared_clp = float(shipping_clp + iva_import + derechos + servicios_aduana + iva_servicios + transporte_local + otros)

# Allocation shares
if alloc_method == "Peso":
    denom = total_peso_u if total_peso_u else 1
    df_calc["Share"] = df_calc["Peso unidades"] / denom
elif alloc_method == "Valor":
    denom = total_fob if total_fob else 1
    df_calc["Share"] = df_calc["FOB CLP total"] / denom
elif alloc_method == "Cantidad":
    denom = total_qty if total_qty else 1
    df_calc["Share"] = df_calc["Qty"] / denom
else:  # Híbrido 50/50
    denom_w = total_peso_u if total_peso_u else 1
    denom_v = total_fob if total_fob else 1
    share_w = df_calc["Peso unidades"] / denom_w
    share_v = df_calc["FOB CLP total"] / denom_v
    df_calc["Share"] = 0.5 * share_w + 0.5 * share_v

df_calc["Costos compartidos asignados"] = df_calc["Share"] * shared_clp
df_calc["Landed CLP total"] = df_calc["FOB CLP total"] + df_calc["Costos compartidos asignados"]
df_calc["Landed CLP unit"] = np.where(df_calc["Qty"] > 0, df_calc["Landed CLP total"] / df_calc["Qty"], 0)

# Pricing calculations
df_calc["Precio base 2.2"] = df_calc["Costo unit CLP"] * float(mult_import)
df_calc["Ajuste matriz %"] = df_calc.apply(
    lambda r: matrix_adjustment(r["Origen"], r["Ratio"], r["Rotación"], r["Mercado"]),
    axis=1
)
df_calc["Precio 2.2 + matriz"] = df_calc["Precio base 2.2"] * (1 + df_calc["Ajuste matriz %"])

# Price to hit GM target on landed
df_calc["Precio p/GM objetivo"] = np.where(
    df_calc["Landed CLP unit"] > 0,
    df_calc["Landed CLP unit"] / (1 - float(gm_target)),
    0
)

if pricing_mode == "Seguir fórmula base (2,2) + Matriz (ajuste)":
    df_calc["Precio recomendado"] = df_calc["Precio 2.2 + matriz"]
elif pricing_mode == "Proteger margen (GM objetivo sobre landed)":
    df_calc["Precio recomendado"] = df_calc["Precio p/GM objetivo"]
else:
    df_calc["Precio recomendado"] = np.maximum(df_calc["Precio 2.2 + matriz"], df_calc["Precio p/GM objetivo"])

df_calc["Precio recomendado redondeado"] = df_calc["Precio recomendado"].apply(lambda x: round_up(x, int(rounding_step)))

# Diagnostics
df_calc["GM real (sobre redondeado)"] = np.where(
    df_calc["Precio recomendado redondeado"] > 0,
    (df_calc["Precio recomendado redondeado"] - df_calc["Landed CLP unit"]) / df_calc["Precio recomendado redondeado"],
    0
)
df_calc["Brecha vs 2.2 (CLP)"] = df_calc["Precio recomendado redondeado"] - df_calc["Precio base 2.2"].apply(lambda x: round_up(x, int(rounding_step)))

# Ordering / display
display_cols = [
    "Producto", "Formato", "Qty", "Moneda", "Costo unit",
    "Costo unit CLP", "Landed CLP unit",
    "Precio base 2.2", "Ajuste matriz %", "Precio 2.2 + matriz",
    "Precio p/GM objetivo", "Precio recomendado redondeado",
    "GM real (sobre redondeado)", "Brecha vs 2.2 (CLP)",
    "Rotación", "Mercado"
]
for c in display_cols:
    if c not in df_calc.columns:
        df_calc[c] = ""

# -----------------------------
# Output
# -----------------------------
st.divider()
st.subheader("Resumen de importación")

c1, c2, c3, c4 = st.columns(4)
c1.metric("FOB total (CLP)", f"{total_fob:,.0f}".replace(",", "."))
c2.metric("Costos compartidos (CLP)", f"{shared_clp:,.0f}".replace(",", "."))
c3.metric("Landed total (CLP)", f"{df_calc['Landed CLP total'].sum():,.0f}".replace(",", "."))
avg_gm = df_calc["GM real (sobre redondeado)"].replace([np.inf, -np.inf], np.nan).dropna()
c4.metric("GM promedio (recomendado)", f"{(avg_gm.mean()*100 if len(avg_gm) else 0):.1f}%")

st.subheader("Detalle por producto (landed + precio sugerido)")
st.dataframe(
    df_calc[display_cols].copy(),
    use_container_width=True,
    hide_index=True
)

# Downloads
st.subheader("Exportar")
out_csv = df_calc[display_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    "Descargar CSV",
    data=out_csv,
    file_name="pricing_mj_importacion.csv",
    mime="text/csv"
)

st.caption("Tip: si quieres usar esto para WooCommerce, agrega una columna Barcode/SKU en la tabla pegada y se exportará igual.")

