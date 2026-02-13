# app.py — Pricing Importaciones (MJ) v2.2
# v2.2 agrega "check de consistencia" FOB:
# - Calcula FOB total en moneda base (EUR si todos son EUR, USD si todos son USD; si mixto, lo indica)
# - Permite ingresar "Subtotal esperado (factura)" en EUR/USD
# - Muestra diferencia absoluta y % y alerta si supera umbral
# Mantiene v2.1:
# - Landed SIN IVA
# - Compartidos (shipping + servicio aduana) prorrateo por PESO
# - Derechos + IVA importación prorrateo por VALOR (FOB)
#   - Derechos sí al landed / IVA importación no al landed
# - Pricing: MJ / Oferta / Protegido con redondeo a ...900
# - Total ítem con IVA + fila TOTAL

import io
import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Pricing Importaciones — Music Jungle (v2.2)", layout="wide")

FORMATS = ["CD", "Cassette", "LP simple", "LP doble", "Box"]
DEFAULT_RATIOS = {"CD": 0.2, "Cassette": 0.25, "LP simple": 1.0, "LP doble": 1.8, "Box": 3.2}

def normalize_money(x):
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
    text = (text or "").strip()
    if not text:
        return pd.DataFrame()
    try:
        df = pd.read_csv(io.StringIO(text), sep="\t", dtype=str)
        if df.shape[1] <= 1:
            raise ValueError("Not TSV")
    except Exception:
        df = pd.read_csv(io.StringIO(text), sep=",", dtype=str)

    df.columns = [c.strip() for c in df.columns]

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

    df = df.rename(columns=colmap)

    for needed in ["Producto", "Formato", "Qty", "Moneda", "Costo unit"]:
        if needed not in df.columns:
            df[needed] = ""

    if "Origen" not in df.columns:
        df["Origen"] = "Importado"

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
    if x is None or np.isnan(x) or x <= 0:
        return 0
    step = 1000
    base = 900
    n = int(np.ceil((float(x) - base) / step))
    return int(base + n * step)

def fmt_clp(n: float | int) -> str:
    return f"{int(round(float(n))):,}".replace(",", ".")

# ---------------- UI ----------------
st.title("Pricing Importaciones — Music Jungle (v2.2)")
st.caption("Landed sin IVA + pricing (MJ/Oferta/Protegido) + chequeo de consistencia del subtotal de factura.")

with st.sidebar:
    st.header("Parámetros")
    cfx1, cfx2 = st.columns(2)
    usdclp = cfx1.number_input("USD → CLP", min_value=1.0, value=950.0, step=1.0)
    eurclp = cfx2.number_input("EUR → CLP", min_value=1.0, value=1150.0, step=1.0)
    iva_chile = st.number_input("IVA Chile", min_value=0.0, max_value=0.30, value=0.19, step=0.01)

    st.divider()
    st.header("Ratios (peso operativo)")
    ratio_map = {}
    for k in FORMATS:
        ratio_map[k] = st.number_input(f"Ratio {k}", min_value=0.01, value=float(DEFAULT_RATIOS[k]), step=0.05)

st.subheader("Pega la tabla de productos")
st.caption("TSV recomendado: Origen, Producto, Formato, Qty, Moneda, Costo unit")
default_example = (
    "Origen\tProducto\tFormato\tQty\tMoneda\tCosto unit\n"
    "Importado\tAqua – Aquarium\tLP simple\t2\tEUR\t14,95\n"
    "Importado\tJamiroquai – High Times: Tour Edition\tLP doble\t3\tEUR\t27,50\n"
    "Importado\tPixies – Surfer Rosa / Come On Pilgrim\tCD\t2\tEUR\t9,50\n"
)
pasted = st.text_area("Tabla pegada", height=180, value=default_example)
df = parse_pasted_table(pasted)

st.subheader("Cargos / costos adicionales")
colc1, colc2, colc3, colc4, colc5 = st.columns(5)
shipping_currency = colc1.selectbox("Moneda shipping", ["EUR", "USD"], index=0)
shipping_amount = colc2.number_input("Shipping internacional (monto)", min_value=0.0, value=0.0, step=1.0)
servicio_aduana = colc3.number_input("Servicio aduana/agente (CLP)", min_value=0.0, value=0.0, step=1000.0)
derechos_aduana = colc4.number_input("Derechos / arancel (CLP)", min_value=0.0, value=0.0, step=1000.0)
iva_importacion = colc5.number_input("IVA importación (CLP) (referencial)", min_value=0.0, value=0.0, step=1000.0)

st.divider()

if df.empty:
    st.warning("Pega una tabla válida para comenzar.")
    st.stop()

# ---------------- Cálculo base ----------------
dfc = df.copy()
dfc["Costo unit CLP"] = dfc.apply(lambda r: fx_to_clp(r["Moneda"], r["Costo unit"], usdclp, eurclp), axis=1)
dfc["FOB CLP total"] = dfc["Costo unit CLP"] * dfc["Qty"]

dfc["Ratio"] = dfc["Formato"].apply(lambda x: compute_ratio(x, ratio_map))
dfc["Peso unidades"] = dfc["Ratio"] * dfc["Qty"]

total_fob_clp = float(dfc["FOB CLP total"].sum())
total_peso = float(dfc["Peso unidades"].sum())

shipping_clp = float(shipping_amount) * (eurclp if shipping_currency == "EUR" else usdclp)
shared_costs = float(shipping_clp + servicio_aduana)

den_w = total_peso if total_peso else 1.0
den_v = total_fob_clp if total_fob_clp else 1.0

dfc["Share_peso"] = dfc["Peso unidades"] / den_w
dfc["Share_valor"] = dfc["FOB CLP total"] / den_v

dfc["Asignado compartidos (shipping+aduana)"] = dfc["Share_peso"] * shared_costs
dfc["Asignado derechos"] = dfc["Share_valor"] * float(derechos_aduana)

dfc["Landed CLP total (sin IVA)"] = dfc["FOB CLP total"] + dfc["Asignado compartidos (shipping+aduana)"] + dfc["Asignado derechos"]
dfc["Landed CLP unit (sin IVA)"] = np.where(dfc["Qty"] > 0, dfc["Landed CLP total (sin IVA)"] / dfc["Qty"], 0.0)

# ---------------- Chequeo consistencia factura ----------------
st.subheader("Chequeo de consistencia (subtotal de factura)")

currs = sorted([c for c in dfc["Moneda"].dropna().unique().tolist() if str(c).strip() != ""])
single_currency = currs[0] if len(currs) == 1 else None

# FOB en moneda origen:
# - Si la tabla es 100% EUR: suma (costo unit * qty) en EUR
# - Si mixta: mostrar por moneda
dfc["FOB moneda origen"] = dfc["Costo unit"] * dfc["Qty"]
fob_by_currency = dfc.groupby("Moneda", dropna=False)["FOB moneda origen"].sum().sort_index()

colchk1, colchk2, colchk3, colchk4 = st.columns(4)
if single_currency:
    fob_origin = float(fob_by_currency.loc[single_currency])
    colchk1.metric("FOB (moneda origen)", f"{fob_origin:,.2f} {single_currency}".replace(",", "."))
else:
    colchk1.metric("FOB (moneda origen)", "Mixto (ver detalle)")

expected_currency = colchk2.selectbox("Moneda subtotal esperado", ["EUR", "USD"], index=0)
expected_subtotal = colchk3.number_input("Subtotal esperado (factura, sin shipping)", min_value=0.0, value=0.0, step=1.0)
tolerance_pct = colchk4.number_input("Umbral alerta (%)", min_value=0.0, value=1.0, step=0.5)

with st.expander("Detalle FOB por moneda (si aplica)"):
    tmp = fob_by_currency.reset_index()
    tmp.columns = ["Moneda", "FOB (moneda)"]
    st.dataframe(tmp, use_container_width=True, hide_index=True)

if expected_subtotal > 0:
    if single_currency and expected_currency == single_currency:
        diff = fob_origin - float(expected_subtotal)
        diff_pct = (diff / expected_subtotal) * 100.0 if expected_subtotal else 0.0

        # Mensaje
        if abs(diff_pct) > float(tolerance_pct):
            st.error(
                f"⚠️ La tabla NO cuadra con el subtotal esperado. "
                f"Diferencia: {diff:,.2f} {single_currency} ({diff_pct:+.2f}%).".replace(",", ".")
            )
            st.caption("Esto suele pasar cuando falta una página de la factura o se pegó una tabla incompleta/duplicada.")
        else:
            st.success(
                f"✅ La tabla cuadra con el subtotal esperado. "
                f"Diferencia: {diff:,.2f} {single_currency} ({diff_pct:+.2f}%).".replace(",", ".")
            )
    else:
        st.warning(
            "No puedo comparar automáticamente porque la tabla tiene moneda mixta o la moneda esperada no coincide. "
            "Si quieres, normaliza la tabla a una sola moneda o pega el subtotal por moneda."
        )

# ---------------- Pricing ----------------
iva_factor = 1.0 + float(iva_chile)

dfc["Precio MJ"] = dfc["Landed CLP unit (sin IVA)"] * 1.7 * iva_factor
dfc["Precio oferta"] = dfc["Landed CLP unit (sin IVA)"] * 1.5 * iva_factor
dfc["Precio protegido"] = dfc["Landed CLP unit (sin IVA)"] * 2.2 * iva_factor

dfc["Precio MJ (…900)"] = dfc["Precio MJ"].apply(round_to_900_up).astype(int)
dfc["Precio oferta (…900)"] = dfc["Precio oferta"].apply(round_to_900_up).astype(int)
dfc["Precio protegido (…900)"] = dfc["Precio protegido"].apply(round_to_900_up).astype(int)

dfc["Total ítem MJ (CLP, IVA incl.)"] = (dfc["Precio MJ (…900)"] * dfc["Qty"]).astype(int)
dfc["Total ítem Oferta (CLP, IVA incl.)"] = (dfc["Precio oferta (…900)"] * dfc["Qty"]).astype(int)
dfc["Total ítem Protegido (CLP, IVA incl.)"] = (dfc["Precio protegido (…900)"] * dfc["Qty"]).astype(int)

# ---------------- Resumen importación (simplificado) ----------------
st.subheader("Resumen de importación")

landed_total = float(dfc["Landed CLP total (sin IVA)"].sum())
fob_total = total_fob_clp
shared_total = shared_costs
derechos_total = float(derechos_aduana)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("FOB total (CLP)", fmt_clp(fob_total))
m2.metric("Compartidos (Shipping + Aduana)", fmt_clp(shared_total))
m3.metric("Derechos (prorrateo por valor)", fmt_clp(derechos_total))
m4.metric("IVA importación (referencial)", fmt_clp(iva_importacion))
m5.metric("Landed total sin IVA (CLP)", fmt_clp(landed_total))

st.caption("Reglas: Shipping + Servicio Aduana se prorratean por PESO. Derechos e IVA importación por VALOR (FOB). "
           "El IVA importación no se incluye en landed (crédito fiscal).")

# ---------------- Tabla + fila TOTAL ----------------
st.subheader("Detalle por producto (landed + pricing)")

view = dfc.copy()
view["Costo unit CLP"] = view["Costo unit CLP"].round(0).astype(int)
view["FOB CLP total"] = view["FOB CLP total"].round(0).astype(int)
view["Landed CLP unit (sin IVA)"] = view["Landed CLP unit (sin IVA)"].round(0).astype(int)
view["Landed CLP total (sin IVA)"] = view["Landed CLP total (sin IVA)"].round(0).astype(int)

display_cols = [
    "Producto", "Formato", "Qty", "Moneda", "Costo unit",
    "Costo unit CLP", "Landed CLP unit (sin IVA)", "Landed CLP total (sin IVA)",
    "Precio MJ (…900)", "Total ítem MJ (CLP, IVA incl.)",
    "Precio oferta (…900)", "Total ítem Oferta (CLP, IVA incl.)",
    "Precio protegido (…900)", "Total ítem Protegido (CLP, IVA incl.)",
]
table_df = view[display_cols].copy()

total_row = {c: "" for c in table_df.columns}
total_row["Producto"] = "TOTAL"
total_row["Qty"] = int(dfc["Qty"].sum())
total_row["Landed CLP total (sin IVA)"] = int(round(float(dfc["Landed CLP total (sin IVA)"].sum())))
total_row["Total ítem MJ (CLP, IVA incl.)"] = int(round(float(dfc["Total ítem MJ (CLP, IVA incl.)"].sum())))
total_row["Total ítem Oferta (CLP, IVA incl.)"] = int(round(float(dfc["Total ítem Oferta (CLP, IVA incl.)"].sum())))
total_row["Total ítem Protegido (CLP, IVA incl.)"] = int(round(float(dfc["Total ítem Protegido (CLP, IVA incl.)"].sum())))

table_df = pd.concat([table_df, pd.DataFrame([total_row])], ignore_index=True)
st.dataframe(table_df, use_container_width=True, hide_index=True)

# ---------------- Export ----------------
st.subheader("Exportar")
out_csv = table_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Descargar CSV",
    data=out_csv,
    file_name="pricing_mj_importacion_v2_2.csv",
    mime="text/csv"
)
st.caption("Chequeo de consistencia: ingresa el subtotal esperado (sin shipping) para detectar tablas incompletas o duplicadas.")
