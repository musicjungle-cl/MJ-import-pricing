import io, re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MJ Import Pricing (MVP+)", layout="wide")
st.title("Import Pricing — Music Jungle (MVP+)")
st.caption("MVP: pegar productos + costos Chile → landed sin IVA + 3 precios (…900) + wrap-up. "
           "Incluye: (1) detector de factura incompleta (subtotal) y (2) edición manual de precios con rentabilidad en vivo.")

# -------------------------
# Defaults (fijos para simplificar)
# -------------------------
IVA_CL = 0.19
RATIOS = {"CD": 0.2, "LP": 1.0, "2LP": 1.8, "BOX": 3.2}

# -------------------------
# Helpers
# -------------------------
def money(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip()
    if not s:
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

def round_900_up(x):
    if x is None or np.isnan(x) or x <= 0:
        return 0
    step, base = 1000, 900
    n = int(np.ceil((float(x) - base) / step))
    return int(base + n * step)

def parse_table(txt):
    txt = (txt or "").strip()
    if not txt:
        return pd.DataFrame()
    try:
        df = pd.read_csv(io.StringIO(txt), sep="\t", dtype=str)
        if df.shape[1] <= 1:
            raise ValueError()
    except:
        df = pd.read_csv(io.StringIO(txt), sep=",", dtype=str)

    df.columns = [c.strip() for c in df.columns]

    ren = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ["producto","product","artist-title","title"]:
            ren[c] = "Producto"
        elif cl in ["formato","format","configuration"]:
            ren[c] = "Formato"
        elif cl in ["qty","cantidad","quantity"]:
            ren[c] = "Qty"
        elif cl in ["moneda","currency"]:
            ren[c] = "Moneda"
        elif cl in ["costo unit","costo unitario","net price","unit cost"]:
            ren[c] = "Costo unit"
        elif cl in ["barcode","ean","gtin","upc","sku"]:
            ren[c] = "SKU"
    df = df.rename(columns=ren)

    for c in ["Producto","Formato","Qty","Moneda","Costo unit"]:
        if c not in df.columns:
            df[c] = ""
    if "SKU" not in df.columns:
        df["SKU"] = ""

    df["Producto"] = df["Producto"].fillna("").astype(str).str.strip()
    df["SKU"] = df["SKU"].fillna("").astype(str).str.strip()
    df["Moneda"] = df["Moneda"].fillna("").astype(str).str.strip().str.upper()
    df["Qty"] = df["Qty"].apply(money).fillna(0).astype(int)
    df["Costo unit"] = df["Costo unit"].apply(money)

    def fmt(x):
        s = str(x).strip().upper()
        if s in ["CD","1CD","2-CD","2CD","3-CD","3CD","DVD","3-DVD","BLURAY","BD"]:
            return "CD"
        if s in ["LP","1LP","VINYL","VINILO"]:
            return "LP"
        if s in ["2LP","2-LP","2 LP","3LP","3-LP","3 LP","DOUBLE LP","TRIPLE LP"]:
            return "2LP"
        if s in ["BOX","BOXSET"]:
            return "BOX"
        return "LP"
    df["Formato"] = df["Formato"].apply(fmt)

    # Filtrar filas vacías
    df = df[(df["Producto"] != "") & (df["Qty"] > 0) & (~df["Costo unit"].isna())]
    return df.reset_index(drop=True)

def fx(currency, amount, eurclp, usdclp):
    if pd.isna(amount):
        return np.nan
    c = (currency or "").upper()
    if c == "EUR":
        return amount * eurclp
    if c == "USD":
        return amount * usdclp
    return amount

def fmt_clp(n):
    return f"{int(round(float(n))):,}".replace(",", ".")

# -------------------------
# UI
# -------------------------
left, right = st.columns([1.35, 0.65])

with left:
    st.subheader("1) Productos (pegar tabla)")
    st.caption("Columnas mínimas: Producto | Formato (CD/LP/2LP/BOX) | Qty | Moneda (EUR/USD) | Costo unit. "
               "Opcional: SKU/Barcode.")

    example = (
        "Producto\tFormato\tQty\tMoneda\tCosto unit\n"
        "Air – Virgin Suicides Redux\tLP\t2\tEUR\t18,50\n"
        "Leftfield – Leftism\t2LP\t2\tEUR\t27,50\n"
        "Beastie Boys – Ill Communication\tCD\t2\tEUR\t5,45\n"
    )
    txt = st.text_area("Tabla", height=180, value=example)

    st.subheader("Shipping origen → aeropuerto (misma moneda de la factura)")
    ship_origin = st.number_input("Shipping origen (EUR/USD)", min_value=0.0, value=0.0, step=1.0)

with right:
    st.subheader("Tipo de cambio")
    eurclp = st.number_input("EUR → CLP", min_value=1.0, value=1150.0, step=1.0)
    usdclp = st.number_input("USD → CLP", min_value=1.0, value=950.0, step=1.0)

    st.subheader("2) Costos Chile (FedEx) — CLP")
    iva_import = st.number_input("IVA importación (CLP)", min_value=0.0, value=0.0, step=1000.0)
    derechos = st.number_input("Derechos aduana (CLP)", min_value=0.0, value=0.0, step=1000.0)
    proceso = st.number_input("Proceso de entrada (CLP)", min_value=0.0, value=0.0, step=1000.0)
    iva_agente = st.number_input("IVA agente aduana (CLP)", min_value=0.0, value=0.0, step=1000.0)

run = st.button("Calcular", type="primary")

if not run:
    st.stop()

df = parse_table(txt)
if df.empty:
    st.error("No pude leer productos válidos. Revisa que existan Qty y Costo unit.")
    st.stop()

# -------------------------
# Detector de factura incompleta (subtotal)
# -------------------------
st.subheader("🧾 Check de factura incompleta (subtotal)")

# Subtotal calculado desde tabla por moneda: sum(qty * costo_unit) en moneda origen
df["Subtotal moneda origen"] = df["Qty"] * df["Costo unit"]
by_cur = df.groupby("Moneda")["Subtotal moneda origen"].sum().reset_index()

colA, colB, colC, colD = st.columns([1.2, 1.0, 1.0, 0.8])
expected_currency = colA.selectbox("Moneda del subtotal esperado", ["EUR", "USD"], index=0)
expected_subtotal = colB.number_input("Subtotal esperado (sin shipping, factura)", min_value=0.0, value=0.0, step=1.0)
tolerance_pct = colC.number_input("Tolerancia alerta (%)", min_value=0.0, value=1.0, step=0.5)
show_breakdown = colD.toggle("Ver detalle", value=False)

if show_breakdown:
    st.dataframe(by_cur, use_container_width=True, hide_index=True)

currs = sorted(df["Moneda"].unique().tolist())
single_currency = (len(currs) == 1)

if expected_subtotal > 0:
    if single_currency and currs[0] == expected_currency:
        computed = float(by_cur.loc[by_cur["Moneda"] == expected_currency, "Subtotal moneda origen"].iloc[0])
        diff = computed - float(expected_subtotal)
        diff_pct = (diff / expected_subtotal) * 100 if expected_subtotal else 0.0
        if abs(diff_pct) > tolerance_pct:
            st.error(f"⚠️ No cuadra: calculado {computed:,.2f} {expected_currency} vs esperado {expected_subtotal:,.2f} "
                     f"({diff_pct:+.2f}%).".replace(",", "."))
            st.caption("Suele indicar: falta una página/filas, pegaste tabla incompleta, o duplicaste líneas.")
        else:
            st.success(f"✅ Cuadra: calculado {computed:,.2f} {expected_currency} "
                       f"({diff_pct:+.2f}%).".replace(",", "."))
    else:
        st.warning("Tu tabla tiene moneda mixta o la moneda esperada no coincide. "
                   "Para el check automático, pega una tabla con una sola moneda (EUR o USD).")

# -------------------------
# Cálculos landed (sin IVA)
# -------------------------
dfc = df.copy()
dfc["Costo unit CLP"] = dfc.apply(lambda r: fx(r["Moneda"], r["Costo unit"], eurclp, usdclp), axis=1)
dfc["FOB CLP"] = dfc["Costo unit CLP"] * dfc["Qty"]

dfc["Ratio"] = dfc["Formato"].map(RATIOS).fillna(1.0)
dfc["Peso op"] = dfc["Ratio"] * dfc["Qty"]

total_fob = float(dfc["FOB CLP"].sum())
total_peso = float(dfc["Peso op"].sum()) if float(dfc["Peso op"].sum()) else 1.0

# shipping origen -> CLP (asumimos en la moneda predominante; si hay EUR, usamos EUR, si no USD)
ship_curr = "EUR" if "EUR" in currs else ("USD" if "USD" in currs else "EUR")
ship_origin_clp = ship_origin * (eurclp if ship_curr == "EUR" else usdclp)

# Prorrateo fijo (como conversamos):
# - Shipping origen + Proceso entrada => por peso
# - Derechos + IVA importación + IVA agente => por valor (solo para caja / tracking)
shared_peso = ship_origin_clp + float(proceso)

den_w = total_peso
den_v = total_fob if total_fob else 1.0

dfc["Share peso"] = dfc["Peso op"] / den_w
dfc["Share valor"] = np.where(total_fob > 0, dfc["FOB CLP"] / den_v, 0.0)

# Derechos SI al landed (sin IVA)
dfc["Asignado shipping+proceso"] = dfc["Share peso"] * shared_peso
dfc["Asignado derechos"] = dfc["Share valor"] * float(derechos)

dfc["Landed unit sin IVA"] = np.where(
    dfc["Qty"] > 0,
    (dfc["FOB CLP"] + dfc["Asignado shipping+proceso"] + dfc["Asignado derechos"]) / dfc["Qty"],
    0.0
)
dfc["Landed total sin IVA"] = dfc["Landed unit sin IVA"] * dfc["Qty"]

# -------------------------
# Pricing 3 escenarios (IVA incluido, redondeo ...900)
# -------------------------
def scenario_price(mult):
    return (dfc["Landed unit sin IVA"] * mult * (1 + IVA_CL)).apply(round_900_up).astype(int)

dfc["Precio oferta (1,5)"] = scenario_price(1.5)
dfc["Precio MJ (1,7)"] = scenario_price(1.7)
dfc["Precio protegido (2,2)"] = scenario_price(2.2)

dfc["Total ítem oferta"] = (dfc["Precio oferta (1,5)"] * dfc["Qty"]).astype(int)
dfc["Total ítem MJ"] = (dfc["Precio MJ (1,7)"] * dfc["Qty"]).astype(int)
dfc["Total ítem protegido"] = (dfc["Precio protegido (2,2)"] * dfc["Qty"]).astype(int)

# -------------------------
# Resumen simple (costo total vs ingresos escenarios)
# -------------------------
st.subheader("📦 Wrap-up rápido (costo vs ingresos)")

landed_total = float(dfc["Landed total sin IVA"].sum())
sales_oferta = float(dfc["Total ítem oferta"].sum())
sales_mj = float(dfc["Total ítem MJ"].sum())
sales_prot = float(dfc["Total ítem protegido"].sum())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Costo total carga (sin IVA)", fmt_clp(landed_total))
m2.metric("Ingresos Oferta (1,5)", fmt_clp(sales_oferta))
m3.metric("Ingresos MJ (1,7)", fmt_clp(sales_mj))
m4.metric("Ingresos Protegido (2,2)", fmt_clp(sales_prot))

mm1, mm2, mm3 = st.columns(3)
mm1.metric("Margen % Oferta", f"{((sales_oferta - landed_total)/sales_oferta*100 if sales_oferta else 0):.1f}%")
mm2.metric("Margen % MJ", f"{((sales_mj - landed_total)/sales_mj*100 if sales_mj else 0):.1f}%")
mm3.metric("Margen % Protegido", f"{((sales_prot - landed_total)/sales_prot*100 if sales_prot else 0):.1f}%")

st.caption("Margen % calculado sobre landed sin IVA (económico).")

# -------------------------
# Tabla principal + edición manual de precios
# -------------------------
st.subheader("📄 Tabla por producto + edición manual de precios")

st.caption("Puedes editar el precio final por producto (IVA incl, …900). Abajo recalcula ingresos y margen con esa combinación.")

out = dfc[[
    "SKU","Producto","Formato","Qty","Moneda","Costo unit",
    "Costo unit CLP","Landed unit sin IVA","Landed total sin IVA",
    "Precio oferta (1,5)","Precio MJ (1,7)","Precio protegido (2,2)"
]].copy()

# CLP sin decimales
for c in ["Costo unit CLP","Landed unit sin IVA","Landed total sin IVA"]:
    out[c] = out[c].round(0).astype(int)

# Columna editable: precio final por producto (por defecto MJ)
out["Precio final editable"] = out["Precio MJ (1,7)"].astype(int)

edited = st.data_editor(
    out,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Precio final editable": st.column_config.NumberColumn(
            "Precio final editable",
            help="Precio final IVA incl. (idealmente terminado en 900). Puedes pegar en bloque.",
            min_value=0,
            step=1000
        ),
        "SKU": st.column_config.TextColumn("SKU", help="Opcional"),
    },
    disabled=[
        "SKU","Producto","Formato","Qty","Moneda","Costo unit",
        "Costo unit CLP","Landed unit sin IVA","Landed total sin IVA",
        "Precio oferta (1,5)","Precio MJ (1,7)","Precio protegido (2,2)"
    ]
)

# Recalcular rentabilidad con precios editados
edited = edited.copy()
edited["Precio final editable"] = edited["Precio final editable"].apply(money).fillna(0).astype(int)
edited["Total ítem (editable)"] = (edited["Precio final editable"] * edited["Qty"]).astype(int)

landed_total_edit = int(edited["Landed total sin IVA"].sum())
sales_edit = int(edited["Total ítem (editable)"].sum())
margin_edit = ((sales_edit - landed_total_edit) / sales_edit * 100) if sales_edit else 0.0
delta_vs_mj = sales_edit - int(sales_mj)

st.subheader("✅ Rentabilidad con tu combinación de precios (editable)")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Costo total carga (sin IVA)", fmt_clp(landed_total_edit))
c2.metric("Ingresos totales (editable)", fmt_clp(sales_edit))
c3.metric("Margen % (editable)", f"{margin_edit:.1f}%")
c4.metric("Δ ingresos vs MJ (1,7)", fmt_clp(delta_vs_mj))

# Fila total al final (para ver claramente costo vs venta)
st.subheader("Totales (fila final)")
tot_row = {c:"" for c in edited.columns}
tot_row["Producto"] = "TOTAL"
tot_row["Qty"] = int(edited["Qty"].sum())
tot_row["Landed total sin IVA"] = landed_total_edit
tot_row["Total ítem (editable)"] = sales_edit
tot_df = pd.concat([edited, pd.DataFrame([tot_row])], ignore_index=True)
st.dataframe(tot_df, use_container_width=True, hide_index=True)

# Export
csv = tot_df.to_csv(index=False).encode("utf-8")
st.download_button("Descargar CSV", data=csv, file_name="mj_import_pricing_mvp_plus.csv", mime="text/csv")
