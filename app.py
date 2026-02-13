import io, re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MJ Import Pricing (MVP)", layout="wide")
st.title("Import Pricing — Music Jungle (MVP)")
st.caption("Pega productos + costos. Obtén landed sin IVA y 3 precios sugeridos (…900) + wrap-up de la carga.")

# --- Defaults fijos (para simplificar) ---
IVA_CL = 0.19
RATIOS = {"CD": 0.2, "LP": 1.0, "2LP": 1.8, "BOX": 3.2}

def money(x):
    if x is None or (isinstance(x, float) and np.isnan(x)): return np.nan
    s = str(x).strip()
    if not s: return np.nan
    s = re.sub(r"[^\d,.\-]", "", s)
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    else:
        if "," in s: s = s.replace(".", "").replace(",", ".")
    try: return float(s)
    except: return np.nan

def round_900_up(x):
    if x is None or np.isnan(x) or x <= 0: return 0
    step, base = 1000, 900
    n = int(np.ceil((float(x) - base) / step))
    return int(base + n * step)

def parse_table(txt):
    txt = (txt or "").strip()
    if not txt: return pd.DataFrame()
    try:
        df = pd.read_csv(io.StringIO(txt), sep="\t", dtype=str)
        if df.shape[1] <= 1: raise ValueError()
    except:
        df = pd.read_csv(io.StringIO(txt), sep=",", dtype=str)
    df.columns = [c.strip() for c in df.columns]

    # Normaliza nombres
    ren = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ["producto","product","artist-title","title"]: ren[c]="Producto"
        elif cl in ["formato","format","configuration"]: ren[c]="Formato"
        elif cl in ["qty","cantidad","quantity"]: ren[c]="Qty"
        elif cl in ["moneda","currency"]: ren[c]="Moneda"
        elif cl in ["costo unit","costo unitario","net price","unit cost"]: ren[c]="Costo unit"
    df = df.rename(columns=ren)

    for c in ["Producto","Formato","Qty","Moneda","Costo unit"]:
        if c not in df.columns: df[c] = ""

    df["Producto"] = df["Producto"].fillna("").astype(str).str.strip()
    df["Moneda"] = df["Moneda"].fillna("").astype(str).str.strip().str.upper()
    df["Qty"] = df["Qty"].apply(money).fillna(0).astype(int)
    df["Costo unit"] = df["Costo unit"].apply(money)

    # Formato simple: CD / LP / 2LP / BOX
    def fmt(x):
        s = str(x).strip().upper()
        if s in ["CD","1CD","2-CD","2CD","3-CD","3CD"]: return "CD"
        if s in ["LP","1LP","VINYL","VINILO"]: return "LP"
        if s in ["2LP","2-LP","2 LP","3LP","3-LP","3 LP","DOUBLE LP","TRIPLE LP"]: return "2LP"
        if s in ["BOX","BOXSET"]: return "BOX"
        return "LP"
    df["Formato"] = df["Formato"].apply(fmt)
    return df

# --- Inputs ---
left, right = st.columns([1.2, 0.8])

with left:
    st.subheader("1) Productos (pegar tabla)")
    st.caption("Columnas mínimas: Producto, Formato (CD/LP/2LP/BOX), Qty, Moneda (EUR/USD), Costo unit.")
    example = (
        "Producto\tFormato\tQty\tMoneda\tCosto unit\n"
        "Air – Virgin Suicides Redux\tLP\t2\tEUR\t18,50\n"
        "Leftfield – Leftism\t2LP\t2\tEUR\t27,50\n"
        "Beastie Boys – Ill Communication\tCD\t2\tEUR\t5,45\n"
    )
    txt = st.text_area("Tabla", height=180, value=example)

    st.subheader("Shipping origen → aeropuerto (misma moneda factura)")
    ship_origin = st.number_input("Shipping origen (EUR/USD)", min_value=0.0, value=0.0, step=1.0)

with right:
    st.subheader("Tipo de cambio")
    eurclp = st.number_input("EUR → CLP", min_value=1.0, value=1150.0, step=1.0)
    usdclp = st.number_input("USD → CLP", min_value=1.0, value=950.0, step=1.0)

    st.subheader("2) Costos Chile (FedEx) en CLP")
    iva_import = st.number_input("IVA importación (CLP)", min_value=0.0, value=0.0, step=1000.0)
    derechos = st.number_input("Derechos de aduana (CLP)", min_value=0.0, value=0.0, step=1000.0)
    proceso = st.number_input("Proceso de entrada (CLP)", min_value=0.0, value=0.0, step=1000.0)
    iva_agente = st.number_input("IVA agente aduana (CLP)", min_value=0.0, value=0.0, step=1000.0)

run = st.button("Calcular", type="primary")

if not run:
    st.stop()

df = parse_table(txt)
if df.empty:
    st.error("Tabla vacía o inválida.")
    st.stop()

# --- Cálculos ---
def fx(currency, amount):
    if pd.isna(amount): return np.nan
    c = (currency or "").upper()
    if c == "EUR": return amount * eurclp
    if c == "USD": return amount * usdclp
    return amount

dfc = df.copy()
dfc["Costo unit CLP"] = dfc.apply(lambda r: fx(r["Moneda"], r["Costo unit"]), axis=1)
dfc["FOB CLP"] = dfc["Costo unit CLP"] * dfc["Qty"]

# pesos operativos
dfc["Ratio"] = dfc["Formato"].map(RATIOS).fillna(1.0)
dfc["Peso op"] = dfc["Ratio"] * dfc["Qty"]

total_fob = float(dfc["FOB CLP"].sum())
total_peso = float(dfc["Peso op"].sum()) if float(dfc["Peso op"].sum()) else 1.0

# shipping origen -> CLP (si moneda mixta, asumimos shipping en EUR si hay EUR, si no USD)
currs = [c for c in dfc["Moneda"].unique().tolist() if c]
ship_curr = "EUR" if "EUR" in currs else ("USD" if "USD" in currs else "EUR")
ship_origin_clp = ship_origin * (eurclp if ship_curr == "EUR" else usdclp)

# Reglas prorrateo fijas:
# - shipping origen + proceso => por peso
# - derechos + IVA import + IVA agente => por valor
shared_peso = ship_origin_clp + float(proceso)
shared_valor = float(derechos) + float(iva_import) + float(iva_agente)

dfc["Share peso"] = dfc["Peso op"] / total_peso
dfc["Share valor"] = np.where(total_fob > 0, dfc["FOB CLP"] / total_fob, 0.0)

dfc["Asignado (peso)"] = dfc["Share peso"] * shared_peso
dfc["Asignado (valor)"] = dfc["Share valor"] * shared_valor

# Landed SIN IVA: incluye FOB + asignado peso + asignado derechos
# (IVA importación y IVA agente NO se incluyen en landed sin IVA)
dfc["Asignado derechos"] = dfc["Share valor"] * float(derechos)
dfc["Landed unit sin IVA"] = np.where(
    dfc["Qty"] > 0,
    (dfc["FOB CLP"] + dfc["Asignado (peso)"] + dfc["Asignado derechos"]) / dfc["Qty"],
    0.0
)
dfc["Landed total sin IVA"] = dfc["Landed unit sin IVA"] * dfc["Qty"]

# Pricing (IVA incluido + ...900)
def price(mult):
    return (dfc["Landed unit sin IVA"] * mult * (1 + IVA_CL)).apply(round_900_up).astype(int)

dfc["Precio oferta (1,5)"] = price(1.5)
dfc["Precio MJ (1,7)"] = price(1.7)
dfc["Precio protegido (2,2)"] = price(2.2)

dfc["Total ítem oferta"] = (dfc["Precio oferta (1,5)"] * dfc["Qty"]).astype(int)
dfc["Total ítem MJ"] = (dfc["Precio MJ (1,7)"] * dfc["Qty"]).astype(int)
dfc["Total ítem protegido"] = (dfc["Precio protegido (2,2)"] * dfc["Qty"]).astype(int)

# --- Outputs ---
st.subheader("3) Tabla de resultados")

out = dfc[[
    "Producto","Formato","Qty","Moneda","Costo unit",
    "Costo unit CLP","Landed unit sin IVA","Landed total sin IVA",
    "Precio oferta (1,5)","Total ítem oferta",
    "Precio MJ (1,7)","Total ítem MJ",
    "Precio protegido (2,2)","Total ítem protegido",
]].copy()

# CLP sin decimales
for c in ["Costo unit CLP","Landed unit sin IVA","Landed total sin IVA"]:
    out[c] = out[c].round(0).astype(int)

# Fila total
total_row = {c:"" for c in out.columns}
total_row["Producto"] = "TOTAL"
total_row["Qty"] = int(out["Qty"].sum())
total_row["Landed total sin IVA"] = int(out["Landed total sin IVA"].sum())
total_row["Total ítem oferta"] = int(out["Total ítem oferta"].sum())
total_row["Total ítem MJ"] = int(out["Total ítem MJ"].sum())
total_row["Total ítem protegido"] = int(out["Total ítem protegido"].sum())
out = pd.concat([out, pd.DataFrame([total_row])], ignore_index=True)

st.dataframe(out, use_container_width=True, hide_index=True)

st.subheader("4) Wrap-up de la carga (costo vs ingresos)")

landed_total = float(dfc["Landed total sin IVA"].sum())
sales_oferta = float(dfc["Total ítem oferta"].sum())
sales_mj = float(dfc["Total ítem MJ"].sum())
sales_prot = float(dfc["Total ítem protegido"].sum())

c1,c2,c3,c4 = st.columns(4)
c1.metric("Costo total carga (sin IVA)", f"{int(landed_total):,}".replace(",", "."))
c2.metric("Ingresos oferta (1,5)", f"{int(sales_oferta):,}".replace(",", "."))
c3.metric("Ingresos MJ (1,7)", f"{int(sales_mj):,}".replace(",", "."))
c4.metric("Ingresos protegido (2,2)", f"{int(sales_prot):,}".replace(",", "."))

st.caption("Margen % aquí es económico (sobre costo sin IVA).")
m5,m6,m7 = st.columns(3)
m5.metric("Margen % oferta", f"{((sales_oferta - landed_total)/sales_oferta*100 if sales_oferta else 0):.1f}%")
m6.metric("Margen % MJ", f"{((sales_mj - landed_total)/sales_mj*100 if sales_mj else 0):.1f}%")
m7.metric("Margen % protegido", f"{((sales_prot - landed_total)/sales_prot*100 if sales_prot else 0):.1f}%")

# Export
csv = out.to_csv(index=False).encode("utf-8")
st.download_button("Descargar CSV", data=csv, file_name="mj_import_pricing_mvp.csv", mime="text/csv")
