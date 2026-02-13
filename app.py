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
    if
