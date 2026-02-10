# -*- coding: utf-8 -*-
"""
Dynamic PyPSA Dashboard
"""


# %% Imports
import os
import re
import math
from functools import lru_cache
from collections import Counter

import pypsa
import numpy as np
import pandas as pd

from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State

import plotly.express as px
import plotly.graph_objects as go

import threading
import traceback
import copy
import plotly.io as pio

# Basistemplate beibehalten (Schriftart Diagramme)
pio.templates["plotly_ari"] = copy.deepcopy(pio.templates["plotly"])
pio.templates["plotly_ari"].layout.font.family = "Arial"
pio.templates.default = "plotly_ari"

# %% Konfiguration: Datenverzeichnis + Cachegröße

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Pfad für die .nc-Dateien
DATA_DIR = BASE_DIR

# LRU Cache: wie viele Datasets (inkl. pypsa.Network + abgeleitete Tabellen/Figuren) gleichzeitig gehalten werden
LRU_CACHE_SIZE = 4

# Globaler Lock für mehr Stabilität
_STATE_BUILD_LOCK = threading.Lock()

#%% Carrier/Subcarrier Taxonomie

CARRIER_SEP = "_"
DEFAULT_SUBCARRIER = "Sonstige"
KNOWN_SECTORS = ("Strom", "Wärme")  # alles andere -> "Sonstige"
SECTORS = ["Strom", "Wärme", "Sonstige"]


def split_carrier_subcarrier(raw, sep: str = CARRIER_SEP, default_sub: str = DEFAULT_SUBCARRIER) -> tuple[str, str]:
    """
    Zerlegt einen Carrier-String im Format 'Carrier_Subcarrier' in (Carrier, Subcarrier). 
    Falls kein Subcarrier vorhanden ist, wird ein Default-Subcarrier (Sonstige) gesetzt.
    
    Input: Inhalt des Carrier-Attributs
    
    Output: Carrier, Subcarrier
    raw = "Carrier_Subcarrier" -> ("Carrier","Subcarrier")
    raw = "Carrier"            -> ("Carrier", default_sub)
    raw = NaN/None             -> ("", default_sub)
    """
    if raw is None or pd.isna(raw):
        return ("", default_sub)
    s = str(raw).strip()
    if not s:
        return ("", default_sub)
    if sep in s:
        a, b = s.split(sep, 1)  # nur am ersten Separator splitten
        a = a.strip()
        b = b.strip()
        return (a if a else "", b if b else default_sub)
    return (s, default_sub)


def sector_subcarrier_from_raw_carrier(raw_carrier) -> tuple[str, str]:
    """
   Leitet aus einem raw carrier den Sektor (nur bekannte Sektoren) und den Subcarrier ab.
       
   Inputs: raw_carrier: Carrier-Feld (z.B. aus n.buses['carrier'] oder r['carrier']).
       
   Ruft split_carrier_subcarrier auf.
   Setzt sector = carrier, wenn carrier in KNOWN_SECTORS, sonst 'Sonstige'.
   Stellt sicher, dass subcarrier nicht leer ist (Default).
       
   Outputs: Tuple[str, str]: (sector, subcarrier)
    """
    carrier, sub = split_carrier_subcarrier(raw_carrier)
    sector = carrier if carrier in KNOWN_SECTORS else "Sonstige"
    if not sub:
        sub = DEFAULT_SUBCARRIER
    return sector, sub


def ensure_bus_taxonomy(n: pypsa.Network) -> None:
    """
    Erzeugt/aktualisiert die Spalten n.buses['sector'] und n.buses['subcarrier'] aus
    n.buses['carrier'].
    
    Inputs: n.pypsa.Network
    
    Abbruch, wenn n.buses fehlt/ leer ist
    Stellt sicher, dass Spalte "carrier" existiert (sonst NA)
    Iteriert über alle Buses, mappt Carrier -> Sector, Subcarrier
    Schreibt die Series als String in n.buses
    
    Outputs: Keine (Inplace-Modifikation von n.buses)        
    """
    if not hasattr(n, "buses") or n.buses is None or n.buses.empty:
        return
    if "carrier" not in n.buses.columns:
        n.buses["carrier"] = pd.NA

    sec = []
    sub = []
    for v in n.buses["carrier"].tolist():
        s, sc = sector_subcarrier_from_raw_carrier(v)
        sec.append(s)
        sub.append(sc)

    n.buses["sector"] = pd.Series(sec, index=n.buses.index, dtype="string")
    n.buses["subcarrier"] = pd.Series(sub, index=n.buses.index, dtype="string")


def sector_subcarrier_from_bus(n: pypsa.Network, bus_name: str) -> tuple[str, str]:
    """
    Gibt (sector, subcarrier) für einen spezifischen Busnamen zurück, bevorzugt aus
    n.buses['sector'/'subcarrier'], sonst aus n.buses['carrier'].
    
    Inputs: n.pypsa.Network, bus_name
    
    Validiert bus_name und Existenz in n.buses.
    Wenn 'sector' und 'subcarrier' existieren: liest Werte robust (NA -> Default).
    Sonst: ruft sector_subcarrier_from_raw_carrier auf Basis von n.buses.at[bus_name,'carrier']
    auf.
    
    Outputs: Tuple[str, str]: (sector, subcarrier)
    """
    if not bus_name or bus_name not in n.buses.index:
        return ("Sonstige", DEFAULT_SUBCARRIER)
    if "sector" in n.buses.columns and "subcarrier" in n.buses.columns:
        s = n.buses.at[bus_name, "sector"]
        sc = n.buses.at[bus_name, "subcarrier"]
        s = "Sonstige" if pd.isna(s) else str(s)
        sc = DEFAULT_SUBCARRIER if pd.isna(sc) else str(sc)
        return (s, sc)
    return sector_subcarrier_from_raw_carrier(n.buses.at[bus_name, "carrier"])

def sector_subcarrier_from_component_row(n: pypsa.Network, comp: str, r: pd.Series) -> tuple[str, str]:
    """
    Ermittelt (sector, subcarrier) für eine Komponentenzeile. Für Links/Lines erfolgt die
    Zuordnung über den Bus; sonst bevorzugt über 'carrier' der Komponente, mit Fallback auf
    den Bus.
    
    Inputs: n.pypsa.Network,
            comp (Komponentenname)
            r.pd.Series (Zeile der statischen Komponententabelle)
    
    Für Links/Lines: liest Bus aus r['bus'] und mappt per sector_subcarrier_from_bus.
    Sonst: versucht r['carrier'] zu interpretieren (sector_subcarrier_from_raw_carrier).
    Wenn daraus 'Sonstige' entsteht: Fallback auf r['bus'] (falls vorhanden) und übernimmt ggf.
    spezifischeren Subcarrier.
    
    Outputs: Tuple: [str, str]: (sector, subcarrier)
    """
    if comp in ("links", "lines"):
        b = r.get("bus", None)
        return sector_subcarrier_from_bus(n, b) if b is not None else ("Sonstige", DEFAULT_SUBCARRIER)

    raw_car = r.get("carrier", None)
    s, sc = sector_subcarrier_from_raw_carrier(raw_car)

    if s == "Sonstige":
        b = r.get("bus", None)
        if b is not None:
            s2, sc2 = sector_subcarrier_from_bus(n, b)
            s = s2
            if sc == DEFAULT_SUBCARRIER and sc2 != DEFAULT_SUBCARRIER:
                sc = sc2

    return (s, sc)



#%% Helper: Labels + Farbschemata


def strip_prefix(s: str) -> str:
    """
    Entfernt einen optionalen Komponenten-Prefix 'comp__' aus einem Label.
    
    Inputs: s[str]
    
    Trennt am ersten "__" und gibt den dahinterstehenden Teil des String zurück
    
    Outputs: Inplace-Modifikation, bereinigter String

    """
    s = str(s)
    return s.split("__", 1)[1] if "__" in s else s

# Anzeigenamen für Labels
_VAR_SUFFIX_RE = re.compile(r"(?i)_(variable|variabel|port)$")

def strip_variable_suffix(s: str) -> str:
    """
    Entfernt ein optionales Suffix '_variable' (case-insensitive) aus einem Label.
    
    Inputs: s [str]
    
    Regex-Substition auf dem String
    
    Outputs: Inplace-Modifikation, bereinigter String

    """
    return _VAR_SUFFIX_RE.sub("", str(s))

_PORT_ONLY_SUFFIX_RE = re.compile(r"(?i)_(p|e)$")

def strip_port_suffix_for_hover(label: str) -> str:
    """
    Entfernt nur Endungen _p oder _e am Ende des Labels (für Hover).
    Beispiele:
      'Stromnetz_p'     -> 'Stromnetz'
      'Stromnetz_e'     -> 'Stromnetz'
      'Stromnetz_out1'  -> bleibt unverändert
      'Stromnetz_p (2)' -> 'Stromnetz (2)'
     
    Inputs: label [str]
    
    Separiert optionales Duplikatsuffix ' (n)'.
    entfernt anschließend per Regex nur Endungen '_p' oder '_e' im Kernstring.
    Setzt Duplikatsuffix wieder an.
    
    Outputs: Bereinigtes Hover-Label [str]        
    """
    s = str(label)

    m = re.match(r"^(.*?)(\s\(\d+\))$", s)
    if m:
        core, dup = m.group(1), m.group(2)
    else:
        core, dup = s, ""

    core = _PORT_ONLY_SUFFIX_RE.sub("", core)
    return core + dup

def display_name_map(names: list[str], show_component_on_dupes: bool = False) -> dict[str, str]:
    """
    Erzeugt eine Map raw_name -> Anzeige-Name und behandelt Duplikate (z.B. gleiche Namen
    aus verschiedenen Komponenten) deterministisch.
    
    Inputs: names: list[str] Rohspaltennamen/ Labels)
            component_on_dupes[bool]: wenn True, hängt bei Duplikaten den Komponententyp in
            eckigen Klammern an. (z. B. bei Stores mit separatem Bus relevant, Fallback für
                                  Differenzierbarkeit bei gleicher Benennung)
    
    Erzeugt 'pretty' Namen durch strip_prefix + strip_variable_suffix.
    Zählt Duplikate via Counter.
    Für eindeutige Namen: gibt pretty zurück.
    Für Duplikate: nummeriert '(1)', '(2)' oder hängt Komponententyp an (optional).
    
    Outputs: dict[str, str]: Mapping raw -> display.
    """
    pretties = [strip_variable_suffix(strip_prefix(n)) for n in names]
    cnt = Counter(pretties)

    out = {}
    seen = Counter()

    for raw, pretty in zip(names, pretties):
        if cnt[pretty] > 1:
            if show_component_on_dupes and "__" in str(raw):
                comp = str(raw).split("__", 1)[0]
                out[raw] = f"{pretty} [{comp}]"
            else:
                seen[pretty] += 1
                out[raw] = f"{pretty} ({seen[pretty]})"
        else:
            out[raw] = pretty

    return out

def _unique_preserve(seq):
    """
    Entfernt Duplikate aus einer Sequenz, behält aber die ursprüngliche Reihenfolge bei.
    
    Inputs: sep: Iterable
    
    Iteriert, merkt sich bereits gesehene Werte in einem set, baut eine Ausgabeliste.
    
    Outputs: List, eindeutige Elemente in Eingabereihenfolge

    """
    seen = set()
    out = []
    for x in seq:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _collect_subcarriers(*objs) -> list[str]:
    """
    Sammelt Subcarrier-Werte aus mehreren DataFrames und/oder dict[str->DataFrame] robust
    (None/leer wird ignoriert).
    
    Inputs: objs: beliebige Objekte (DataFrame oder dict)
    
    Iteriert über alle Objekte; extrahiert Spalte 'subcarrier', wenn vorhanden.
    Trimmt Whitespace, entfernt leere Strings.
    Bildet eindeutige Menge und sortiert.
    Platziert DEFAULT_SUBCARRIER ans Ende (falls vorhanden).
    
    Outputs: list[str]: Sortierte Subcarrier-Liste
    """
    vals = []
    for obj in objs:
        if obj is None:
            continue
        if isinstance(obj, dict):
            for _k, df in obj.items():
                if isinstance(df, pd.DataFrame) and (not df.empty) and ("subcarrier" in df.columns):
                    vals.extend(df["subcarrier"].dropna().astype(str).tolist())
        elif isinstance(obj, pd.DataFrame):
            if (not obj.empty) and ("subcarrier" in obj.columns):
                vals.extend(obj["subcarrier"].dropna().astype(str).tolist())
    vals = [v.strip() for v in vals if str(v).strip() != ""]
    uniq = sorted(set(vals))
    if DEFAULT_SUBCARRIER in uniq:
        uniq = [u for u in uniq if u != DEFAULT_SUBCARRIER] + [DEFAULT_SUBCARRIER]
    return uniq


def make_subcarrier_color_map(subcarriers: list[str]) -> dict[str, str]:
    """
    Erstellt eine deterministische Farbzuordnung subcarrier -> Farbe anhand mehrerer
    Plotly-Paletten.
    
    Inputs: subcarriers: list[str]
    
    Kombiniert mehrere qualitative Paletten, entfernt Duplikate.
    Bei Bedarf Palette zyklisch verlängern.
    Zuweisung in der Reihenfolge der Eingabeliste.
    
    Outputs: dict[str;str]: subcarrier => Farbcode
    """
    palettes = (
        px.colors.qualitative.Vivid
        + px.colors.qualitative.Bold
        + px.colors.qualitative.Dark24
        + px.colors.qualitative.Alphabet
    )
    colors = _unique_preserve(palettes)

    if not subcarriers:
        return {}

    if len(subcarriers) > len(colors):
        rep = int(math.ceil(len(subcarriers) / len(colors)))
        colors = (colors * rep)[:len(subcarriers)]
    else:
        colors = colors[:len(subcarriers)]

    return {sc: col for sc, col in zip(subcarriers, colors)}


def make_label_color_map(labels: list[str]) -> dict[str, str]:
    """
    Erstellt eine deterministische Farbzuordnung für beliebige Labels (z.B. Zeitreihen-Spalten).
    
    Inputs: labels: list[str]

    Kombiniert Paletten, erzeugt eindeutige Farbenliste.
    normalisiert/ trimmt Labels, bildet eindeutige sortierte Menge.
    Bei Bedarf Palette zyklisch verlängern.
    
    Outputs: dict[str,str]: label => Farbe
    """
    palettes = (
        px.colors.qualitative.Vivid
        + px.colors.qualitative.Bold
        + px.colors.qualitative.Dark24
        + px.colors.qualitative.Alphabet
    )
    colors = _unique_preserve(palettes)

    labels = [str(l) for l in labels if str(l).strip() != ""]
    uniq = sorted(set(labels))
    if not uniq:
        return {}

    if len(uniq) > len(colors):
        rep = int(math.ceil(len(uniq) / len(colors)))
        colors = (colors * rep)[:len(uniq)]
    else:
        colors = colors[:len(uniq)]

    return {lab: col for lab, col in zip(uniq, colors)}


def make_cost_color_map() -> dict[str, str]:
    """
    Definiert ein konsistentes Farbschema für Kostenarten (CAPEX/OPEX).
    
    Inputs: Keine
    
    Nimmt erste zwei Farben der Plotly-Vivid-Palette (mit Fallback-Hexcodes).
    
    Outputs: dict[str,str]: Mapping Kostenarten => Farben

    """
    vivid = px.colors.qualitative.Vivid
    return {
        "CAPEX": vivid[0] if len(vivid) > 0 else "#1f77b4",
        "OPEX":  vivid[1] if len(vivid) > 1 else "#ff7f0e",
    }

COST_COLOR_MAP = make_cost_color_map()



#%% Helper: JahresFilter für Nennleistungen/ Kapazitäten


def _filter_df_sector_years(df_sector: pd.DataFrame, selected_years, years_all: list[int]) -> tuple[pd.DataFrame, list[int]]:
    """
    Filtert einen sektorspezifischen DataFrame nach ausgewählten Investitionsperioden (Jahre).
    
    Inputs: df_sector: Sektoren-DataFrame mit Spalte "year"
            selected_years: Iterable (z. B. aus Dropdown)
            years_all: list[int] aller vorhandenen Jahre im Dataset.
            
    Wenn keine MIP-Jahre vorhanden: gibt DF unverändert zurück.
    Wenn selected_years leer: gibt leeres DF zurück.
    Konvertiert selected_years robust nach int.
    Filtert df_sector['year'] auf die Schnittmenge.
    
    Outputs: tuple[DataFrame, list[int]]: (gefiltertes DF, gefilterte Jahre).
    """
    if df_sector is None or df_sector.empty:
        return df_sector, years_all

    if not years_all:
        return df_sector, years_all  # Single-year: keine Filterung

    if not selected_years:
        return df_sector.iloc[0:0].copy(), []

    sel = []
    for y in selected_years:
        try:
            sel.append(int(y))
        except Exception:
            pass

    years_f = [y for y in years_all if y in set(sel)]
    if not years_f:
        return df_sector.iloc[0:0].copy(), []

    dff = df_sector.copy()
    dff = dff[dff["year"].astype(int).isin(years_f)]
    return dff, years_f

#%% Helper: Capacities auf aktive Assets statt neugebaute Assets beziehen

def expand_caps_to_active_periods(
    df_caps: pd.DataFrame,
    df_life: pd.DataFrame,
    years: list[int],
    value_col: str = "p_nom",
) -> pd.DataFrame:
    """
    Transformiert 'Zubau je Build' in 'aktive Kapazität je Investitionsperiode' unter
    Berücksichtigung von build_year und end_year (Lifetime).
    
    Inputs: df_caps: DataFrame mit Kapazitätswerten (z.B. p_nom/e_nom) und Spalten: 
            component,name
            df_life: DataFrame mit Lebensdauerinformationen: component, name, 
            build_year, end_year.
            years: list[int] Investitionsperioden.
            value_col: Spaltenname der Kapazität (Default 'p_nom')
    
    Single-year: ergänzt 'year' als leerer String und gibt DF zurück.
    Merged df_caps mit df_life (left join).
    Fallback: fehlende build_year -> horizon_start; end_year -> inf.
    Mappt build_year auf nächste Investitionsperiode (build_period).
    Erzeugt je Periode p alle Anlagen, die build_period <= p < end_year erfüllen.
    Verkettung der Teilmengen; bereinigt Hilfsspalten.
    
    Outputs: DataFrame: Erweiterte Tabelle mit zusätzlicher Spalte 'year'.
        
    """
    if df_caps is None or df_caps.empty:
        return df_caps

    # Single-year: keine Werte zur Verarbeitung
    if not years:
        out = df_caps.copy()
        out["year"] = ""
        return out

    # Life-Map
    life_cols = ["component", "name", "build_year", "end_year"]
    life = df_life[life_cols].copy() if (df_life is not None and not df_life.empty) else pd.DataFrame(columns=life_cols)

    d = df_caps.copy()
    d = d.merge(life, on=["component", "name"], how="left")

    # Fallbacks, wenn build/end fehlen (z.B. keine lifetime/build_year gesetzt)
    horizon_start = int(min(years))
    d["build_year"] = pd.to_numeric(d["build_year"], errors="coerce").fillna(horizon_start).astype(int)
    d["end_year"] = pd.to_numeric(d["end_year"], errors="coerce")
    d["end_year"] = d["end_year"].where(np.isfinite(d["end_year"]), np.inf)
    d["end_year"] = d["end_year"].fillna(np.inf)

    # Optional: falls build_year nicht exakt auf einer Investitionsperiode liegt, auf nächste Periode mappen
    years_sorted = sorted(int(y) for y in years)

    def map_to_period(by: int) -> int:
        if by in years_sorted:
            return by
        future = [y for y in years_sorted if y >= by]
        return future[0] if future else years_sorted[-1]

    d["build_period"] = d["build_year"].apply(map_to_period).astype(int)

    # Expand: jede Zeile in jede Periode, in der sie aktiv ist
    parts = []
    for p in years_sorted:
        m = d[(d["build_period"] <= p) & (p < d["end_year"])].copy()
        if m.empty:
            continue
        m["year"] = int(p)
        parts.append(m)

    if not parts:
        out = d.iloc[0:0].copy()
        out["year"] = []
        return out

    out = pd.concat(parts, ignore_index=True)
    out = out.drop(columns=["build_period"], errors="ignore")
    return out



#%% Zeitreihen: Aufbau (dyn -> df_dyn_all + meta)


def _timestep_and_period_from_df(df: pd.DataFrame):
    """
    Extrahiert aus einem Zeitreihen-DataFrame den Zeitschritt-Index und optional die
    Investitionsperiode.
    
    Inputs: df: DataFrame mit Zeitindex, ggf. MultiIndex (period, snapshot) oder Tuple-Index
    
    Wenn MultiIndex: timestep = letzter Level, period = Level 0
    Wenn Tuple-Index: timestep = letzter Tuple-Teil, period = erster Tuple-Teil
    Sonst: timestep = Index, period = None
    
    Outputs: (pd.Index timestep, pd.Series|None period)
    """
    idx = df.index

    if isinstance(idx, pd.MultiIndex):
        period = idx.get_level_values(0)
        timestep = idx.get_level_values(-1)
        return pd.Index(timestep, name="timestep"), pd.Series(period, name="period")

    # Tuple-Index abfangen
    if len(idx) > 0 and isinstance(idx[0], tuple) and len(idx[0]) >= 2:
        period = [t[0] for t in idx]
        timestep = [t[-1] for t in idx]  # typischerweise Timestamp
        return pd.Index(timestep, name="timestep"), pd.Series(period, name="period")

    return pd.Index(idx, name="timestep"), None



def _nonempty_bus_mask(s: pd.Series) -> pd.Series:
    """
    Hilfsfunktion: erzeugt eine Maske für nicht-leere Bus-Strings in einer Series.
    
    Inputs: s: pd.series
    
    Konvertiert nach String-Dtype, prüft notna und strip != ''
    
    Outputs: pd.Series[bool]: True für gültige Busnamen.

    """
    s2 = s.astype("string")
    return s2.notna() & (s2.str.strip() != "")


def get_existing_link_ports(n: pypsa.Network, max_i: int = 9) -> list[int]:
    """
    Ermittelt, welche Link-Ports (bus0..busN) im Netzwerk tatsächlich belegt sind.
    
    Inputs: · n: pypsa.Network
              max_i: int (Default 9)
              
    Startet immer mit Port 0.
    Prüft für i=1..max_i, ob Spalte bus{i} existiert und mindestens ein nicht-leerer Eintrag
    vorhanden ist
    
    Outputs: list[int]: Liste der existierenden Ports
    """
    ports = [0]
    if not hasattr(n, "links") or n.links.empty:
        return ports
    df = n.links
    for i in range(1, max_i + 1):
        col = f"bus{i}"
        if col in df.columns and _nonempty_bus_mask(df[col]).any():
            ports.append(i)
    return ports


def links_with_bus_i(n: pypsa.Network, i: int) -> list[str]:
    """
    Gibt die Namen der Links zurück, für die bus{i} gesetzt ist (für i=0 und i=1 alle Links).
    
    Inputs: n: pypsa.Network
            i: int
            
    Für i=0: alle Link-Indizes.
    Für i>0: filtert n.links nach nicht-leeren bus{i}.
    
    Outputs: list[str]: Link-Namen
    """
    if i == 0:
        return list(n.links.index)
    col = f"bus{i}"
    if col not in n.links.columns:
        return []
    return n.links.index[_nonempty_bus_mask(n.links[col])].astype(str).tolist()


def build_dynamic_timeseries_df(
    n: pypsa.Network,
    components=None,
    add_component_prefix: bool = False,
    make_link_line_ports_positive: bool = True,
) -> pd.DataFrame:
    """
    Baut einen  Zeitreihen-DataFrame aus dynamischen PyPSA-Komponenten
    (links/generators/loads/stores/storage_units/lines) inklusive Periode.
    
    Inputs: n: pypsa.Network
            components: Liste der zu verarbeitenden Komponenten (Default: typische Komponenten)
            add_component_prefix: wenn True, Spaltenformat 'component__asset_attr', 
            sonst asset_attr
            make_link_line_ports_positive: wenn True, nimmt abs() für p-Ports von links/lines
    
    Iteriert über Komponenten und wählt pro Komponente geeignete Attribute (p, p_set, p0...pn)
    Für Links: berücksichtigt nur Ports, deren bus{i} gesetzt ist und filtert die Spalten
    entsprechend
    Normalisiert Index über _timestep_and_period_from_df (timestep als Index; period in
    separater Series)
    Benennung der Spalten nach Standard 'Komponententyp__Komponentenname_Variable'
    Verkettung nach Attributblöcken (p0,p1,... zuerst)
    Fügt 'period' ein (oder Platzhalter) und setzt Index als Spalte 'timestep' (reset_index)
    
    Outputs: DataFrame: Spalten 'period', 'timestep' plus Zeitreihen-Spalten.
    """

    if components is None:
        components = ["links", "generators", "loads", "stores", "storage_units", "lines"]

    frames_by_attr = {}
    period_series = None

    for comp_name in components:
        if not hasattr(n, "components") or not hasattr(n.components, comp_name):
            continue
        if not hasattr(n, comp_name):
            continue

        # dynamische tables: via n.components API (wie in deinem Stand), aber abgesichert
        if not hasattr(n, "components") or not hasattr(n.components, comp_name):
            continue
        comp = getattr(n.components, comp_name)
        dyn = comp.dynamic

        if comp_name == "links":
            ports = get_existing_link_ports(n, max_i=9)
            attrs = [f"p{i}" for i in ports if f"p{i}" in dyn]
        elif comp_name == "lines":
            attrs = [a for a in ("p0", "p1") if a in dyn]
        else:
            if "p" in dyn:
                attrs = ["p"]
            elif "p_set" in dyn:
                attrs = ["p_set"]
            else:
                attrs = []

        for attr in attrs:
            df = dyn.get(attr)
            if df is None or df.shape[1] == 0:
                continue

            df2 = df.copy()

            # Links: nur echte Ports (bus{i} gesetzt)
            if comp_name == "links" and re.match(r"^p(\d+)$", str(attr)):
                i = int(re.match(r"^p(\d+)$", str(attr)).group(1))
                valid = set(links_with_bus_i(n, i))
                if not valid:
                    continue
                df2 = df2.loc[:, [c for c in df2.columns.astype(str) if c in valid]]
                if df2.shape[1] == 0:
                    continue

            t_idx, p_ser = _timestep_and_period_from_df(df2)
            df2.index = t_idx

            if p_ser is not None and period_series is None:
                period_series = pd.Series(p_ser.values, index=t_idx, name="period")

            if make_link_line_ports_positive and comp_name in ("links", "lines") and re.match(r"^p\d+$", str(attr)):
                df2 = df2.abs()
            # Spaltenbenennung für die Zeitreihen "Komponententyp__Komponentenname_Variable"
            if add_component_prefix:
                df2.columns = [f"{comp_name}__{col}_{attr}" for col in df2.columns]
            else:
                df2.columns = [f"{col}_{attr}" for col in df2.columns]

            frames_by_attr.setdefault(attr, []).append(df2)

    if not frames_by_attr:
        return pd.DataFrame(columns=["period", "timestep"])

    port_attrs = sorted([a for a in frames_by_attr.keys() if re.match(r"^p\d+$", a)], key=lambda s: int(s[1:]))
    other_attrs = [a for a in frames_by_attr.keys() if a not in port_attrs]
    attr_order = port_attrs + other_attrs

    parts = []
    for a in attr_order:
        part = pd.concat(frames_by_attr[a], axis=1)
        part = part.reindex(columns=sorted(part.columns))
        parts.append(part)

    out = pd.concat(parts, axis=1)

    if period_series is not None:
        out.insert(0, "period", period_series.loc[out.index].values)
    else:
        out.insert(0, "period", "Nur ein Zeitraum vorhanden")

    out = out.reset_index()  # timestep wird zur Spalte
    return out


def infer_internal_store_buses(n: pypsa.Network) -> set[str]:
    """
    Identifiziert Busse, die nur von Stores genutzt werden (interne Speicherbusse), um sie z.B.
    aus bestimmten Plots auszuschließen.
    
    Inputs: n: pypsa.Network
    
    Sammelt Busse aus Stores.
    Sammelt Systembusse aus Loads, Generators, Storage Units.
    Gibt Store-Busse zurück, die nicht in den Systembussen vorkommen.
    
    Outputs: set[str]: interne Store-Buses
    """
    if not hasattr(n, "buses") or n.buses.empty:
        return set()

    store_buses = set()
    if hasattr(n, "stores") and not n.stores.empty and "bus" in n.stores.columns:
        store_buses = set(n.stores["bus"].dropna().astype(str))

    if not store_buses:
        return set()

    load_buses = set(n.loads["bus"].dropna().astype(str)) if hasattr(n, "loads") and not n.loads.empty and "bus" in n.loads.columns else set()
    gen_buses  = set(n.generators["bus"].dropna().astype(str)) if hasattr(n, "generators") and not n.generators.empty and "bus" in n.generators.columns else set()
    su_buses   = set(n.storage_units["bus"].dropna().astype(str)) if hasattr(n, "storage_units") and not n.storage_units.empty and "bus" in n.storage_units.columns else set()

    system_buses = load_buses | gen_buses | su_buses
    return {b for b in store_buses if b not in system_buses}


def parse_ts_col(col: str):
    """
    Parst eine Zeitreihen-Spaltenbezeichnung im Format 'component__asset_attr' in
    (component, asset, attr).
    
    Inputs: col: str
    
    Splittet an '__'.
    Splittet Rest an letztem '_' (rsplit) in asset und attr.
    Gibt None zurück, wenn Format nicht passt.
    
    Outputs: tuple[str,str,str] | None
    """
    if "__" not in col:
        return None
    comp, rest = col.split("__", 1)
    if "_" not in rest:
        return None
    asset, attr = rest.rsplit("_", 1)
    return comp, asset, attr


def infer_bus_for_timeseries(n: pypsa.Network, comp: str, asset: str, attr: str):
    """
    Leitet aus (component, asset, attr) den zugehörigen Bus ab (insb. für Link-Ports bus{i})
    
    Inputs: n: pypsa.Network
            comp: str
            asset: str
            attr: str
            
    Für links/lines und attr 'p{i}': nutzt bus{i}
    Sonst: nutzt generisches 'bus'-Feld der statischen Tabelle, falls vorhanden
    
    Outputs: bus[str] oder None
    """
    m = re.match(r"^p(\d+)$", str(attr))
    if comp in ("links", "lines") and m:
        i = int(m.group(1))
        df = getattr(n, comp, None)
        if df is None or asset not in df.index:
            return None
        bus_col = f"bus{i}"
        if bus_col in df.columns:
            b = df.at[asset, bus_col]
            return None if pd.isna(b) else b
        return None

    df = getattr(n, comp, None)
    if df is None or df.empty or asset not in df.index:
        return None
    if "bus" in df.columns:
        b = df.at[asset, "bus"]
        return None if pd.isna(b) else b
    return None


def build_timeseries_meta(n: pypsa.Network, df_dyn_all: pd.DataFrame, internal_store_buses: set[str]) -> pd.DataFrame:
    """
    Erstellt Metadaten pro Zeitreihen-Spalte: Komponente, Asset, Attr, Bus, Sektor/Subcarrier
    sowie Flag für interne Speicherbuses.
    
    Inputs: n: pypsa.Networt
            df_dyn_all: DataFrame aus build_dynamic_timeseries (flat)
            internal_store_buses: set[str] aus infer_internal_store_buses
    
    Iteriert über alle Spalten außer 'timestep'
    Parst Spaltennamen via parse_ts_col; überspringt unpassende Spalten
    Leitet Bus per infer_bus_for_timeseries ab
    Bestimmt sector/subcarrier über sector_subcarrier_from_bus
    Markiert Busse, die als internal_store_buses erkannt wurden
    Setzt Index der Meta-Tabelle auf die Spaltennamen ('col')
    
    Outputs: DataFrame: Meta-Table mit Index "coL"
    """
    cols = [c for c in df_dyn_all.columns if c != "timestep"]
    rows = []
    for col in cols:
        parsed = parse_ts_col(col)
        if parsed is None:
            continue
        comp, asset, attr = parsed
        bus = infer_bus_for_timeseries(n, comp, asset, attr)
        if comp == "links" and re.match(r"^p\d+$", str(attr)) and bus is None:
            continue

        sector, sub = sector_subcarrier_from_bus(n, bus)
        rows.append({
            "col": col,
            "component": comp,
            "asset": asset,
            "attr": attr,
            "bus": bus,
            "sector": sector,
            "subcarrier": sub,
            "is_internal_store_bus": (bus is not None and str(bus) in internal_store_buses)
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("col")


def insert_nan_breaks(data: pd.DataFrame, gap_factor: float = 3.0) -> pd.DataFrame:
    """
    Fügt NaN-Zeilen ein, um in Linienplots sichtbare Unterbrechungen bei großen Zeitsprüngen
    zu erzeugen.
    
    Inputs: data: DataFrame mit DatetimeIndex
            gap_factor: float (Default 3.0)
    
    Berechnet Zeitdifferenzen zwischen Zeitschritten
    Ermittelt mediane Schrittweite (step) und definiert gap_threshold = step*gap_factor
    Für Deltas > threshold: fügt einen NaN-Zeitpunkt unmittelbar vor der Lücke ein
    Verkettung und Sortierung
    
    Outputs: DataFrame wie Input, nur ergänzt um NaN-Zeilen
    """
    data = data.sort_index()
    dt = data.index.to_series().diff()
    dt_pos = dt[dt > pd.Timedelta(0)]
    if dt_pos.empty:
        return data

    step = dt_pos.median()
    gap_threshold = step * gap_factor
    gap_mask = dt > gap_threshold
    if not gap_mask.any():
        return data

    break_times = data.index[gap_mask] - pd.Timedelta("1ns")
    break_times = break_times[~break_times.isin(data.index)]
    if len(break_times) == 0:
        return data

    breaks = pd.DataFrame(np.nan, index=break_times, columns=data.columns)
    breaks = breaks.astype(data.dtypes.to_dict(), errors="ignore")
    return pd.concat([data, breaks], axis=0).sort_index()


def build_sector_timeseries_fig(
    df_dyn_all: pd.DataFrame,
    meta: pd.DataFrame,
    sector: str,
    unit: str = "kW",
    max_traces: int = 30,
    default_component_visible: str = "generators",
    ts_color_map: dict[str, str] | None = None,
) -> go.Figure:
    """
    Erstellt ein Plotly-Linienplot für Zeitreihen eines Sektors, 
    inkl. Hover-Infos und optionaler Farbzuteilung.
    
    Inputs: df_dyn_all: DataFrame mit 'timestep' und Zeitreihen-Spalten
            meta: Meta-DataFrame (Index = Spaltennamen) aus build_timeseries_meta
            sector: 'Strom'/'Wärme'/'Sonstige'
            unit: String, z.B. 'kW'
            max_traces: maximale Anzahl dargestellter Zeitreihen (Top-N nach Peak)
            default_component_visible: welche Komponente standardmäßig sichtbar ist,
            andere "legendonly"
            ts_color_map: optional dict col->color
    
    Filtert meta nach sektor und blendet interne Store-Link-Busse aus
    Zieht entsprechende Spalten aus df_dyn_all und setzt DatetimeIndex
    Sortiert, fügt NaN-Breaks ein
    Rangiert Spalten nach absolutem Peak und wählt Top-N
    Erzeugt pro Spalte eine Scatter-Linie mit Hovertemplate (Asset, Subcarrier, Variable)
    Konfiguriert Achsen, Titel und Legende
    
    Outputs: go.figure: Plotly-Abbildung
    """

    fig = go.Figure()

    if meta is None or meta.empty or "timestep" not in df_dyn_all.columns:
        fig.update_layout(title=f"Zeitreihen ({sector}) [{unit}] (keine Daten)")
        return fig

    m = meta[meta["sector"] == sector].copy()
    m = m[~((m["component"] == "links") & (m["is_internal_store_bus"] == True))]

    cols = m.index.tolist()
    if not cols:
        fig.update_layout(title=f"Zeitreihen ({sector}) [{unit}] (keine Daten)")
        return fig

    t = pd.to_datetime(df_dyn_all["timestep"])
    data = df_dyn_all[cols].copy()
    data.index = t
    data = data.sort_index()
    if data.empty:
        fig.update_layout(title=f"Zeitreihen ({sector}) [{unit}] (keine Daten)")
        return fig

    data = insert_nan_breaks(data, gap_factor=3.0)

    peak = data.abs().max().sort_values(ascending=False)
    cols_sorted = peak.index.tolist()
    cols_plot = cols_sorted[:max_traces] if (max_traces is not None and len(cols_sorted) > max_traces) else cols_sorted

    for col in cols_plot:
        comp = col.split("__", 1)[0] if "__" in col else ""
        vis = True if comp == default_component_visible else "legendonly"

        asset = meta.at[col, "asset"] if col in meta.index else strip_prefix(col)
        attr  = meta.at[col, "attr"]  if col in meta.index else ""

        sc = None
        if col in meta.index and "subcarrier" in meta.columns:
            sc = meta.at[col, "subcarrier"]
            sc = DEFAULT_SUBCARRIER if pd.isna(sc) else str(sc)

        line_kwargs = {}
        if ts_color_map is not None:
            ccol = ts_color_map.get(str(col))
            if ccol:
                line_kwargs["color"] = ccol

        fig.add_trace(go.Scatter(
            name=asset,
            x=data.index,
            y=data[col].values,
            mode="lines",
            connectgaps=False,
            visible=vis,
            line=line_kwargs if line_kwargs else None,
            hovertemplate=(
                f"{asset}<br>"
                f"Energieträger: {sc if sc is not None else ''}<br>"
                f"Variable: {attr}<br>"
                "%{x}<br>%{y:.2f} " + unit +
                "<extra></extra>"
            )
        ))

    fig.update_layout(
        title=f"Zeitreihen ({sector}) [{unit}]",
        xaxis_title="Zeit",
        yaxis_title=f"Leistung [{unit}]",
        legend_title="Komponente",
        margin=dict(l=30, r=30, t=60, b=50),
    )
    return fig

#%% Statische Tables + Kapazitäten + Ausbaupfad + Lifetime

def _default_static_components(n):
    """
    Ermittelt eine konservative Liste statischer Tabellen, die im Netzwerk vorhanden sind.
    
    Inputs: n: pypsa.Network
    
    Prüft Kandidatenliste
    (buses, carriers, generators, links, loads, stores, storage_units, lines)
    Behält Komponenten, die als Attribut am Netzwerk oder unter n.components existieren
    
    Outputs: list[str]: Vorhandene Komponenten
    """
    candidates = [
        "buses", "carriers",
        "generators", "links", "loads",
        "stores", "storage_units",
        "lines",
    ]
    # pypsa: n.components.* existiert ggf., wird dennoch konservativ geprüft
    out = []
    for x in candidates:
        if hasattr(n, x) or (hasattr(n, "components") and hasattr(n.components, x)):
            out.append(x)
    return out


def get_investment_years(n: pypsa.Network):
    """
    Liest Investitionsperioden (Jahre) aus einem PyPSA-Netzwerk, falls MIP aktiv ist.
    
    Inputs: n: pypsa.Network
    
    Wenn n.has_investment_periods True: gibt n.investment_periods als int-Liste zurück,
    sonst []
    
    Outputs: list[int]    
    """
    if getattr(n, "has_investment_periods", False):
        return [int(y) for y in list(n.investment_periods)]
    return []


def split_base_and_year(name: str, years_set: set[int]):
    """
    Trennt einen Namenssuffix '_YYYY' ab, wenn YYYY eine der Investitionsperioden ist.
    
    Inputs: name: str
            years_set: set[int]
            
    Regex sucht Suffix _[4 Zeichen]
    Wenn Jahr in years_set: gibt (basename, year) zurück, sonst (name, None)
    
    Outputs: (base_name: str, year: int|None)
    """
    name = str(name)
    m = re.search(r"_(\d{4})$", name)
    if m:
        y = int(m.group(1))
        if y in years_set:
            return name[:-5], y
    return name, None


def nominal_from_static(df_static: pd.DataFrame) -> pd.Series:
    """
    Liest aus einer statischen Komponententabelle den passenden Nennleistungs- bzw.
    Kapazitätsspaltenvektor
    
    Inputs: df_static: DataFrame
    
    Sucht in Reihenfolge p_nom_opt, p_nom, s_nom_opt, s_nom; sonst leere float-Serie
    
    Outputs: pd.Series: Nennleistungen/ Kapazitäten
    """
    for col in ("p_nom_opt", "p_nom", "s_nom_opt", "s_nom"):
        if col in df_static.columns:
            return df_static[col]
    return pd.Series(index=df_static.index, dtype=float)


def build_capacity_table(n: pypsa.Network) -> pd.DataFrame:
    """
    Erstellt eine tabellarische Übersicht der Nennleistungen je Asset, inkl. Ports (bei Links:
    in/out) und Zuordnung zu Sektor/Subcarrier
    
    Inputs: n: pypsa.Network
    
    Ermittelt MIP-Jahre und years_set
    Iteriert über generators, storage_units, links, sofern vorhanden
    Für generators/storage_units: p_nom aus nominal_from_static; Name -> (base, year) via
    split_base_and_year
    Für links: ermittelt bus0 (in) und bus1..bus9 (out) und berücksichtigt Effizienzen 
    (efficiency, efficiency2 etc.)
    Erzeugt Zeilen mit: sector, subcarrier, component, name, base_name, year, port, p_nom
    
    Outputs: DataFrame: Leistungstabelle (kW-bezogen, p_nom)
    """
    years = get_investment_years(n)
    years_set = set(years)
    rows = []

    if hasattr(n, "generators") and not n.generators.empty:
        df = n.generators
        p_nom = nominal_from_static(df).fillna(0.0)
        for name, r in df.iterrows():
            base, year = split_base_and_year(name, years_set)
            s, sc = sector_subcarrier_from_component_row(n, "generators", r)
            rows.append({
                "sector": s, "subcarrier": sc, "component": "generators",
                "name": str(name), "base_name": base, "year": year,
                "port": "p", "p_nom": float(p_nom.get(name, 0.0)),
            })

    if hasattr(n, "storage_units") and not n.storage_units.empty:
        df = n.storage_units
        p_nom = nominal_from_static(df).fillna(0.0)
        for name, r in df.iterrows():
            base, year = split_base_and_year(name, years_set)
            s, sc = sector_subcarrier_from_component_row(n, "storage_units", r)
            rows.append({
                "sector": s, "subcarrier": sc, "component": "storage_units",
                "name": str(name), "base_name": base, "year": year,
                "port": "p", "p_nom": float(p_nom.get(name, 0.0)),
            })

    if hasattr(n, "links") and not n.links.empty:
        df = n.links
        p_nom = nominal_from_static(df).fillna(0.0)
        for name, r in df.iterrows():
            base, year = split_base_and_year(name, years_set)
            p_in = float(p_nom.get(name, 0.0))

            bus0 = r.get("bus0")
            s0, sc0 = sector_subcarrier_from_bus(n, bus0)
            rows.append({
                "sector": s0, "subcarrier": sc0, "component": "links",
                "name": str(name), "base_name": base, "year": year,
                "port": "in", "p_nom": p_in,
            })

            for i in range(1, 10):
                bus_col = f"bus{i}"
                if bus_col not in df.columns:
                    break
                bus_i = r.get(bus_col)
                if pd.isna(bus_i) or bus_i is None or str(bus_i).strip() == "":
                    continue

                eff_col = "efficiency" if i == 1 else f"efficiency{i}"
                eff = r.get(eff_col)
                eff = 1.0 if pd.isna(eff) else float(eff)

                si, sci = sector_subcarrier_from_bus(n, bus_i)
                rows.append({
                    "sector": si, "subcarrier": sci, "component": "links",
                    "name": str(name), "base_name": base, "year": year,
                    "port": f"out{i}", "p_nom": p_in * eff,
                })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_nom"] = out["p_nom"].fillna(0.0)
    out["subcarrier"] = out.get("subcarrier", DEFAULT_SUBCARRIER).fillna(DEFAULT_SUBCARRIER)
    return out


def prepare_multicategory(
    df_caps: pd.DataFrame,
    n: pypsa.Network,
    add_component_prefix: bool = True,
    value_col: str = "p_nom"
):
    """
    Bereitet Kapazitätstabellen für Plotly 'multicategory' x-Achsen auf (year, label), aggregiert
    nach (sector, subcarrier, year, label, component)
    
    Inputs: df_caps: DataFrame aus build_capacity_table oder build_energy_capacity_table 
                     (ggf. bereits erweitert)
                     n: pypsa.Network
                     add_component_prefix: ob label 'component__...' bekommt
                     value_col: zu aggregierende Spalte (p_nom oder e_nom)
                     
    Ermittelt years aus get_investment_years
    Sichert Spalte 'subcarrier' und setzt Defaults
    Baut label aus component/base_name/port
    Wenn years vorhanden: repliziert konstanten Bestand (year NaN) über alle Jahre
    Gruppiert und summiert value_col
    Teilt Ergebnis in dict je Sektor (SECTORS)
    Strippt Suffix "_sector", bei Links wird Subcarrier des entspr. Ports angehängt
    
    Outputs: (dict[str,DataFrame] by_sector, list[int] years)
    """
    years = get_investment_years(n)

    if df_caps is None or df_caps.empty or df_caps.shape[1] == 0:
        empty_cols = ["sector", "subcarrier", "year", "label", "component", value_col]
        result = {s: pd.DataFrame(columns=empty_cols) for s in SECTORS}
        return result, years

    df = df_caps.copy()

    if "subcarrier" not in df.columns:
        df["subcarrier"] = DEFAULT_SUBCARRIER
    df["subcarrier"] = df["subcarrier"].fillna(DEFAULT_SUBCARRIER).astype(str)

    def _label_suffix(value) -> str:
        if value is None:
            return ""
        if isinstance(value, float) and pd.isna(value):
            return ""
        text = str(value).strip()
        return text

    def _build_label(row) -> str:
        base = strip_variable_suffix(row.get("base_name", ""))
        component = str(row.get("component", ""))

        if component == "links":
            sector_suffix = _label_suffix(row.get("sector"))
            if sector_suffix:
                if sector_suffix == "Sonstige":
                    carrier_suffix = _label_suffix(row.get("subcarrier"))
                    core = f"{base}_{carrier_suffix}" if carrier_suffix else f"{base}_{sector_suffix}"
                else:
                    core = f"{base}_{sector_suffix}"
            else:
                port_suffix = _label_suffix(row.get("port"))
                core = f"{base}_{port_suffix}" if port_suffix else base
        else:
            core = base

        if add_component_prefix:
            return f"{component}__{core}"
        return core

    df["label"] = df.apply(_build_label, axis=1)

    if years:
        const = df[df["year"].isna()].copy()
        per = df[df["year"].notna()].copy()
        per["year"] = per["year"].astype(int)

        if not const.empty:
            const = const.drop(columns=["year"]).assign(_k=1)
            yrs = pd.DataFrame({"year": years, "_k": 1})
            const = const.merge(yrs, on="_k").drop(columns=["_k"])

        df2 = pd.concat([per, const], ignore_index=True)
    else:
        df2 = df.copy()
        df2["year"] = ""

    df2 = df2.groupby(["sector", "subcarrier", "year", "label", "component"], as_index=False)[value_col].sum()

    result = {}
    for sector in SECTORS:
        result[sector] = df2[df2["sector"] == sector].copy()
    return result, years


def build_sector_bar(
    df_sector: pd.DataFrame,
    sector: str,
    years,
    value_col: str,
    unit: str,
    title_prefix: str,
    color_map: dict[str, str] | None = None,
) -> go.Figure:
    """
    Erstellt gruppierte Balkendiagramme je Sektor (x: [year, label], y: Leistung/ Kapaz.), 
    farbcodiert nach Subcarrier.
    
    Inputs: df_sector: DataFrame eines Sektors aus prepare_multicategory
            sector: str
            years: list[int] oder []
            value_col: 'p_nom' oder 'e_nom'
            unit: 'kW' oder 'kWh'
            title_prefix: z.B. 'Nennleistungen'
            color_map: optional dict subcarrier->color
            
    Normalisiert subcarrier, baut year_str und ordnet Jahre kategorisch.
    Sortiert und baut Anzeige-Labels (display_name_map) plus Hover-Bereinigung
    Erzeugt pro Subcarrier einen Bar-Trace; x ist multicategory (year_str, label_disp)
    Setzt Layout (barmode group, Achsen, Legende)
    
    Outputs: go.figure (Leistungs- bzw. Kapazitätsdiagramm)
    """
    fig = go.Figure()

    if df_sector is None or df_sector.empty:
        fig.update_layout(title=f"{title_prefix} ({sector}) [{unit}] (keine Daten)")
        return fig

    df = df_sector.copy()
    df["subcarrier"] = df.get("subcarrier", DEFAULT_SUBCARRIER).fillna(DEFAULT_SUBCARRIER).astype(str)

    df["year_str"] = df["year"].astype(str)
    if years:
        years_str = [str(y) for y in years]
        df["year_str"] = pd.Categorical(df["year_str"], categories=years_str, ordered=True)

    df = df.sort_values(["year_str", "subcarrier", "label", "component"])
    name_map = display_name_map(df["label"].astype(str).unique().tolist())
    df["label_disp"] = df["label"].astype(str).map(name_map)
    df["label_disp_hover"] = df["label_disp"].astype(str).apply(strip_port_suffix_for_hover)

    sub_order = sorted(df["subcarrier"].dropna().astype(str).unique().tolist())
    for sc in sub_order:
        dsc = df[df["subcarrier"].astype(str) == sc].copy()
        if dsc.empty:
            continue

        marker = {}
        if color_map is not None:
            col = color_map.get(str(sc))
            if col:
                marker["color"] = col

        fig.add_trace(go.Bar(
            name=str(sc),
            x=[dsc["year_str"].astype(str).tolist(), dsc["label_disp"].astype(str).tolist()],
            y=dsc[value_col].astype(float).tolist(),
            customdata=np.column_stack([
                dsc["year_str"].astype(str).values,
                dsc["label_disp_hover"].astype(str).values,
            ]),
            marker=marker if marker else None,
            hovertemplate=(
                "%{customdata[0]} - %{customdata[1]}<br>"
                "Energieträger: " + str(sc) + "<br>"
                "%{y:.2f} " + unit +
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        barmode="group",
        title=f"{title_prefix} ({sector}) [{unit}]",
        yaxis_title=f"{title_prefix} [{unit}]",
        legend_title="Energieträger",
        margin=dict(l=30, r=30, t=60, b=150),
        showlegend=True
    )
    fig.update_xaxes(type="multicategory", tickangle=45)
    return fig


def build_expansion_path_scatter(
    by_sector: dict,
    sector: str,
    years: list,
    value_col: str = "p_nom",
    unit: str = "kW",
    max_series: int = 25,
    color_map: dict[str, str] | None = None,
) -> go.Figure:
    """
    Visualisiert den Ausbaupfad über Investitionsperioden: pro Asset eine Linie, Legende
    gruppiert nach Subcarrier
    
    Inputs: by_sector: dict[str,DataFrame] (aus prepare_multicategory)
            sector: str
            years: list[int]
            value_col: 'p_nom'
            unit: 'kW'
            max_series: int Top-N Linien nach max(value)
            color_map: optional dict subcarrier->color
            
    Filtert DataFrame nach Sektor, normalisiert Subcarrier
    Wählt Top-N Labels nach max(value_col)
    Erzeugt je label eine Scatter-Linie über inv_period; Legende zeigt Subcarrier-Gruppen
    Nutzt legend.groupclick='togglegroup' zum ein/ausblenden von Gruppen
    
    Outputs: go.figure (Scatter-Plot Ausbaupfade)
    """
    df = by_sector.get(sector)
    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"Ausbaupfad ({sector}) [{unit}] (keine Daten)")
        return fig

    d = df.copy()
    d["subcarrier"] = d.get("subcarrier", DEFAULT_SUBCARRIER).fillna(DEFAULT_SUBCARRIER).astype(str)

    # Top-N Series
    top_labels = (
        d.groupby("label")[value_col]
         .max()
         .sort_values(ascending=False)
         .head(max_series)
         .index
    )
    d = d[d["label"].isin(top_labels)].copy()

    name_map = display_name_map(d["label"].astype(str).unique().tolist())
    d["label_disp"] = d["label"].astype(str).map(name_map)
    d["label_disp_hover"] = d["label_disp"].astype(str).apply(strip_port_suffix_for_hover)

    # X-Achse (Investitionsperioden)
    if years:
        years_str = [str(int(y)) for y in years]
        d["inv_period"] = d["year"].astype(int).astype(str)
        x_order = years_str
        title = f"Ausbaupfad ({sector}) [{unit}]"
    else:
        d["inv_period"] = "Single"
        x_order = ["Single"]
        title = f"Ausbaupfad ({sector}) [{unit}]"

    # Liniendiagramm: eine Linie pro label, gruppiert in der Legende nach subcarrier
    fig = go.Figure()

    shown_in_legend = set()

    # Sortieren nach Perioden
    if years:
        d["_year_sort"] = d["inv_period"].astype(int)
        d = d.sort_values(["_year_sort", "subcarrier", "label", "component"])
    else:
        d = d.sort_values(["subcarrier", "label", "component"])

    for label, g in d.groupby("label", sort=False):
        g2 = g.copy()

        sc = str(g2["subcarrier"].iloc[0])
        label_disp_hover = str(g2["label_disp_hover"].iloc[0])

        # Reihenfolge entlang der X-Achse
        if years:
            g2 = g2.sort_values("_year_sort")

        x = g2["inv_period"].astype(str).tolist()
        y = g2[value_col].astype(float).tolist()

        showlegend = sc not in shown_in_legend
        if showlegend:
            shown_in_legend.add(sc)

        col = (color_map.get(sc) if (color_map is not None) else None)

        # customdata für Hover
        customdata = np.column_stack([
            np.full(len(g2), label_disp_hover),
            np.full(len(g2), sc),
        ])
        # ... innerhalb: for label, g in d.groupby("label", sort=False):

        x = g2["inv_period"].astype(str).tolist()
        y = g2[value_col].astype(float).tolist()

        # nur am letzten Punkt beschriften
        txt = [""] * len(x)
        if len(txt) > 0:
            txt[0] = label_disp_hover   # das ist der "Komponentenname" ohne _p/_e

        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="lines+markers+text",
            text=txt,
            textposition="middle left",
            cliponaxis=False,
            legendgroup=sc,
            showlegend=showlegend,
            name=sc,
            line=dict(color=col) if col else None,
            marker=dict(size=15, color=col) if col else dict(size=15),
            customdata=customdata,
            hovertemplate=(
                "%{x} - %{customdata[0]}<br>"
                "Energieträger: %{customdata[1]}<br>"
                "%{y:.2f} " + unit +
                "<extra></extra>"
            ),
        ))

    fig.update_xaxes(range=[-1, len(x_order)-0.5])

    fig.update_layout(
        title=title,
        legend_title="Energieträger",
        margin=dict(l=30, r=30, t=60, b=50),
        legend=dict(groupclick="togglegroup"),
    )
    fig.update_xaxes(
        title="Investitionsperiode",
        type="category",
        categoryorder="array",
        categoryarray=x_order,
    )
    fig.update_yaxes(title=f"Leistung [{unit}]")

    return fig


# Links zur Anbindung von Stores flaggen, damit sie nicht im Lifetime-Diagramm angezeigt werden

def link_is_store_connection_topology(row: pd.Series, store_buses: set[str], max_i: int = 9) -> bool:
    """
    Erkennt Links, die topologisch an interne Store-Busse angeschlossen sind 
    (um sie aus Lifetime-Plots auszublenden)
    
    Inputs: row: pd.Series (Link-Zeile)
            store_buses: set[str]
            max_i: int (Default 9)
            
    Iteriert über bus0..bus{max_i}
    Wenn irgendein bus{i} in store_buses liegt: True
    
    Outputs: bool
    """
    if not store_buses:
        return False
    for i in range(0, max_i + 1):
        b = row.get(f"bus{i}", None)
        if b is None or pd.isna(b):
            continue
        b = str(b).strip()
        if b and b in store_buses:
            return True
    return False


def build_lifetime_table(n: pypsa.Network) -> pd.DataFrame:
    """
    Erstellt eine Tabelle mit Aktivitätszeiträumen (build_year, end_year) und Lifetime-Flags je
    Komponente; inkl. Ausschlussflag für Store-Anbindungslinks
    
    Inputs: n:pypsa.Network
    
    Ermittelt Investitionshorizont (MIP: min/max(years), sonst aus snapshots)
    Erkennt interne Store-Busse (infer_internal_store_buses)
    Iteriert über generators, stores, storage_units, links, lines (wenn vorhanden)
    Leitet Kapazität aus p_nom/s_nom/e_nom ab und ignoriert sehr kleine Werte (EPS)
    Normalisiert build_year; berechnet end_year aus lifetime (oder setzt display_end)
    Setzt Flags: lifetime_missing, lifetime_infinite und exclude_from_lifetime_plot (bei
    Store-Link)
    Ermittelt sector/subcarrier je Komponente 
    (Links: primärer Output-Bus; Sonderfall "variabel")
    
    Outputs: DataFrame: lifetime-Tabelle mit build_year/end_year und Metadaten
    """
    years = get_investment_years(n)
    years_set = set(years)

    EPS = 1e-6

    # Topologische Store-Bus-Menge (robust gegen Namenskonventionen)
    internal_store_buses = infer_internal_store_buses(n)

    def capacity_from_row(component: str, r: pd.Series) -> tuple[float, str]:
        if component == "stores":
            for col in ("e_nom_opt", "e_nom"):
                if col in r.index and pd.notna(r.get(col)):
                    try:
                        return float(r.get(col)), "kWh"
                    except Exception:
                        pass
            return 0.0, "kWh"

        for col in ("p_nom_opt", "p_nom", "s_nom_opt", "s_nom"):
            if col in r.index and pd.notna(r.get(col)):
                try:
                    return float(r.get(col)), "kW"
                except Exception:
                    pass
        return 0.0, "kW"

    if years:
        horizon_start = int(min(years))
        last_period = int(max(years))
    else:
        try:
            snap_year_min = int(pd.to_datetime(pd.Index(n.snapshots)).min().year)
            snap_year_max = int(pd.to_datetime(pd.Index(n.snapshots)).max().year)
            horizon_start = snap_year_min
            last_period = snap_year_max
        except Exception:
            horizon_start = int(pd.Timestamp.today().year)
            last_period = horizon_start

    def norm_build_year(by):
        if by is None or pd.isna(by):
            return horizon_start
        try:
            by_f = float(by)
        except Exception:
            return horizon_start
        if not np.isfinite(by_f):
            return horizon_start
        y = int(by_f)
        if y < 1900:
            return horizon_start
        return y

    def _iter_component_tables():
        tables = []
        if hasattr(n, "generators") and not n.generators.empty:
            tables.append(("generators", n.generators))
        if hasattr(n, "stores") and not n.stores.empty:
            tables.append(("stores", n.stores))
        if hasattr(n, "storage_units") and not n.storage_units.empty:
            tables.append(("storage_units", n.storage_units))
        if hasattr(n, "links") and not n.links.empty:
            tables.append(("links", n.links))
        if hasattr(n, "lines") and not n.lines.empty:
            tables.append(("lines", n.lines))
        return tables

    finite_lifetimes_active = []
    finite_lifetimes_all = []

    for comp_name, df in _iter_component_tables():
        if "lifetime" not in df.columns:
            continue
        for _, r in df.iterrows():
            cap, _unit = capacity_from_row(comp_name, r)
            if cap <= EPS:
                continue
            lt = r.get("lifetime", None)
            if lt is None or pd.isna(lt):
                continue
            try:
                lt_f = float(lt)
            except Exception:
                continue
            if not np.isfinite(lt_f):
                continue

            start = norm_build_year(r.get("build_year", None))
            end = start + lt_f

            finite_lifetimes_all.append(lt_f)
            if start <= last_period < end:
                finite_lifetimes_active.append(lt_f)

    if finite_lifetimes_active:
        max_life_active = float(max(finite_lifetimes_active))
    elif finite_lifetimes_all:
        max_life_active = float(max(finite_lifetimes_all))
    else:
        max_life_active = 1.0

    display_end = int(last_period + max_life_active)

    rows = []

    def _link_primary_output_bus(row: pd.Series):
        for i in range(1, 10):
            b = row.get(f"bus{i}", None)
            if b is not None and not pd.isna(b) and str(b).strip() != "":
                return b
        b0 = row.get("bus0", None)
        if b0 is not None and not pd.isna(b0) and str(b0).strip() != "":
            return b0
        return None

    def _line_sector_bus(row: pd.Series):
        b0 = row.get("bus0", None)
        if b0 is not None and not pd.isna(b0) and str(b0).strip() != "":
            return b0
        b1 = row.get("bus1", None)
        if b1 is not None and not pd.isna(b1) and str(b1).strip() != "":
            return b1
        return None

    def add_rows(df: pd.DataFrame, component: str):
        if df is None or df.empty:
            return

        has_by = "build_year" in df.columns
        has_lt = "lifetime" in df.columns

        for name, r in df.iterrows():
            cap, cap_unit = capacity_from_row(component, r)
            if cap <= EPS:
                continue

            start = norm_build_year(r.get("build_year", None)) if has_by else horizon_start

            lt = r.get("lifetime", None) if has_lt else None
            lifetime_missing = (lt is None) or (pd.isna(lt)) or (not has_lt)

            lifetime_infinite = False
            lifetime_val = None

            if lifetime_missing:
                end = display_end
            else:
                try:
                    lt_f = float(lt)
                except Exception:
                    lt_f = np.nan

                if not np.isfinite(lt_f):
                    lifetime_infinite = True
                    lifetime_val = np.inf
                    end = display_end
                else:
                    lifetime_val = lt_f
                    end = int(start + lt_f)

            if end < start:
                end = start

            base, _ = split_base_and_year(str(name), years_set)

            if component in ("generators", "stores", "storage_units"):
                sec, sc = sector_subcarrier_from_component_row(n, component, r)
            elif component == "lines":
                bus = _line_sector_bus(r)
                sec, sc = sector_subcarrier_from_bus(n, bus)
            elif component == "links":
                bus_out = _link_primary_output_bus(r)
                sec, sc = sector_subcarrier_from_bus(n, bus_out)
                is_store_link = link_is_store_connection_topology(r, internal_store_buses, max_i=9)
            
                if str(sc).strip().lower() == "variabel":
                    bus0 = r.get("bus0", None)
                    if bus0 is not None and not pd.isna(bus0) and str(bus0).strip() != "" and str(bus0) in n.buses.index:
                        raw_bus0_car = n.buses.at[str(bus0), "carrier"] if "carrier" in n.buses.columns else pd.NA
                        car0, _sub0 = split_carrier_subcarrier(raw_bus0_car)
                        sc = car0 if car0 else DEFAULT_SUBCARRIER
                    else:
                        sc = DEFAULT_SUBCARRIER
            else:
                sec, sc = ("Sonstige", DEFAULT_SUBCARRIER)

            sc = DEFAULT_SUBCARRIER if (sc is None or pd.isna(sc) or str(sc).strip() == "") else str(sc)

            rows.append({
                "sector": sec,
                "subcarrier": sc,
                "component": component,
                "name": str(name),
                "base_name": base,
                "build_year": int(start),
                "end_year": int(end),
                "lifetime": lifetime_val,
                "lifetime_infinite": lifetime_infinite,
                "lifetime_missing": lifetime_missing,
                "capacity": float(cap),
                "capacity_unit": cap_unit,
                "exclude_from_lifetime_plot": bool(is_store_link) if component == "links" else False,
            })

    add_rows(n.generators if hasattr(n, "generators") else None, "generators")
    add_rows(n.stores if hasattr(n, "stores") else None, "stores")
    add_rows(n.storage_units if hasattr(n, "storage_units") else None, "storage_units")
    add_rows(n.lines if hasattr(n, "lines") else None, "lines")
    add_rows(n.links if hasattr(n, "links") else None, "links")

    return pd.DataFrame(rows)


def build_lifetime_timeline_fig(
        df_life: pd.DataFrame, 
        sector: str, 
        color_map: dict[str, str] | None = None
        ) -> go.Figure:
    """
    Erstellt ein Timeline-Diagramm (px.timeline) der Aktivitätszeiträume je Komponente in
    einem Sektor, farbcodiert nach Subcarrier

    Inputs: df_life: DataFrame aus build_lifetime_table
            sector: str
            color_map: optional dict subcarrier->color
            
    Filtert auf Sektor, blendet exclude_from_lifetime_plot aus
    Erzeugt Hover-Felder: comp_name_disp (ohne Prefix/Suffix), lifetime_disp ('unbekannt',
    'durchgehend vorhanden', oder Zahl).                                        
    Konvertiert build_year/end_year in Datumswerte (01-01)
    Erzeugt px.timeline und setzt Hovertemplate
    Konfiguriert Achsen (Jahrestakt) und Layout
    
    Outputs: go.figure (Lifetime-Gantt-Chart)
    """
    
    if df_life is None or df_life.empty:
        return go.Figure().update_layout(title=f"Lebensdauer – {sector} (keine Daten)")

    d = df_life[df_life["sector"] == sector].copy()
    
    # Store-Anbindungslinks im Lifetime-Diagramm ausblenden
    if "exclude_from_lifetime_plot" in d.columns:
        d = d[~((d["component"].astype(str) == "links") & (d["exclude_from_lifetime_plot"].astype(bool)))]
        
    if d.empty:
        return go.Figure().update_layout(title=f"Lebensdauer – {sector} (keine Daten)")

    d["subcarrier"] = d.get("subcarrier", DEFAULT_SUBCARRIER).fillna(DEFAULT_SUBCARRIER).astype(str)
    d = d.sort_values(["build_year", "subcarrier", "component", "name"], ascending=[True, True, True, True])
    

    # --- Schöner Komponentenname (ohne _YYYY etc.), nur für Hover ---
    if "base_name" in d.columns:
        d["comp_name_disp"] = d["base_name"].astype(str).map(strip_prefix).map(strip_variable_suffix)
    else:
        d["comp_name_disp"] = d["name"].astype(str).map(strip_prefix).map(strip_variable_suffix)

    # --- Lebensdauer als String für Hover ---
    # Reihenfolge: missing -> "unbekannt", infinite -> "∞", sonst Zahl
    lt = pd.to_numeric(d.get("lifetime", pd.Series(index=d.index, dtype=float)), errors="coerce")
    d["lifetime_disp"] = np.where(
        d.get("lifetime_missing", False).astype(bool),
        "unbekannt",
        np.where(
            d.get("lifetime_infinite", False).astype(bool),
            "durchgehend vorhanden",
            lt.round(2).astype(str)
        )
    )

    d["start_dt"] = pd.to_datetime(d["build_year"].astype(int).astype(str) + "-01-01", errors="coerce")
    d["end_dt"]   = pd.to_datetime(d["end_year"].astype(int).astype(str) + "-01-01", errors="coerce")
    d = d.dropna(subset=["start_dt", "end_dt"])
    if d.empty:
        return go.Figure().update_layout(title=f"Lebensdauer – {sector} (keine gültigen Jahre)")

    fig = px.timeline(
        d,
        x_start="start_dt",
        x_end="end_dt",
        y="name",
        color="subcarrier",
        color_discrete_map=color_map if color_map is not None else None,
        custom_data=["comp_name_disp", "lifetime_disp", "build_year", "end_year"],  # NEU
        title=f"Lebensdauer / Aktivitätszeitraum – {sector}",
    )

    # --- NEU: Hover exakt wie gewünscht ---
    fig.update_traces(
        hovertemplate=(
            "%{customdata[0]}<br>"
            "Lebensdauer (Jahre): %{customdata[1]}<br>"
            "Baujahr: %{customdata[2]}<br>"
            "Endjahr: %{customdata[3]}"
            "<extra></extra>"
        )
    )

    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickformat="%Y", dtick="M12", title="Jahr")
    fig.update_yaxes(title="Komponente")
    fig.update_layout(margin=dict(l=30, r=30, t=60, b=50), height=650, legend_title="Energieträger")
    return fig

#%% Active-Assets pro Investitionsperiode (Timeseries-Filter)

def active_assets_in_period(df_life: pd.DataFrame, period_value) -> set[tuple[str, str]]:
    """
    Bestimmt die Menge aktiver Assets (component, name) für eine Investitionsperiode anhand
    df_life
    
    Inputs: df_life: DataFrame
            period_value: z.B. '2030'
            
    Konvertiert period_value nach int
    Filtert df_life auf build_year <= p < end_year
    Gibt Set aus (component, name) zurück

    Outputs: set[tuple[str,str]]
    """
    if df_life is None or df_life.empty:
        return set()
    try:
        p = int(period_value)
    except Exception:
        return set()

    d = df_life.copy().dropna(subset=["build_year", "end_year"])
    active = d[(d["build_year"].astype(int) <= p) & (p < d["end_year"].astype(int))]
    return set(zip(active["component"].astype(str), active["name"].astype(str)))


def filter_meta_to_active(
        meta: pd.DataFrame, 
        active_set: set[tuple[str, str]], 
        df_life: pd.DataFrame) -> pd.DataFrame:
    """
    Filtert die Zeitreihen-Metadaten (meta) auf Assets, die in einer Periode aktiv sind;
    Komponenten ohne Lifetime-Info bleiben erhalten.
    
    Inputs: meta: Meta-DataFrame (Index=Spalte)
            active_set: set[(component, asset)]
            df_life: Lifetime-DataFrame
            
    Ermittelt Komponenten, die überhaupt Lifetime-Informationen haben.
    Behält Meta-Zeilen, wenn Komponente keine Lifetime führt oder (comp, asset) 
    in active_set enthalten ist
    
    Outputs: DataFrame: gefilterter Meta-df
    """
    if meta is None or meta.empty or df_life is None or df_life.empty:
        return meta
    comps_with_life = set(df_life["component"].astype(str).unique())

    keep_mask = []
    for _, r in meta.iterrows():
        comp = str(r.get("component", ""))
        asset = str(r.get("asset", ""))
        if comp not in comps_with_life:
            keep_mask.append(True)
            continue
        keep_mask.append((comp, asset) in active_set)
    return meta.loc[keep_mask].copy()


#%% Speicherkapazität (kWh) Tabelle

def energy_nominal_from_store(df_static: pd.DataFrame) -> pd.Series:
    """
    Liest für Stores die Energiespeicherkapazität (e_nom_opt/e_nom) als Series aus
    
    Inputs: df_static: DataFrame (nur Stores werden verarbeitet)
    
    Sucht e_nom_opt, dann e_nom, sonst leere float-Serie
    
    Outputs: pd.Series    
    """
    for col in ("e_nom_opt", "e_nom"):
        if col in df_static.columns:
            return df_static[col]
    return pd.Series(index=df_static.index, dtype=float)


def build_energy_capacity_table(n: pypsa.Network) -> pd.DataFrame:
    """
    Erstellt eine tabellarische Übersicht der Speicherkapazitäten (kWh) für Stores und Storage
    Units (über p_nom*max_hours)
    
    Inputs: N: pypsa.Network
    
    Ermittelt years/years_set
    Stores: e_nom aus energy_nominal_from_store; Name -> (base, year); Sektor/Subcarrier per
    sector_subcarrier_from_component_row
    Storage Units: e_nom = p_nom * max_hours; analoges Mapping
    Erzeugt DataFrame mit Spalten: sector, subcarrier, component, name, base_name, year,
    port="e", e_nom
    
    Outputs: DataFrame: Speicherkapazitäten [kWh]
    """
    cols = ["sector", "subcarrier", "component", "name", "base_name", "year", "port", "e_nom"]
    years = get_investment_years(n)
    years_set = set(years)
    rows = []

    if hasattr(n, "stores") and not n.stores.empty:
        df = n.stores
        e_nom = energy_nominal_from_store(df).fillna(0.0)
        for name, r in df.iterrows():
            base, year = split_base_and_year(name, years_set)
            s, sc = sector_subcarrier_from_component_row(n, "stores", r)
            rows.append({
                "sector": s, "subcarrier": sc, "component": "stores",
                "name": str(name), "base_name": base, "year": year,
                "port": "e", "e_nom": float(e_nom.get(name, 0.0)),
            })

    if hasattr(n, "storage_units") and not n.storage_units.empty:
        df = n.storage_units
        p_nom = nominal_from_static(df).fillna(0.0)
        max_hours = df["max_hours"] if "max_hours" in df.columns else pd.Series(0.0, index=df.index)
        max_hours = max_hours.fillna(0.0).astype(float)
        e_nom = (p_nom.astype(float) * max_hours)

        for name, r in df.iterrows():
            base, year = split_base_and_year(name, years_set)
            s, sc = sector_subcarrier_from_component_row(n, "storage_units", r)
            rows.append({
                "sector": s, "subcarrier": sc, "component": "storage_units",
                "name": str(name), "base_name": base, "year": year,
                "port": "e", "e_nom": float(e_nom.get(name, 0.0)),
            })

    out = pd.DataFrame(rows, columns=cols)
    if out.empty:
        return out
    out["e_nom"] = out["e_nom"].fillna(0.0)
    out["subcarrier"] = out.get("subcarrier", DEFAULT_SUBCARRIER).fillna(DEFAULT_SUBCARRIER)
    return out


#%% Wirtschaftlichkeit (Kosten)

COST_UNIT = "€/ Jahr"
MARGINAL_COST_IS_EUR_PER_MWH = False
COST_COMPONENTS = ["generators", "links", "storage_units", "stores", "lines",]
DEFAULT_DISCOUNT_RATE = 0.0  # r=0 => Overnight = Annuität * Lifetime


def _get_objective_snapshot_weights(n: pypsa.Network) -> pd.Series:
    """
    Liest Snapshot-Gewichtungen für die Zielfunktion ('objective') robust aus
    n.snapshot_weightings
    
    Inputs: n: pypsa.Network
    
    Falls snapshot_weightings fehlt: Series(1.0)
    Falls DataFrame mit Spalte 'objective': nutzt diese Spalte
    Falls Attribut 'objective' vorhanden: nutzt sw.objective
    Sonst Fallback: Series(1.0)
    
    Outputs: pd.Series: Weightings je Snapshot
    """
    sw = getattr(n, "snapshot_weightings", None)
    if sw is None:
        return pd.Series(1.0, index=n.snapshots, name="objective")
    if isinstance(sw, pd.DataFrame) and "objective" in sw.columns:
        return sw["objective"]
    if hasattr(sw, "objective"):
        return sw.objective
    return pd.Series(1.0, index=n.snapshots, name="objective")


def _nominal_opt_series(comp_name: str, df: pd.DataFrame) -> pd.Series:
    """
    Liest die optimierte Nennleistung/ -kapazität je Komponente (p_nom_opt / e_nom_opt etc.) als
    numerische Series

    Inputs: comp_name: str
            df: DataFrame (static)
            
    Stores: e_nom_opt/e_nom
    Sonst: p_nom_opt/p_nom/s_nom_opt/s_nom
    Fallback: 0.0-Series
    
    Outputs: pd.Series
    """
    if comp_name == "stores":
        for col in ("e_nom_opt", "e_nom"):
            if col in df.columns:
                return pd.to_numeric(df[col], errors="coerce")
        return pd.Series(0.0, index=df.index)
    for col in ("p_nom_opt", "p_nom", "s_nom_opt", "s_nom"):
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(0.0, index=df.index)


def _safe_cost_series(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Gibt eine numerische Kostenseries aus df[col] zurück
    Fehlt die Spalte, wird 0.0 zurückgegeben
    
    Inputs: df: DataFrame
            col: str
    
    Outputs: pd.Series
    """
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index)


def _infer_build_year(name: str, df: pd.DataFrame, years: list[int]) -> int | None:
    """
    Leitet ein build_year für ein Asset ab, bevorzugt aus Namenssuffix '_YYYY', sonst aus Spalte
    build_year, sonst aus min(years)
    
    Inputs: name: str
            df: DataFrame.
            years: list[int]
            
    Prüft Suffix via split_base_and_year
    Sonst: versucht df.at[name,'build_year'] zu lesen
    Fallback: min(years) oder None
    
    Outputs: int | None
    """
    years_set = set(years)
    _base, y_suffix = split_base_and_year(name, years_set)
    if y_suffix is not None:
        return int(y_suffix)
    if "build_year" in df.columns:
        by = df.at[name, "build_year"]
        if by is not None and not pd.isna(by):
            try:
                return int(float(by))
            except Exception:
                pass
    if years:
        return int(min(years))
    return None


def _infer_end_year(name: str, df: pd.DataFrame, build_year: int | None) -> float:
    """
    Berechnet end_year aus build_year + lifetime; ohne lifetime oder fehlendem build_year -> inf
    
    Inputs: name: str
            df: DataFrame
            build_year: int|None
            
    Wenn build_year None: inf
    Wenn 'lifetime' fehlt/ungültig: inf
    Sonst: build_year + lifetime
    
    Outputs: float: end_year (kann inf sein)
    """    
    if build_year is None:
        return np.inf
    if "lifetime" not in df.columns:
        return np.inf
    lt = df.at[name, "lifetime"]
    if lt is None or pd.isna(lt):
        return np.inf
    try:
        lt_f = float(lt)
    except Exception:
        return np.inf
    if not np.isfinite(lt_f):
        return np.inf
    return float(build_year) + lt_f


def _map_build_to_investment_period(build_year: int, years: list[int]) -> int:
    """
    Mappt ein build_year auf die nächste (>=) Investitionsperiode in years.
    
    Inputs: build_year: int
            years: list[int]
    
    Wenn build_year exakt in years: identisch
    Sonst: wählt kleinste Periode >= build_year, sonst letzte Periode
    
    Outputs: int: Investitionsperiode
    """
    if not years:
        raise ValueError("years leer, obwohl MIP erwartet wurde.")
    if build_year in years:
        return build_year
    future = [y for y in years if y >= build_year]
    return future[0] if future else years[-1]

def _annuity_factor(r: float, n_years: float) -> float:
    """
    Berechnet den Annuitätsfaktor a(r,n) zur Annualisierung von Overnight-Kosten.
    
    Inputs: r: float (discount rate)
            n_years: float (lifetime)
            
    Validiert Endlichkeit und n_years>0
    Sonderfall r·0: 1/n_years
    Sonst: r / (1 - (1+r)^(-n))
    
    Outputs: float (oder NaN bei ungültigen Eingaben)
    """
    try:
        r = float(r)
        n_years = float(n_years)
    except Exception:
        return np.nan
    if (not np.isfinite(r)) or (not np.isfinite(n_years)) or n_years <= 0:
        return np.nan
    if abs(r) < 1e-12:
        return 1.0 / n_years
    return r / (1.0 - (1.0 + r) ** (-n_years))


def _overnight_from_annualized(annualized: float, r: float, lifetime: float) -> float:
    """
    Rekonstruiert Overnight-Kosten aus annualisierten Kosten via Division durch
    Annuitätsfaktor
    
    Inputs: annualized: float
            r: float
            lifetime: float
    
    Berechnet a = _annuity_factor(r,lifetime)
    Wenn a ungültig: gibt annualized zurück
    Sonst: annualized/a
    
    Outputs: float: overnight
    """
    try:
        annualized = float(annualized)
    except Exception:
        return 0.0
    a = _annuity_factor(r, lifetime)
    if (a is None) or (not np.isfinite(a)) or a <= 0:
        return annualized
    return annualized / a


def _infer_build_year_strict(name: str, df: pd.DataFrame, years: list[int]) -> int | None:
    """
    Strenge Neubau-Erkennung: zählt nur, wenn ein eindeutiges Build-Jahr existiert (Suffix oder
    build_year-Spalte)
    
    Inputs: name: str
            df: DataFrame
            years: list[int]
            
    Prüft Suffix _YYYY in years
    Sonst: liest build_year, wenn vorhanden und konvertierbar
    Sonst None
    
    Outputs: int|None
    """
    years_set = set(years)
    _base, y_suffix = split_base_and_year(str(name), years_set)
    if y_suffix is not None:
        return int(y_suffix)

    if "build_year" in df.columns:
        by = df.at[name, "build_year"]
        if by is not None and not pd.isna(by):
            try:
                return int(float(by))
            except Exception:
                pass
    return None


def build_investment_capex_df(n: pypsa.Network) -> pd.DataFrame:
    """
    Erzeugt eine Tabelle nicht-annuisierter Investitionen je Investitionsperiode (Overnight
    CAPEX) für neu gebaute Assets
    
    Inputs: n: pypsa.Network
    
    Ermittelt Investitionsperioden years
    Iteriert über COST_COMPONENTS und deren statische Tabellen
    Für jedes Asset: nimmt nominelle Kapazität nom_i
    Wenn MIP: nur Neubau-Assets via _infer_build_year_strict; bestimmt build_period
    Bestimmt unit_cost: entweder explizit 'capital_cost_overnight' oder rekonstruiert overnight
    aus annualisiertem capital_cost via discount_rate und lifetime
    Überspringt Assets ohne gültige lifetime (um Fehlinterpretationen zu vermeiden)
    Berechnet investment_capex = unit_cost * nom_i; sammelt Zeilen (period, component,
    name, base_name, investment_capex)
    
    Outputs: DataFrame: Investitions-CAPEX je Asset und Periode
    """
    years = get_investment_years(n)
    years_set = set(years)

    rows = []

    for comp_name in COST_COMPONENTS:
        if not hasattr(n, comp_name):
            continue
        static_df = getattr(n, comp_name)
        if static_df is None or static_df.empty:
            continue

        nom = _nominal_opt_series(comp_name, static_df).fillna(0.0).astype(float)
        cap_cost_annual = _safe_cost_series(static_df, "capital_cost")  # typischerweise €/unit/a

        # falls irgendwann explizite Overnight-Kosten im Datensatz vorliegen
        cap_cost_overnight = None
        if "capital_cost_overnight" in static_df.columns:
            cap_cost_overnight = pd.to_numeric(static_df["capital_cost_overnight"], errors="coerce").fillna(0.0)

        for name in static_df.index:
            nom_i = float(nom.get(name, 0.0))
            if nom_i <= 0.0:
                continue

            if years:
                by = _infer_build_year_strict(str(name), static_df, years)
                if by is None:
                    continue  # kein eindeutig neuer Build -> nicht zählen
                build_period = _map_build_to_investment_period(int(by), years)
                period_label = str(build_period)
            else:
                period_label = "Single"

            if cap_cost_overnight is not None:
                unit_cost = float(cap_cost_overnight.get(name, 0.0))
            else:
                ann = float(cap_cost_annual.get(name, 0.0))

           # Default: r=0, wenn nicht explizit vorhanden
                r = DEFAULT_DISCOUNT_RATE
                if "discount_rate" in static_df.columns:
                    v = static_df.at[name, "discount_rate"]
                    if v is not None and not pd.isna(v):
                        try:
                            r = float(v)
                        except Exception:
                            r = DEFAULT_DISCOUNT_RATE

          # Lifetime ist Pflicht, sonst keine De-Annualisierung möglich
                lt = np.nan
                if "lifetime" in static_df.columns:
                    v = static_df.at[name, "lifetime"]
                    if v is not None and not pd.isna(v):
                       try:
                           lt = float(v)
                       except Exception:
                           lt = np.nan

    # Wenn lifetime fehlt/ungültig: überspringen (sonst würden Annuitäten als Overnight fehlinterpretiert)
                if (not np.isfinite(lt)) or (lt <= 0):
                     continue

                unit_cost = _overnight_from_annualized(ann, r, lt)

            inv = unit_cost * nom_i
            if (not np.isfinite(inv)) or abs(inv) <= 0.0:
                continue

            base_name, _ = split_base_and_year(str(name), years_set)
            rows.append({
                "period": period_label,
                "component": comp_name,
                "name": str(name),
                "base_name": base_name,
                "investment_capex": float(inv),
            })

    return pd.DataFrame(rows)


def _get_dispatch_df(n: pypsa.Network, comp_name: str) -> pd.DataFrame | None:
    """
    Liefert ein geeignetes Dispatch-DataFrame (p) für eine Komponente aus
    n.components.dynamic
    
    Inputs: n:pypsa.Network
            comp_name: str
    
    Für links: bevorzugt p0, sonst kleinster vorhandener p{i}
    Für lines: p0, sonst p1
    Für andere: p, sonst None

    Outputs: DataFrame|None
    """
    if not hasattr(n, "components") or not hasattr(n.components, comp_name):
        return None
    dyn = getattr(n.components, comp_name).dynamic

    if comp_name == "links":
        if "p0" in dyn:
            return dyn.get("p0")
        port_attrs = [a for a in dyn.keys() if re.match(r"^p\d+$", str(a))]
        if port_attrs:
            port_attrs = sorted(port_attrs, key=lambda s: int(str(s)[1:]))
            return dyn.get(port_attrs[0])
        return None

    if comp_name == "lines":
        if "p0" in dyn:
            return dyn.get("p0")
        if "p1" in dyn:
            return dyn.get("p1")
        return None

    if "p" in dyn:
        return dyn.get("p")
    return None

def _variable_opex_by_period(
    n: pypsa.Network,
    comp_name: str,
    static_df: pd.DataFrame,
    years: list[int],
    weights: pd.Series,
) -> pd.DataFrame:
    """
    Berechnet variable OPEX je Periode aus Dispatch * marginal_cost * Snapshot-Weightings
    
    Inputs: n: pypsa.Network
            comp_name: str
            static_df: DataFrame (statisch, enthält marginal_cost)
            years: list[int]
            weights: pd.Series (Snapshot-Weights)
            
    Lädt Dispatch p_df via _get_dispatch_df; falls None: gibt Null-DF zurück
    Passt marginal_cost auf Dispatch-Spalten an; Passt weights auf Index an
    Berechnet Energie = |p| * w
    Optional: Umrechnung kWh->MWh, wenn MARGINAL_COST_IS_EUR_PER_MWH True
    Kostenzeitreihe = Energie * marginal_cost
    MIP: gruppiert nach period-level (level=0); sonst summiert zu 'Single'

    Outputs: DataFrame: index=Perioden, columns=Assets, values=variable OPEX
    """
    p_df = _get_dispatch_df(n, comp_name)
    if p_df is None or p_df.empty:
        idx = [str(y) for y in years] if years else ["Single"]
        return pd.DataFrame(0.0, index=idx, columns=static_df.index)

    mc = _safe_cost_series(static_df, "marginal_cost").reindex(p_df.columns).fillna(0.0)
    w = weights.reindex(p_df.index).fillna(0.0)

    energy = p_df.abs().mul(w, axis=0)
    if MARGINAL_COST_IS_EUR_PER_MWH:
        energy = energy / 1000.0

    cost_ts = energy.mul(mc, axis=1)

    if isinstance(cost_ts.index, pd.MultiIndex):
        out = cost_ts.groupby(level=0).sum()
        out.index = out.index.astype(str)
        return out

    out = pd.DataFrame(cost_ts.sum(), columns=["Single"]).T
    out.index = ["Single"]
    return out

def build_investment_capex_totals_fig(df_inv: pd.DataFrame, years: list[int]) -> go.Figure:
    """
    Erstellt ein Säulendiagramm der Gesamtinvestitionen (Overnight CAPEX) je Periode;
    Single-year als einzelner Balken 'CAPEX (Summe)'
    
    Inputs: df_inv: DataFrame aus build_investment_capex_df
            years: list[int]
            
    Aggregiert df_inv nach 'period' und summiert investment_capex
    Bei MIP: reindex nach years-Reihenfolge
    Bei Single: ersetzt 'Single' durch 'CAPEX (Summe)'
    Erstellt go.Bar mit Hovertemplate und Layout
    
    Outputs: go.figure (Säulendiagramm Gesamtinvestitionen)
    """
    
    fig = go.Figure()
    if df_inv is None or df_inv.empty:
        fig.update_layout(title="Investitionen (Overnight CAPEX) (keine Daten)")
        return fig

    if years:
        order = [str(y) for y in years]
        agg = df_inv.groupby("period")["investment_capex"].sum().reindex(order).fillna(0.0)
        x = agg.index.tolist()
        y = agg.values.tolist()
        title = "Gesamtinvestitionen: Zubau je Investitionsperiode (nicht annuisiert) [€]"
        x_title = "Investitionsperiode"
    else:
        # Single-year: "Single" -> "CAPEX (Summe)"
        agg = df_inv.groupby("period")["investment_capex"].sum().fillna(0.0)

        if len(agg.index) == 1 and str(agg.index[0]) == "Single":
            x = ["CAPEX (Summe)"]
            y = [float(agg.iloc[0])]
        else:
            x = [("CAPEX (Summe)" if str(p) == "Single" else str(p)) for p in agg.index.tolist()]
            y = agg.values.tolist()

        title = "Gesamtinvestitionen (CAPEX (Summe), nicht annuisiert) [€]"
        x_title = "Kostenart"

    fig.add_trace(go.Bar(
        x=x,
        y=y,
        hovertemplate="%{x}<br>%{y:.2f} €<extra></extra>",
        name="Investitionen",
        marker=dict(color=COST_COLOR_MAP.get("CAPEX")),
    ))

    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title="Investitionen [€]",
        margin=dict(l=30, r=30, t=60, b=50),
        showlegend=True,
        legend_title="Kostenart",
    )
    return fig



def build_costs_df(n: pypsa.Network) -> pd.DataFrame:
    """
    Berechnet annualisierte CAPEX und OPEX (fix+variabel) je Komponente und Periode (nur für
    aktive Assets)
    
    Inputs: n: pypsa.Network
    
    Liest years und objective Snapshot-Weights
    Iteriert über COST_COMPONENTS und Assets mit positiver Nennleistung/ Kapazität
    Leitet build_year und end_year ab; bei MIP: berücksichtigt nur aktive Perioden
    CAPEX = capital_cost (annualisiert) * nom_i in jeder aktiven Periode
    Fix-OPEX = fixed_cost * nom_i; variabel via _variable_opex_by_period
    Sammelt Zeilen (period, component, name, base_name, label, capex, opex, opex_fix, opex_var)
    
    Outputs: Dataframe: Kosten je Asset und Peruide
    """
    years = get_investment_years(n)
    weights = _get_objective_snapshot_weights(n)

    rows = []
    years_set = set(years)

    for comp_name in COST_COMPONENTS:
        if not hasattr(n, comp_name):
            continue
        static_df = getattr(n, comp_name)
        if static_df is None or static_df.empty:
            continue
        nom = _nominal_opt_series(comp_name, static_df).fillna(0.0).astype(float)
        cap_cost = _safe_cost_series(static_df, "capital_cost")
        fix_cost = _safe_cost_series(static_df, "fixed_cost")
        var_opex = _variable_opex_by_period(n, comp_name, static_df, years, weights)

        for name in static_df.index:
            nom_i = float(nom.get(name, 0.0))
            if nom_i <= 0.0:
                continue

            base_name, _y_suffix = split_base_and_year(str(name), years_set)
            build_year = _infer_build_year(str(name), static_df, years)
            end_year = _infer_end_year(str(name), static_df, build_year)

            periods_iter = [str(y) for y in years] if years else ["Single"]

            for p in periods_iter:
                if years and build_year is not None:
                    p_int = int(p)
                    active = (p_int >= int(build_year)) and (p_int < end_year)
                    if not active:
                        continue

                capex = 0.0
                # CAPEX als Annuität: in jeder aktiven Periode ansetzen
                capex = float(cap_cost.get(name, 0.0)) * nom_i


                opex_fix = float(fix_cost.get(name, 0.0)) * nom_i
                opex_var = float(var_opex.at[p, name]) if (p in var_opex.index and name in var_opex.columns) else 0.0
                opex = opex_fix + opex_var

                if capex == 0.0 and opex == 0.0:
                    continue

                rows.append({
                    "period": str(p),
                    "component": comp_name,
                    "name": str(name),
                    "base_name": base_name,
                    "label": f"{comp_name}__{base_name}",
                    "capex": capex,
                    "opex": opex,
                    "opex_fix": opex_fix,
                    "opex_var": opex_var,
                })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out[(out["capex"].abs() + out["opex"].abs()) > 0.0].copy()
    return out


def build_cost_totals_fig(df_cost: pd.DataFrame, years: list[int]) -> go.Figure:
    """
    Erstellt ein gestapeltes Balkendiagramm CAPEX/OPEX je Investitionsperiode (MIP)
    
    Inputs: df_cost: DataFrame aus build_costs_df
            years: list[int]
            
    Aggregiert df_cost nach period und summiert capex/opex
    Reindex nach years-Reihenfolge (falls vorhanden)
    Erstellt zwei Bar-Traces (CAPEX, OPEX) in Stacked-Ansicht
    
    Outputs: go.figure (Kostendiagramm)        
    """
    fig = go.Figure()
    if df_cost is None or df_cost.empty:
        fig.update_layout(title="Kosten je Investitionsperiode (keine Daten)")
        return fig

    if years:
        order = [str(y) for y in years]
        agg = (df_cost.groupby("period")[["capex", "opex"]].sum().reindex(order).fillna(0.0))
        x = agg.index.tolist()
    else:
        agg = df_cost.groupby("period")[["capex", "opex"]].sum()
        x = agg.index.tolist()

    capex_col = COST_COLOR_MAP.get("CAPEX")
    opex_col  = COST_COLOR_MAP.get("OPEX")

    fig.add_trace(go.Bar(
        name="CAPEX",
        x=x,
        y=agg["capex"].values,
        marker=dict(color=capex_col) if capex_col else None,
        hovertemplate="%{x}<br>%{y:.2f} €<extra></extra>"
    ))
    fig.add_trace(go.Bar(
        name="OPEX",
        x=x,
        y=agg["opex"].values,
        marker=dict(color=opex_col) if opex_col else None,
        hovertemplate="%{x}<br>%{y:.2f} €<extra></extra>"
    ))

    fig.update_layout(
        title=f"Kosten je Investitionsperiode (CAPEX als Annuität vs. OPEX) [{COST_UNIT}]",
        barmode="stack",
        xaxis_title="Investitionsperiode",
        yaxis_title=f"Kosten [{COST_UNIT}]",
        margin=dict(l=30, r=30, t=60, b=50),
        legend_title="Kostenart",
    )
    return fig


def build_cost_totals_singleyear_fig(df_cost: pd.DataFrame, period_label: str = "Single") -> go.Figure:
    """
    Erstellt ein Balkendiagramm für Single-year: zwei Balken (CAPEX, OPEX) statt x='Single'
    
    Inputs: df_cost: DataFrame
            period_label: str (Default 'Single')
    
    Filtert auf period_label, summiert capex und opex
    Erstellt zwei Balken mit x=['CAPEX'] und x=['OPEX']
    
    Outputs: go.figure (Kostendiagramm Single-Year)
    """
    fig = go.Figure()
    if df_cost is None or df_cost.empty:
        fig.update_layout(title="Kosten (keine Daten)")
        return fig

    d = df_cost[df_cost["period"].astype(str) == str(period_label)].copy()
    capex = float(d["capex"].sum()) if "capex" in d.columns else 0.0
    opex  = float(d["opex"].sum())  if "opex"  in d.columns else 0.0

    capex_col = COST_COLOR_MAP.get("CAPEX")
    opex_col  = COST_COLOR_MAP.get("OPEX")

    # x-Achse als Kostenarten statt "Single"
    fig.add_trace(go.Bar(
        name="CAPEX",
        x=["CAPEX"],
        y=[capex],
        marker=dict(color=capex_col) if capex_col else None,
        hovertemplate="CAPEX<br>%{y:.2f} €<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="OPEX",
        x=["OPEX"],
        y=[opex],
        marker=dict(color=opex_col) if opex_col else None,
        hovertemplate="OPEX<br>%{y:.2f} €<extra></extra>",
    ))

    fig.update_layout(
        title=f"Kosten gesamt (Einjahresanalyse: CAPEX als Annuität vs. OPEX) [{COST_UNIT}]",
        barmode="group",
        xaxis_title="Kostenart",
        yaxis_title=f"Kosten [{COST_UNIT}]",
        margin=dict(l=30, r=30, t=60, b=50),
        legend_title="Kostenart",
    )
    return fig



def build_cost_single_fig(df_cost: pd.DataFrame, max_components: int | None = 30) -> go.Figure:
    """
    Erstellt ein gestapeltes Balkendiagramm der Kosten je Komponente (Label) für eine einzelne
    Periode
    
    Inputs: df_cost: DataFrame (typisch bereits auf eine Periode gefiltert)
            max_components: int|None
            
    Gruppiert nach label, summiert capex und opex; berechnet total und sortiert absteigend
    Optional: begrenzt auf Top-N
    Mappt Labels auf Anzeige-Namen und erzeugt Bar-Traces (CAPEX/OPEX)
    
    Outputs: go.figure (Stacked Kostendiagramm Single-Year)
    """
    fig = go.Figure()
    if df_cost is None or df_cost.empty:
        fig.update_layout(title="Kosten nach Komponenten (keine Daten)")
        return fig

    g = (df_cost.groupby("label")[["capex", "opex"]].sum())
    g["total"] = g["capex"] + g["opex"]
    g = g.sort_values("total", ascending=False)
    if max_components is not None and len(g) > max_components:
        g = g.head(max_components)

    labels = g.index.tolist()
    name_map = display_name_map(labels)
    x = [name_map.get(l, l) for l in labels]

    capex_col = COST_COLOR_MAP.get("CAPEX")
    opex_col  = COST_COLOR_MAP.get("OPEX")

    fig.add_trace(go.Bar(name="CAPEX", x=x, y=g["capex"].values,
                         marker=dict(color=capex_col) if capex_col else None))
    fig.add_trace(go.Bar(name="OPEX",  x=x, y=g["opex"].values,
                         marker=dict(color=opex_col) if opex_col else None))

    fig.update_layout(
        title=f"Kosten nach Komponenten [{COST_UNIT}], negative Kosten sind Erlöse",
        barmode="stack",
        xaxis_title="Komponente",
        yaxis_title=f"Kosten [{COST_UNIT}]",
        margin=dict(l=30, r=30, t=60, b=120),
        legend_title="Kostenart",
    )
    fig.update_xaxes(tickangle=45)
    return fig


def build_cost_composition_fig(
    df_cost: pd.DataFrame,
    base_period: str,
    selected_period: str,
    max_components: int | None = 30,
) -> go.Figure:
    """
    Vergleicht Kostenverteilung zwischen Base-Periode und Vergleichsperiode (MIP) als
    gruppierte/relative Balken mit Muster für Base
    
    Inputs: df_cost: DataFrame
            base_period: str
            selected_period: str
            max_components: int|None
            
    Filtert df_cost nach beiden Perioden
    Aggregiert nach (period,label) und berechnet Summe
    Wählt Top-N Komponenten anhand selected_period (Fallback: global)
    Erstellt vier Bar-Traces: Base CAPEX/OPEX (Pattern), Selected CAPEX/OPEX (ohne Pattern)
    Konfiguriert Legenden-Gruppierung und Achsen
    
    Outputs: go.figure (Kostenzusammensetzung nach Komponenten - Jahresvergleich)
    """
    fig = go.Figure()
    if df_cost is None or df_cost.empty:
        fig.update_layout(title="Kostenverteilung (keine Daten)")
        return fig

    d = df_cost[df_cost["period"].isin([base_period, selected_period])].copy()
    if d.empty:
        fig.update_layout(title="Kostenverteilung (keine Daten)")
        return fig

    g = (d.groupby(["period", "label"])[["capex", "opex"]].sum().reset_index())
    g["total"] = g["capex"] + g["opex"]

    sel = g[g["period"] == selected_period].set_index("label")["total"]
    if sel.empty:
        sel = g.set_index("label")["total"]
    sel = sel.sort_values(ascending=False)
    if max_components is not None and len(sel) > max_components:
        keep = set(sel.head(max_components).index)
        g = g[g["label"].isin(keep)].copy()

    labels = [l for l in sel.index.tolist() if l in set(g["label"])]
    name_map = display_name_map(labels)
    x = [name_map.get(l, l) for l in labels]

    def _vals(period: str, col: str):
        s = (g[g["period"] == period].set_index("label")[col])
        return [float(s.get(l, 0.0)) for l in labels]

    capex_col = COST_COLOR_MAP.get("CAPEX")
    opex_col  = COST_COLOR_MAP.get("OPEX")

    fig.add_trace(go.Bar(
        x=x, y=_vals(base_period, "capex"),
        name="CAPEX", legendgroup="CAPEX", showlegend=True,
        offsetgroup="Base",
        marker=dict(color=capex_col, opacity=0.55, pattern=dict(shape="/")) if capex_col else dict(opacity=0.55, pattern=dict(shape="/")),
        hovertemplate=f"Base ({base_period})<br>%{{x}}<br>%{{y:.2f}} {COST_UNIT}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=x, y=_vals(base_period, "opex"),
        name="OPEX", legendgroup="OPEX", showlegend=True,
        offsetgroup="Base",
        marker=dict(color=opex_col, opacity=0.55, pattern=dict(shape="/")) if opex_col else dict(opacity=0.55, pattern=dict(shape="/")),
        hovertemplate=f"Base ({base_period})<br>%{{x}}<br>%{{y:.2f}} {COST_UNIT}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=x, y=_vals(selected_period, "capex"),
        name="CAPEX", legendgroup="CAPEX", showlegend=False,
        offsetgroup="Selected",
        marker=dict(color=capex_col, opacity=1.0) if capex_col else dict(opacity=1.0),
        hovertemplate=f"Vergleich ({selected_period})<br>%{{x}}<br>%{{y:.2f}} {COST_UNIT}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=x, y=_vals(selected_period, "opex"),
        name="OPEX", legendgroup="OPEX", showlegend=False,
        offsetgroup="Selected",
        marker=dict(color=opex_col, opacity=1.0) if opex_col else dict(opacity=1.0),
        hovertemplate=f"Vergleich ({selected_period})<br>%{{x}}<br>%{{y:.2f}} {COST_UNIT}<extra></extra>",
    ))

    fig.update_layout(
        title=f"Kostenverteilung: Base ({base_period}) vs. Vergleich ({selected_period}) [{COST_UNIT}]",
        barmode="relative",
        xaxis_title="Komponente",
        yaxis_title=f"Kosten [{COST_UNIT}]",
        margin=dict(l=30, r=30, t=60, b=140),
        legend_title="Kostenart",
        legend=dict(groupclick="togglegroup"),
    )
    fig.update_xaxes(tickangle=45)
    return fig


#%% Variantenvergleich / Sensitivität – Helper (Kosten + Kapazitäten)

def _basename(nc_path: str) -> str:
    """
    Erzeugt einen robusten, kurzen Varianten-Namen aus einem Dateipfad (os.path.basename)
    
    Inputs: nc_path: str    
    
    Versucht basename, fallback '(keine Auswahl)'
    
    Outputs: str
    """
    try:
        return os.path.basename(str(nc_path)) if nc_path else "(keine Auswahl)"
    except Exception:
        return "(keine Auswahl)"


def _cost_totals_for_period(df_cost: pd.DataFrame, period: str) -> tuple[float, float]:
    """
    Aggregiert (CAPEX, OPEX) aus df_cost für eine bestimmte Periode.
    
    Inputs: df_cost: DataFrame
            period: str
            
    Filtert df_cost nach Periode
    Summiert capex und opex (Fallback 0.0)
    
    Outputs: tuple[float,float]: (capex, opex)
    """
    if df_cost is None or df_cost.empty:
        return 0.0, 0.0
    d = df_cost[df_cost["period"].astype(str) == str(period)].copy()
    if d.empty:
        return 0.0, 0.0
    capex = float(d["capex"].sum()) if "capex" in d.columns else 0.0
    opex  = float(d["opex"].sum())  if "opex"  in d.columns else 0.0
    return capex, opex

def _multicat_series_for_period(
    st: dict,
    period_value: str | None,
    by_key: str,
    value_col: str,
    component_allow: set[str] | None = None,
) -> pd.Series:
    """
    Erzeugt eine Series (index=label) für eine bestimmte Periode aus den vorbereiteten
    multicategory-Tabellen im State

    Inputs: st: dict (Dataset-State)
            period_value: str|None
            by_key: Key im State ('by_sector_p' oder 'by_sector_e')
            value_col: 'p_nom' oder 'e_nom'
            component_allow: optional set[str] um Komponenten einzuschränken
    
    Extrahiert alle Sektor-DFs unter st[by_key] und verkettet sie
    Filtert nach Periode (MIP) oder lässt alles wie gehabt (Single-year)
    Optional: filtert auf erlaubte Komponenten
    Gruppiert nach label und summiert value_col
    
    Outputs: pd.Series
    """
    if st is None or (not st.get("ok", False)):
        return pd.Series(dtype=float)

    by_sector = st.get(by_key, {})
    if not isinstance(by_sector, dict) or not by_sector:
        return pd.Series(dtype=float)

    frames = []
    for _sec, df in by_sector.items():
        if isinstance(df, pd.DataFrame) and (not df.empty):
            frames.append(df.copy())

    if not frames:
        return pd.Series(dtype=float)

    d = pd.concat(frames, ignore_index=True)

    if "label" not in d.columns or value_col not in d.columns:
        return pd.Series(dtype=float)

    # Periodenfilter
    years = st.get("years", [])
    if years:
        try:
            p = int(period_value)
        except Exception:
            return pd.Series(dtype=float)
        d = d[pd.to_numeric(d["year"], errors="coerce").fillna(-1).astype(int) == p]
    else:
        # Single-year: keine Filterung
        pass

    if d.empty:
        return pd.Series(dtype=float)

    # Komponentenfilter (z.B. nur Speicher)
    if component_allow is not None:
        if "component" not in d.columns:
            return pd.Series(dtype=float)
        allow = {str(x) for x in component_allow}
        d = d[d["component"].astype(str).isin(allow)]
        if d.empty:
            return pd.Series(dtype=float)

    s = d.groupby("label")[value_col].sum().astype(float)
    s = s.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return s


def _capacity_series_for_period(st: dict, period_value: str | None) -> pd.Series:
    """
    liefert aktive Nennleistungen (p_nom) je Label für eine Periode aus
    st['by_sector_p']
    
    Inputs: st: dict
            period_value: str|None
            
    Verkettung aller Sektor-DFs aus by_sector_p
    Filter nach year == period (MIP) oder keine Filterung (Single)
    Gruppierung nach label und Summation von p_nom
    
    Outputs: pd.Series
    """
    if st is None or (not st.get("ok", False)):
        return pd.Series(dtype=float)

    by_sector_p = st.get("by_sector_p", {})
    if not isinstance(by_sector_p, dict) or not by_sector_p:
        return pd.Series(dtype=float)

    frames = []
    for _sec, df in by_sector_p.items():
        if isinstance(df, pd.DataFrame) and (not df.empty):
            frames.append(df.copy())

    if not frames:
        return pd.Series(dtype=float)

    d = pd.concat(frames, ignore_index=True)
    if d.empty or "p_nom" not in d.columns or "label" not in d.columns:
        return pd.Series(dtype=float)

    years = st.get("years", [])
    if years:
        # MIP: year ist numerisch, period_value kommt als String
        try:
            p = int(period_value)
        except Exception:
            return pd.Series(dtype=float)
        d = d[pd.to_numeric(d["year"], errors="coerce").fillna(-1).astype(int) == p]
    else:
        # Single-year: year ist "" (leerer String) in prepare_multicategory
        # Keine Filterung nötig
        pass

    if d.empty:
        return pd.Series(dtype=float)

    s = d.groupby("label")["p_nom"].sum().astype(float)
    s = s.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return s


def build_variant_cost_compare_fig(
    st_base: dict,
    st_cmp: dict,
    base_name: str,
    cmp_name: str,
    period_cost: str,
) -> go.Figure:
    """
    Stacked Vergleich: CAPEX/OPEX für zwei Varianten in einer Grafik
    CAPEX ist hier bewusst die annualisierte CAPEX (wie in df_cost), OPEX inkl. fix+variabel
    
    Inputs: st_base: dict
            st_cmp: dict
            base_name: str
            cmp_name: str
            period_cost: str
            
    Extrahiert df_cost für beide States
    Aggregiert CAPEX/OPEX via _cost_totals_for_period
    Erstellt zwei Bar-Traces (CAPEX/OPEX) für x=[base_name, cmp_name] im barmode='stack'
    
    Outputs: go.figure (Kostenvergleich zweier Varianten, Stacked)
    """
    fig = go.Figure()

    df_cost_a = st_base.get("df_cost", pd.DataFrame()) if st_base else pd.DataFrame()
    df_cost_b = st_cmp.get("df_cost", pd.DataFrame()) if st_cmp else pd.DataFrame()

    capex_a, opex_a = _cost_totals_for_period(df_cost_a, period_cost)
    capex_b, opex_b = _cost_totals_for_period(df_cost_b, period_cost)

    x = [base_name, cmp_name]

    capex_col = COST_COLOR_MAP.get("CAPEX")
    opex_col  = COST_COLOR_MAP.get("OPEX")

    fig.add_trace(go.Bar(
        name="CAPEX",
        x=x,
        y=[capex_a, capex_b],
        marker=dict(color=capex_col) if capex_col else None,
        hovertemplate="%{x}<br>CAPEX: %{y:.2f} €<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="OPEX",
        x=x,
        y=[opex_a, opex_b],
        marker=dict(color=opex_col) if opex_col else None,
        hovertemplate="%{x}<br>OPEX: %{y:.2f} €<extra></extra>",
    ))

    title = f"Gesamtkostenvergleich (CAPEX als Annuität + OPEX) – {period_cost} [{COST_UNIT}]"
    fig.update_layout(
        title=title,
        barmode="stack",
        xaxis_title="Variante",
        yaxis_title=f"Kosten [{COST_UNIT}]",
        margin=dict(l=30, r=30, t=60, b=50),
        legend_title="Kostenart",
    )
    return fig


def build_variant_capacity_compare_fig(
    st_base: dict,
    st_cmp: dict,
    base_name: str,
    cmp_name: str,
    period_value: str | None,
    top_n: int = 30,
) -> go.Figure:
    """
    Vergleich der aktiven Nennleeistungen je Komponente/Label (kW) 
    für ein gewähltes Jahr. Darstellung: gruppierte Balken (Variante A vs. Variante B) 
    für Top-N Labels.
    
    Inputs: st_base: dict
            st_cmp: dict
            base_name: str
            cmp_name: str
            period_value: str|None
            top_n: int
            
    Erzeugt Kapazitäts-Serien für beide Varianten via _capacity_series_for_period
    Vereint Indices, baut Vergleichs-DF und sortiert nach Maxwert
    Begrenzt auf Top-N und mappt Labels auf Anzeige-Namen
    Erstellt gruppierte Balken (A vs B) mit Hover
    
    Outputs: go.figure (Variantenvergleich Leistungen)
    """
    fig = go.Figure()

    s_a = _capacity_series_for_period(st_base, period_value)
    s_b = _capacity_series_for_period(st_cmp, period_value)

    if s_a.empty and s_b.empty:
        fig.update_layout(title="Nennleistungen (keine Daten)")
        return fig

    idx = sorted(set(s_a.index.tolist()) | set(s_b.index.tolist()))
    df = pd.DataFrame({
        base_name: s_a.reindex(idx).fillna(0.0).astype(float),
        cmp_name:  s_b.reindex(idx).fillna(0.0).astype(float),
    }, index=idx)

    df["max"] = df.max(axis=1)
    df = df.sort_values("max", ascending=False)

    if top_n is not None and len(df) > top_n:
        df = df.head(top_n)

    labels = df.index.tolist()
    name_map = display_name_map(labels)
    x = [name_map.get(l, l) for l in labels]

    # Variantenfarben (bewusst NICHT die Kostenfarben)
    vivid = px.colors.qualitative.Vivid
    col_a = vivid[2] if len(vivid) > 2 else None
    col_b = vivid[3] if len(vivid) > 3 else None

    fig.add_trace(go.Bar(
        name=base_name,
        x=x,
        y=df[base_name].values,
        marker=dict(color=col_a) if col_a else None,
        hovertemplate="%{x}<br>" + base_name + ": %{y:.2f} kW<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name=cmp_name,
        x=x,
        y=df[cmp_name].values,
        marker=dict(color=col_b) if col_b else None,
        hovertemplate="%{x}<br>" + cmp_name + ": %{y:.2f} kW<extra></extra>",
    ))

    period_txt = str(period_value) if period_value not in (None, "", "Single") else "Single"
    fig.update_layout(
        title=f"Nennleistungen – {period_txt} [kW]",
        barmode="group",
        xaxis_title="Komponente",
        yaxis_title="Leistung [kW]",
        margin=dict(l=30, r=30, t=60, b=140),
        legend_title="Variante",
    )
    fig.update_xaxes(tickangle=45)
    return fig

def build_variant_storage_capacity_compare_fig(
    st_base: dict,
    st_cmp: dict,
    base_name: str,
    cmp_name: str,
    period_value: str | None,
    top_n: int = 30,
) -> go.Figure:
    """
    Vergleich der aktiven Speicherkapazitäten (kWh) für Stores + Storage Units.
    Darstellung analog zum Leistungsvergleich: gruppierte Balken, Top-N Labels.
    
    Inputs: st_base: dict
            st_cmp: dict
            base_name: str
            cmp_name: str
            period_value: str|None
            top_n: int
    
    Zieht Series via _multicat_series_for_period für by_sector_e und filtert auf
    {'stores','storage_units'}
    Vereint, sortiert nach Max, Top-N und display_name_map
    Erstellt gruppierte Balken mit Hover
    
    Outputs: go.figure (Variantenvergleich Speicherkapazität, Säulendiagramm)
    """
    fig = go.Figure()

    s_a = _multicat_series_for_period(
        st=st_base,
        period_value=period_value,
        by_key="by_sector_e",
        value_col="e_nom",
        component_allow={"stores", "storage_units"},
    )
    s_b = _multicat_series_for_period(
        st=st_cmp,
        period_value=period_value,
        by_key="by_sector_e",
        value_col="e_nom",
        component_allow={"stores", "storage_units"},
    )

    if s_a.empty and s_b.empty:
        fig.update_layout(title="Speicherkapazität (keine Daten)")
        return fig

    idx = sorted(set(s_a.index.tolist()) | set(s_b.index.tolist()))
    df = pd.DataFrame({
        base_name: s_a.reindex(idx).fillna(0.0).astype(float),
        cmp_name:  s_b.reindex(idx).fillna(0.0).astype(float),
    }, index=idx)

    df["max"] = df.max(axis=1)
    df = df.sort_values("max", ascending=False)

    if top_n is not None and len(df) > top_n:
        df = df.head(top_n)

    labels = df.index.tolist()
    name_map = display_name_map(labels)
    x = [name_map.get(l, l) for l in labels]

    # gleiche Variantenfarben wie beim Leistungs-Vergleich
    vivid = px.colors.qualitative.Vivid
    col_a = vivid[2] if len(vivid) > 2 else None
    col_b = vivid[3] if len(vivid) > 3 else None

    fig.add_trace(go.Bar(
        name=base_name,
        x=x,
        y=df[base_name].values,
        marker=dict(color=col_a) if col_a else None,
        hovertemplate="%{x}<br>" + base_name + ": %{y:.2f} kWh<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name=cmp_name,
        x=x,
        y=df[cmp_name].values,
        marker=dict(color=col_b) if col_b else None,
        hovertemplate="%{x}<br>" + cmp_name + ": %{y:.2f} kWh<extra></extra>",
    ))

    period_txt = str(period_value) if period_value not in (None, "", "Single") else "Single"
    fig.update_layout(
        title=f"Speicherkapazität – {period_txt} [kWh]",
        barmode="group",
        xaxis_title="Komponente",
        yaxis_title="Energie [kWh]",
        margin=dict(l=30, r=30, t=60, b=140),
        legend_title="Variante",
    )
    fig.update_xaxes(tickangle=45)
    return fig


#%% Sankey

def _get_energy_weights(n: pypsa.Network) -> pd.Series:
    """
    Liest Snapshot-Gewichtungen zur Energieintegration (generators oder objective) robust aus
    n.snapshot_weightings
    
    Inputs: n: pypsa.Network
    
    Wenn sw DataFrame: bevorzugt 'generators', sonst 'objective', sonst 1.0
    Wenn sw Objekt: bevorzugt Attribute generators oder objective
    Fallback: 1.0
    
    Outputs: pd.Series: Gewichtungen je Snapshot
    """
    sw = getattr(n, "snapshot_weightings", None)
    if sw is None:
        return pd.Series(1.0, index=n.snapshots, name="w")

    if isinstance(sw, pd.DataFrame):
        for col in ("generators", "objective"):
            if col in sw.columns:
                return pd.to_numeric(sw[col], errors="coerce").fillna(0.0)
        return pd.Series(1.0, index=n.snapshots, name="w")

    if hasattr(sw, "generators"):
        return pd.to_numeric(sw.generators, errors="coerce").fillna(0.0)
    if hasattr(sw, "objective"):
        return pd.to_numeric(sw.objective, errors="coerce").fillna(0.0)

    return pd.Series(1.0, index=n.snapshots, name="w")


def _filter_snapshots_by_period(n: pypsa.Network, period_value) -> pd.Index:
    """
    Filtert n.snapshots nach einer Investitionsperiode, kompatibel mit MultiIndex, 
    Tuple-Index oder DatetimeIndex
    
    Inputs: n: pypsa.Network
            period_value: beliebig (z.B. '2030', 'Single')
            
    Wenn period_value None/'Single'/'': gibt alle Snapshots zurück
    Wenn MultiIndex: filtert Level 0 auf period_value
    Wenn Tuple-Index: filtert Tuple[0] auf period_value
    Fallback: interpretiert Snapshots als Datetime und filtert nach Jahr
    
    Outputs: pd.Index: gefilterte Snapshots
    """
    snaps = pd.Index(n.snapshots)

    # "Single" / None -> alles
    if period_value is None or str(period_value) in ("Single", "", "Nur ein Zeitraum vorhanden"):
        return snaps

    # Standard-MIP: MultiIndex (period, snapshot)
    if isinstance(snaps, pd.MultiIndex):
        try:
            p = str(int(period_value))
        except Exception:
            p = str(period_value)
        lvl0 = snaps.get_level_values(0).astype(str)
        sel = snaps[lvl0 == p]
        return sel if len(sel) > 0 else snaps

    # Tuple-Index (period, snapshot) -> ohne to_datetime filtern
    if len(snaps) > 0 and isinstance(snaps[0], tuple) and len(snaps[0]) >= 2:
        try:
            p = str(int(period_value))
        except Exception:
            p = str(period_value)
        sel_list = [t for t in snaps if str(t[0]) == p]
        sel = pd.Index(sel_list)
        return sel if len(sel) > 0 else snaps

    # Fallback: Snapshots sind DatetimeIndex -> nach Jahr filtern
    dt = pd.to_datetime(snaps, errors="coerce")
    try:
        y = int(period_value)
    except Exception:
        return snaps
    mask = dt.year == y
    sel = snaps[mask]
    return sel if len(sel) > 0 else snaps



def build_sankey_fig(
    n: pypsa.Network,
    df_life: pd.DataFrame | None = None,
    period_value=None,
    max_links: int | None = None,
    value_unit: str = "MWh",
    meta_ts: pd.DataFrame | None = None,            
    ts_color_map: dict[str, str] | None = None,
) -> go.Figure:
    """
    Erstellt ein Sankey-Diagramm der integrierten Energieflüsse (MWh) aus Generators, Loads,
    Storage Units, Links und Lines. Optional werden Farben aus den Zeitreihen übernommen
    und aktive Assets je Periode gefiltert.
    
    Inputs: n: pypsa.Network
            df_life: Lifetime-DF (optional)
            period_value: Investitionsperiode oder None
            max_links: optional Top-K Links (nach Flussstärke)
            value_unit: Anzeigeeinheit (intern wird kWh->MWh dividiert)
            meta_ts: Meta-DF der Zeitreihen (optional)
            ts_color_map: dict Zeitreihen-Spalte -> Farbe (optional)
            
    Bestimmt relevante Snapshots per _filter_snapshots_by_period und Gewichte per
    _get_energy_weights
    Wenn df_life+period vorhanden: filtert auf aktive Assets
    Aggregiert Flüsse: Generator->Bus (und ggf. Bus->Generator bei negativen/exportartigen),
    Bus->Load, Bus<->StorageUnit, Bus<->Link, Bus<->Line
    Konsolidiert Stores auf ihren Bus (Busknoten wird als Storelabel gezeigt), optional farblich
    aus Zeitreihen abgeleitet
    Reduziert optional auf Top-K Kanten, skaliert kWh->MWh
    Erzeugt go.Sankey mit Node-Labels, Node-Farben und Hovertemplate
    
    Outputs: go.figure (Sankey-Diagramm)
    """

    snaps_sel = _filter_snapshots_by_period(n, period_value)
    if len(snaps_sel) == 0:
        return go.Figure().update_layout(title="Sankey-Diagramm (keine Werte gefunden)")

    w = _get_energy_weights(n).reindex(snaps_sel).fillna(0.0)

    comps_with_life = set()
    active_set: set[tuple[str, str]] = set()

    if df_life is not None and not df_life.empty and period_value is not None:
        comps_with_life = set(df_life["component"].astype(str).unique())
        active_set = active_assets_in_period(df_life, period_value)

    def _is_active(component: str, name: str) -> bool:
        if not comps_with_life:
            return True
        if component not in comps_with_life:
            return True
        return (component, str(name)) in active_set

    # Farbableitung: gleiche Quelle wie Zeitreihen (meta_ts + ts_color_map)
    def _asset_color(comp: str, asset: str) -> str | None:
        if meta_ts is None or meta_ts.empty or not ts_color_map:
            return None

        m = meta_ts[
            (meta_ts["component"].astype(str) == str(comp)) &
            (meta_ts["asset"].astype(str) == str(asset))
        ]
        if m.empty:
            return None

        pref = {
            "generators": ["p"],
            "loads": ["p", "p_set"],
            "storage_units": ["p"],
            "stores": ["p"],
            "links": ["p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9"],
            "lines": ["p0", "p1"],
        }

        for a in pref.get(str(comp), []):
            cols = m[m["attr"].astype(str) == a].index.tolist()
            for c in cols:
                col = ts_color_map.get(str(c))
                if col:
                    return col

        # Fallback: erste passende TS-Spalte
        for c in m.index.tolist():
            col = ts_color_map.get(str(c))
            if col:
                return col
        return None

    key_to_label: dict[str, str] = {}
    key_to_color: dict[str, str] = {}

    def _register_node(key: str, label: str, color: str | None = None) -> str:
        if key not in key_to_label:
            key_to_label[key] = label
        if color and (key not in key_to_color):
            key_to_color[key] = color
        return key

    # Stores mit separatem Bus "zusammenziehen" (wie bisher), aber mit Farbe aus Store-TS
    store_by_bus: dict[str, list[str]] = {}
    if (
        hasattr(n, "stores") and n.stores is not None and (not n.stores.empty)
        and ("bus" in n.stores.columns)
    ):
        for st_name, b in n.stores["bus"].dropna().astype(str).items():
            if not _is_active("stores", st_name):
                continue
            b = str(b).strip()
            if b:
                store_by_bus.setdefault(b, []).append(str(st_name))

    def _bus_node(bus: str) -> str:
        b = str(bus).strip()
        if not b:
            return _register_node("bus::(unknown)", "(unknown)", "rgba(200,200,200,0.85)")

        # Falls Bus ein/mehrere Stores trägt: Label ersetzen
        if b in store_by_bus and len(store_by_bus[b]) > 0:
            sts_raw = store_by_bus[b]
            sts_disp = [strip_prefix(s) for s in sts_raw]

            if len(sts_disp) == 1:
                label = sts_disp[0]
            elif len(sts_disp) == 2:
                label = f"{sts_disp[0]} + {sts_disp[1]}"
            else:
                label = f"{sts_disp[0]} + {sts_disp[1]} (+{len(sts_disp)-2})"

            # Farbe vom ersten Store ableiten (konsistent, deterministisch)
            col = _asset_color("stores", sts_raw[0])
            return _register_node(f"busstore::{b}", label, col or "rgba(200,200,200,0.85)")

        return _register_node(f"bus::{b}", strip_prefix(b), "rgba(200,200,200,0.85)")

    def _gen_node(gen: str) -> str:
        g = str(gen).strip()
        return _register_node(f"gen::{g}", strip_prefix(g), _asset_color("generators", g))

    def _load_node(ld: str) -> str:
        l = str(ld).strip()
        return _register_node(f"load::{l}", strip_prefix(l), _asset_color("loads", l))

    def _link_node(lk: str) -> str:
        l = str(lk).strip()
        return _register_node(f"link::{l}", strip_prefix(l), _asset_color("links", l))

    def _line_node(ln: str) -> str:
        l = str(ln).strip()
        return _register_node(f"line::{l}", strip_prefix(l), _asset_color("lines", l))

    def _su_node(su: str) -> str:
        s = str(su).strip()
        return _register_node(f"su::{s}", strip_prefix(s), _asset_color("storage_units", s))

    edges: dict[tuple[str, str], float] = {}

    def add_edge(src_key: str, dst_key: str, val: float):
        try:
            v = float(val)
        except Exception:
            return
        if not np.isfinite(v) or v <= 0.0:
            return
        edges[(src_key, dst_key)] = edges.get((src_key, dst_key), 0.0) + v

    def _is_export_like_generator(gen_name: str) -> bool:
        name_l = str(gen_name).lower()
        if "einspeisung" in name_l:
            return True
        if hasattr(n, "generators") and (gen_name in n.generators.index) and ("sign" in n.generators.columns):
            s = n.generators.at[gen_name, "sign"]
            if s is not None and not pd.isna(s):
                try:
                    return float(s) < 0.0
                except Exception:
                    pass
        return False

    # --- GENERATORS ---
    if hasattr(n, "components") and hasattr(n.components, "generators"):
        dyn = n.components.generators.dynamic
        p = dyn.get("p")
        if p is not None and not p.empty:
            psel = p.reindex(snaps_sel)
        if p is not None and not p.empty:
            psel = p.reindex(snaps_sel)
            for gen in psel.columns:
                if gen not in n.generators.index:
                    continue
                if not _is_active("generators", gen):
                    continue

                bus = n.generators.at[gen, "bus"]
                if bus is None or pd.isna(bus) or str(bus).strip() == "":
                    continue

                s = psel[gen].copy()
                e_pos = s.clip(lower=0.0).mul(w).sum()
                e_neg = (-s.clip(upper=0.0)).mul(w).sum()

                if _is_export_like_generator(gen):
                    add_edge(_bus_node(bus), _gen_node(gen), e_pos)
                    add_edge(_gen_node(gen), _bus_node(bus), e_neg)
                else:
                    add_edge(_gen_node(gen), _bus_node(bus), e_pos)
                    add_edge(_bus_node(bus), _gen_node(gen), e_neg)

    # --- LOADS ---
    if hasattr(n, "components") and hasattr(n.components, "loads"):
        dyn = n.components.loads.dynamic
        p = dyn.get("p")
        if p is None or p.empty:
            p = dyn.get("p_set")
        if p is not None and not p.empty:
            psel = p.reindex(snaps_sel)
            cons = psel.clip(lower=0.0).mul(w, axis=0).sum(axis=0)
            for ld, e in cons.items():
                if ld not in n.loads.index:
                    continue
                bus = n.loads.at[ld, "bus"]
                if bus is None or pd.isna(bus) or str(bus).strip() == "":
                    continue
                add_edge(_bus_node(bus), _load_node(ld), e)

    # --- STORAGE_UNITS ---
    if hasattr(n, "components") and hasattr(n.components, "storage_units"):
        dyn = n.components.storage_units.dynamic
        p = dyn.get("p")
        if p is not None and not p.empty:
            psel = p.reindex(snaps_sel)
            charge = (-psel.clip(upper=0.0)).mul(w, axis=0).sum(axis=0)
            discharge = (psel.clip(lower=0.0)).mul(w, axis=0).sum(axis=0)
            for su in psel.columns:
                if su not in n.storage_units.index:
                    continue
                if not _is_active("storage_units", su):
                    continue

                bus = n.storage_units.at[su, "bus"]
                if bus is None or pd.isna(bus) or str(bus).strip() == "":
                    continue
                su_k = _su_node(su)
                b_k = _bus_node(bus)
                add_edge(b_k, su_k, float(charge.get(su, 0.0)))
                add_edge(su_k, b_k, float(discharge.get(su, 0.0)))

    # --- LINKS ---
    if hasattr(n, "components") and hasattr(n.components, "links"):
        dyn = n.components.links.dynamic
        ports = get_existing_link_ports(n, max_i=9)
        for i in ports:
            attr = f"p{i}"
            p = dyn.get(attr)
            if p is None or p.empty:
                continue
            psel = p.reindex(snaps_sel)

            for link in psel.columns:
                if link not in n.links.index:
                    continue
                if not _is_active("links", link):
                    continue

                bus_col = f"bus{i}"
                if bus_col not in n.links.columns:
                    continue
                bus = n.links.at[link, bus_col]
                if bus is None or pd.isna(bus) or str(bus).strip() == "":
                    continue

                series = psel[link]
                e_out = series.clip(lower=0.0).mul(w).sum()
                e_in  = (-series.clip(upper=0.0)).mul(w).sum()

                b_k = _bus_node(bus)
                l_k = _link_node(link)

                add_edge(b_k, l_k, e_out)
                add_edge(l_k, b_k, e_in)

    # --- LINES ---
    if hasattr(n, "components") and hasattr(n.components, "lines"):
        dyn = n.components.lines.dynamic

        for i in (0, 1):
            attr = f"p{i}"
            p = dyn.get(attr)
            if p is None or p.empty:
                continue

            psel = p.reindex(snaps_sel)

            for line in psel.columns:
                if line not in n.lines.index:
                    continue
                if not _is_active("lines", line):
                    continue

                bus_col = f"bus{i}"
                if bus_col not in n.lines.columns:
                    continue
                bus = n.lines.at[line, bus_col]
                if bus is None or pd.isna(bus) or str(bus).strip() == "":
                    continue

                series = psel[line]

                # Konvention: p_i > 0 => Bus(i) -> Line, p_i < 0 => Line -> Bus(i)
                e_in_to_line  = series.clip(lower=0.0).mul(w).sum()
                e_out_of_line = (-series.clip(upper=0.0)).mul(w).sum()

                b_k = _bus_node(bus)
                ln_k = _line_node(line)

                add_edge(b_k, ln_k, e_in_to_line)
                add_edge(ln_k, b_k, e_out_of_line)

    items = [(k, v) for (k, v) in edges.items() if v > 0.0]
    items.sort(key=lambda kv: kv[1], reverse=True)
    if max_links is not None and len(items) > max_links:
        items = items[:max_links]

    # --- kWh -> MWh ---
    items = [(k, float(v) / 1000.0) for (k, v) in items if np.isfinite(v)]
    items = [(k, v) for (k, v) in items if v > 0.0]

    if not items:
        title = "Sankey-Diagramm (keine Werte > 0 gefunden)"
        if isinstance(pd.Index(n.snapshots), pd.MultiIndex) and period_value is not None:
            title += f" – Investitionsperiode {period_value}"
        return go.Figure().update_layout(title=title)

    node_index: dict[str, int] = {}
    labels: list[str] = []
    node_colors: list[str] = []

    def idx(key: str) -> int:
        if key not in node_index:
            node_index[key] = len(labels)
            labels.append(key_to_label.get(key, key))
            node_colors.append(key_to_color.get(key, "rgba(200,200,200,0.85)"))
        return node_index[key]

    sources, targets, values = [], [], []

    for (src_key, dst_key), v in items:
        sources.append(idx(src_key))
        targets.append(idx(dst_key))
        values.append(float(v))

    title = ""
    if isinstance(pd.Index(n.snapshots), pd.MultiIndex) and period_value is not None:
        title += f" – Investitionsperiode {period_value}"

    fig = go.Figure(
        data=[go.Sankey(
            arrangement="snap",
            node=dict(
                label=labels,
                color=node_colors,
                pad=14,
                thickness=14,
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                hovertemplate="%{source.label} → %{target.label}<br>%{value:.2f}  MWh<extra></extra>",
            ),
        )]
    )
    fig.update_layout(title=title, margin=dict(l=20, r=20, t=60, b=20), height=750)
    return fig


# %% Datasets finden + LRU State

def list_nc_files(folder: str) -> list[str]:
    """
    Listet alle .nc-Dateien in einem Ordner auf (vollständige Pfade)
    
    Inputs: folder [str]
    
    Prüft Ordnerexistenz
    Filtert Dateien nach Endung '.nc' (case-insensitive)
    Sortiert die Pfade
    
    Outputs: list [str]
    """
    if not folder or not os.path.isdir(folder):
        return []
    files = []
    for fn in os.listdir(folder):
        if fn.lower().endswith(".nc"):
            files.append(os.path.join(folder, fn))
    return sorted(files)


def _empty_state(reason: str = "keine Daten"):
    """
    Erzeugt einen konsistenten Default-State, wenn kein Dataset geladen werden kann
    
    Inputs: reason [str]
    
    Setzt ok=False und füllt alle erwarteten State-Keys mit leeren DataFrames/Defaults
    
    Outputs: dict: Dashboard-State
    """
    return {
        "ok": False,
        "reason": reason,
        "n": None,
        "years": [],
        "has_mip": False,
        "default_sector": "Sonstige",
        "df_dyn_all": pd.DataFrame(columns=["period", "timestep"]),
        "meta_ts": pd.DataFrame(),
        "timeseries_color_map": {},
        "by_sector_p": {s: pd.DataFrame() for s in SECTORS},
        "by_sector_e": {s: pd.DataFrame() for s in SECTORS},
        "subcarrier_color_map": {},
        "df_life": pd.DataFrame(),
        "df_cost": pd.DataFrame(),
        "df_inv_capex": pd.DataFrame(),
        "years_cost": [],
        "has_mip_cost": False,
        "base_period": None,
        "compare_years": [],
        "ts_period_options": [{"label": "Single", "value": "Single"}],
        "default_ts_period": "Single",
        "sank_period_options": [{"label": "Single", "value": "Single"}],
        "default_sank_period": "Single",
    }


def _build_dataset_state(nc_path: str) -> dict:
    """
    Lädt eine .nc-Datei als pypsa.Network und baut alle abgeleiteten Tabellen,
    Farbzuordnungen und UI-Optionen für das Dashboard (State-Object)
    
    Inputs: nc_path: str (Pfad zur .nc-Datei)
    
    Validiert Dateipfad; lädt pypsa.Network
    Erzeugt Bus-Taxonomie (ensure_bus_taxonomy)
    Ermittelt MIP-Jahre und has_mip
    Baut Zeitreihen-DF (build_dynamic_timeseries_df), interne Store-Busse, Meta
    (build_timeseries_meta) und Farben
    Erstellt TS-Periodenoptionen aus df_dyn_all['period']
    Baut Lifetime-DF, Leistungen (kW) und Kapazitäten (kWh), expandiert diese auf
    aktive Perioden und bereitet sie nach Sektoren auf
    Sammelt alle Subcarrier und erstellt subcarrier_color_map
    Setzt default_sector anhand verfügbarer Daten
    Erstellt Sankey-Periodenoptionen anhand n.snapshots (MultiIndex/Tuple/Datetime)
    Berechnet Kosten (build_costs_df) und Investitions-CAPEX (build_investment_capex_df)
    Leitet years_cost, has_mip_cost, base_period und compare_years ab
    Gibt State-Dict zurück; Fehler werden abgefangen und in _empty_state resultiert
    
    Outputs: dict: vollständiger Dataset-State
    """
    
    if not nc_path or (not os.path.isfile(nc_path)):
        return _empty_state("Datei nicht gefunden")

    try:
        n = pypsa.Network(nc_path)

        # Taxonomie vorbereiten
        ensure_bus_taxonomy(n)

        years = get_investment_years(n)
        has_mip = bool(getattr(n, "has_investment_periods", False)) and len(years) > 0

        # Zeitreihen
        df_dyn_all = build_dynamic_timeseries_df(n, add_component_prefix=True)
        internal_store_buses = infer_internal_store_buses(n)
        meta_ts = build_timeseries_meta(n, df_dyn_all, internal_store_buses) if not df_dyn_all.empty else pd.DataFrame()
        timeseries_color_map = make_label_color_map(meta_ts.index.tolist() if (meta_ts is not None and not meta_ts.empty) else [])

        # Perioden-Optionen
        if "period" in df_dyn_all.columns:
            periods = sorted(df_dyn_all["period"].dropna().astype(str).unique().tolist())
        else:
            periods = ["Single"]
        ts_period_options = [{"label": p, "value": p} for p in periods] if periods else [{"label": "Single", "value": "Single"}]
        default_ts_period = ts_period_options[0]["value"] if ts_period_options else "Single"

        # Lifetime
        df_life = build_lifetime_table(n)

        # Kapazitäten (kW): aktiv je Periode
        df_caps = build_capacity_table(n)
        df_caps_active = expand_caps_to_active_periods(df_caps, df_life, years, value_col="p_nom")
        by_sector_p, _ = prepare_multicategory(df_caps_active, n, add_component_prefix=True, value_col="p_nom")

        # Energie (kWh): aktiv je Periode
        df_energy = build_energy_capacity_table(n)
        df_energy_active = expand_caps_to_active_periods(df_energy, df_life, years, value_col="e_nom")
        by_sector_e, _ = prepare_multicategory(df_energy_active, n, add_component_prefix=True, value_col="e_nom")

        # Farblogik Subcarrier
        all_subcarriers = _collect_subcarriers(
            by_sector_p,
            by_sector_e,
            df_life,
            meta_ts.reset_index() if (meta_ts is not None and not meta_ts.empty) else None
        )
        subcarrier_color_map = make_subcarrier_color_map(all_subcarriers)

        # Default-Sektor
        available = [
            s for s in SECTORS
            if (s in by_sector_p and by_sector_p[s] is not None and not by_sector_p[s].empty)
            or (s in by_sector_e and by_sector_e[s] is not None and not by_sector_e[s].empty)
        ]
        default_sector = available[0] if available else "Sonstige"

        # Sankey Perioden
        snaps = pd.Index(n.snapshots)

        if isinstance(snaps, pd.MultiIndex):
            periods = sorted(set(snaps.get_level_values(0).astype(str)))

        elif len(snaps) > 0 and isinstance(snaps[0], tuple) and len(snaps[0]) >= 2:
        # pypsa/netcdf: snapshots als Tupel (period, timestamp)
             periods = sorted({str(t[0]) for t in snaps})

        else:
        # Datetime-Snapshots
            dt = pd.to_datetime(snaps, errors="coerce")
            dt = dt[~pd.isna(dt)]
            periods = sorted({str(int(y)) for y in dt.year})

        sank_period_options = [{"label": p, "value": p} for p in periods] if periods else [{"label":"Single","value":"Single"}]
        default_sank_period = sank_period_options[0]["value"]

        # Kosten
        df_cost = build_costs_df(n)
        df_inv_capex = build_investment_capex_df(n)

        years_cost = years[:]
        has_mip_cost = has_mip and (df_cost is not None) and (not df_cost.empty)

        base_period = str(min(years_cost)) if (has_mip_cost and years_cost) else None
        compare_years = [y for y in years_cost if str(y) != str(base_period)] if (has_mip_cost and base_period is not None) else []

        return {
            "ok": True,
            "reason": "",
            "n": n,
            "years": years,
            "has_mip": has_mip,
            "default_sector": default_sector,
            "df_dyn_all": df_dyn_all,
            "meta_ts": meta_ts,
            "timeseries_color_map": timeseries_color_map,
            "by_sector_p": by_sector_p,
            "by_sector_e": by_sector_e,
            "subcarrier_color_map": subcarrier_color_map,
            "df_life": df_life,
            "df_cost": df_cost,
            "df_inv_capex": df_inv_capex,
            "years_cost": years_cost,
            "has_mip_cost": has_mip_cost,
            "base_period": base_period,
            "compare_years": compare_years,
            "ts_period_options": ts_period_options,
            "default_ts_period": default_ts_period,
            "sank_period_options": sank_period_options,
            "default_sank_period": default_sank_period,
        }

    except Exception as e:
        traceback.print_exc()
        return _empty_state(f"State-Build Fehler: {e!s}")


@lru_cache(maxsize=LRU_CACHE_SIZE)
def get_dataset_state(nc_path: str) -> dict:
    """
    LRU-gecachter Zugriff auf Dataset-States; serialisiert State-Build per globalem Lock
    (Stabilität bei parallelen Dash-Callbacks)
    
    Inputs: nc_path: str
    
    Lock _STATE_BUILD_LOCK hält parallele Builds zurück
    Ruft _build_dataset_state auf; Ergebnis wird durch lru_cache gepuffert
    (maxsize=LRU_CACHE_SIZE)
    
    Outputs: dict: Dataset-State
    """
    with _STATE_BUILD_LOCK:
        return _build_dataset_state(nc_path)


#%% Dash App (Layout)

nc_files = list_nc_files(DATA_DIR)
file_options = [{"label": os.path.basename(p), "value": p} for p in nc_files]
default_file = file_options[0]["value"] if file_options else None

app = Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div(
    style={"padding": "12px"},
    children=[
        html.Div(
            style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "gap": "16px"},
            children=[
                html.H2("Dashboard", style={"margin": 0}),
                html.Div(
                    style={"display": "flex", "alignItems": "center", "gap": "8px"},
                    children=[
                        html.Label("Datenbasis", style={"margin": 0}),
                        dcc.Dropdown(
                            id="datafile-dropdown",
                            options=file_options,
                            value=default_file,
                            clearable=False,
                            style={"width": "420px"},
                            placeholder="Keine .nc-Dateien gefunden",
                        ),
                    ],
                ),
            ],
        ),

        html.Hr(),

        dcc.Tabs(
            id="main-tabs",
            value="tab-cap",
            children=[
                dcc.Tab(
                    id="tab-cap",
                    label="Leistungen/ Kapazitäten",
                    value="tab-cap",
                    children=[
                        html.Div(
                            style={"display": "flex", "gap": "16px"},
                            children=[
                                html.Div(
                                    style={"flex": "1"},
                                    children=[
                                        html.H3("Nennleistungen"),
                                        dcc.Graph(id="cap-power-graph", style={"height": "420px"}),
                                        html.H3("Speicherkapazitäten"),
                                        dcc.Graph(id="cap-energy-graph", style={"height": "420px"}),
                                    ],
                                ),
                                html.Div(
                                    style={"width": "120px", "borderLeft": "1px solid #ddd", "paddingLeft": "12px"},
                                    children=[
                                        html.H4("Filter"),
                                        html.Label("Sektor"),
                                        dcc.Dropdown(
                                            id="cap-sector-dropdown",
                                            options=[{"label": s, "value": s} for s in SECTORS],
                                            value="Sonstige",
                                            clearable=False,
                                        ),
                                        html.Div(
                                            id="cap-year-filter",
                                            style={"display": "none", "marginTop": "12px"},
                                            children=[
                                                html.Label("Jahre"),
                                                dcc.Dropdown(
                                                    id="cap-year-dropdown",
                                                    options=[],
                                                    value=[],
                                                    multi=True,
                                                    clearable=False,
                                                    placeholder="Jahre wählen…",
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),

                dcc.Tab(
                    id="tab-ts",
                    label="Zeitreihen",
                    value="tab-ts",
                    children=[
                        html.Div(
                            style={"display": "flex", "gap": "16px"},
                            children=[
                                html.Div(
                                    style={"flex": "1"},
                                    children=[
                                        html.H3("Zeitreihen nach Sektor und Investitionsperiode"),
                                        dcc.Graph(id="ts-strom-graph", style={"height": "420px"}),
                                        dcc.Graph(id="ts-waerme-graph", style={"height": "420px"}),
                                        dcc.Graph(id="ts-sonst-graph", style={"height": "420px"}),
                                    ],
                                ),
                                html.Div(
                                    id="ts-filter-container",
                                    style={"width": "240px", "borderLeft": "1px solid #ddd", "paddingLeft": "12px", "display": "none"},
                                    children=[
                                        html.H4("Filter"),
                                        html.Label("Investitionsperiode"),
                                        dcc.Dropdown(
                                            id="ts-period-dropdown",
                                            options=[{"label": "Single", "value": "Single"}],
                                            value="Single",
                                            clearable=False,
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),

                dcc.Tab(
                    id="tab-exp",
                    label="Ausbaupfad/ Lebensdauer",
                    value="tab-exp",
                    disabled=True,
                    children=[
                        html.Div(
                            style={"display": "flex", "gap": "16px"},
                            children=[
                                html.Div(
                                    style={"flex": "1", "minWidth": 0},
                                    children=[
                                        html.H3("Ausbaupfad"),
                                        dcc.Graph(id="exp-path-graph", style={"height": "520px"}),
                                        html.Hr(),
                                        html.H3("Lebensdauer / Aktivitätszeitraum"),
                                        dcc.Graph(id="exp-life-graph", style={"height": "650px"}),
                                    ],
                                ),
                                html.Div(
                                    style={"width": "120px", "borderLeft": "1px solid #ddd", "paddingLeft": "12px"},
                                    children=[
                                        html.H4("Filter"),
                                        html.Label("Sektor"),
                                        dcc.Dropdown(
                                            id="exp-sector-dropdown",
                                            options=[{"label": s, "value": s} for s in SECTORS],
                                            value="Sonstige",
                                            clearable=False,
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),

                dcc.Tab(
                    id="tab-sankey",
                    label="Sankey-Diagramm (Energieflüsse)",
                    value="tab-sankey",
                    children=[
                        html.Div(
                            style={"display": "flex", "gap": "16px"},
                            children=[
                                html.Div(
                                    style={"flex": "1", "minWidth": 0},
                                    children=[
                                        dcc.Graph(id="sankey-graph", style={"height": "780px"}),
                                    ],
                                ),
                                html.Div(
                                    id="sankey-filter-container",
                                    style={"width": "240px", "borderLeft": "1px solid #ddd", "paddingLeft": "12px", "display": "none"},
                                    children=[
                                        html.H4("Filter"),
                                        html.Label("Investitionsperiode"),
                                        dcc.Dropdown(
                                            id="sankey-period-dropdown",
                                            options=[{"label": "Single", "value": "Single"}],
                                            value="Single",
                                            clearable=False,
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),

                dcc.Tab(
                    id="tab-cost",
                    label="Wirtschaftlichkeit (CAPEX/OPEX)",
                    value="tab-cost",
                    disabled=True,
                    children=[
                        html.Div(
                            style={"display": "flex", "flexDirection": "column", "gap": "18px"},
                            children=[
                                html.Div(
                                    children=[
                                        html.H3("Kosten je Investitionsperiode (CAPEX + OPEX)"),
                                        dcc.Graph(id="cost-period-totals-graph", style={"height": "420px"}),
                                        
                                        html.H3("Gesamtinvestitionen"),
                                        dcc.Graph(id="cost-investment-capex-graph", style={"height": "420px"}),
                                    ]
                                ),
                                html.Div(
                                    style={"display": "flex", "gap": "16px", "alignItems": "flex-start"},
                                    children=[
                                        html.Div(
                                            style={"flex": "1", "minWidth": 0},
                                            children=[
                                                html.H3("Kostenverteilung"),
                                                dcc.Graph(id="cost-composition-graph", style={"height": "520px"}),
                                            ],
                                        ),
                                        html.Div(
                                            style={"width": "320px", "borderLeft": "1px solid #ddd", "paddingLeft": "12px", "alignSelf": "stretch"},
                                            children=[
                                                html.H4("Filter"),
                                                html.Div(id="cost-base-year-text"),
                                                html.Br(),
                                                html.Div(
                                                    id="cost-slider-container",
                                                    style={"display": "none"},
                                                    children=[
                                                        html.Label("Vergleichsjahr"),
                                                        dcc.Slider(
                                                            id="cost-year-slider",
                                                            min=0, max=0, step=None,
                                                            marks={},
                                                            value=0,
                                                            disabled=True,
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                dcc.Tab(
        id="tab-var",
        label="Variantenvergleich / Sensitivität",
        value="tab-var",
        disabled=(len(file_options) < 2),
        children=[
            html.Div(style={"height": "12px"}), # Spacer
            html.Div(
                style={"display": "flex", "flexDirection": "column", "gap": "14px"},
                children=[
                    # Kopfzeile: Vergleichs-Auswahl
                    html.Div(
                        style={"display": "flex", "gap": "16px", "alignItems": "flex-end", "flexWrap": "wrap"},
                        children=[
                            html.Div(
                                style={"minWidth": "420px"},
                                children=[
                                    html.Label("Vergleichsvariante"),
                                    dcc.Dropdown(
                                        id="var-compare-dropdown",
                                        options=file_options,   # wird per Callback gefiltert
                                        value=None,
                                        clearable=False,
                                        placeholder="Variante wählen…",
                                    ),
                                ],
                            ),
                            html.Div(
                                id="var-year-filter",
                                style={"display": "none", "minWidth": "220px"},
                                children=[
                                    html.Label("Betrachtungsjahr"),
                                    dcc.Dropdown(
                                        id="var-year-dropdown",
                                        options=[],
                                        value=None,
                                        clearable=False,
                                    ),
                                ],
                            ),
                        ],
                    ),

                    html.Hr(),

                    html.Div(
                        children=[
                            html.H3("Systemkosten"),
                            html.Div(
                                style={"display": "flex", "gap": "16px", "alignItems": "flex-start"},
                                children=[
                                    # Graph links
                                    html.Div(
                                        style={"flex": "1", "minWidth": 0},
                                        children=[
                                            dcc.Graph(id="var-cost-compare-graph", style={"height": "420px"}),
                                        ],
                                    ),
                                    # Info rechts
                                    html.Div(
                                        style={
                                            "width": "320px",
                                            "borderLeft": "1px solid #ddd",
                                            "paddingLeft": "12px",
                                            "alignSelf": "stretch",
                                        },
                                        children=[
                                            html.H4("Derzeit ausgewählt"),
                                            html.Div(
                                                id="var-compare-hint",
                                                style={"color": "#555", "whiteSpace": "pre-wrap"},
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ]
                    ),

                    html.Div(
                        children=[
                            html.H3(" Leistung aller Komponenten"),
                            dcc.Graph(id="var-capacity-compare-graph", style={"height": "520px"}),
                        ]
                    ),
                    html.Div(
                        children=[
                            html.H3("Kapazität der Speicherkomponenten"),
                            dcc.Graph(id="var-storage-capacity-compare-graph", style={"height": "520px"}),
                        ]
                    ),

                ],
            )
        ],
    ),

            ],
        ),
    ],
)


# %% Control Sync Callback (Dataset -> UI Defaults/Visibility)

@app.callback(
    Output("cap-year-dropdown", "options"),
    Output("cap-year-dropdown", "value"),
    Output("cap-year-filter", "style"),

    Output("ts-period-dropdown", "options"),
    Output("ts-period-dropdown", "value"),
    Output("ts-filter-container", "style"),

    Output("sankey-period-dropdown", "options"),
    Output("sankey-period-dropdown", "value"),
    Output("sankey-filter-container", "style"),

    Output("tab-exp", "disabled"),
    Output("tab-cost", "disabled"),

    Output("cap-sector-dropdown", "value"),
    Output("exp-sector-dropdown", "value"),

    Output("cost-base-year-text", "children"),
    Output("cost-slider-container", "style"),
    Output("cost-year-slider", "marks"),
    Output("cost-year-slider", "max"),
    Output("cost-year-slider", "value"),
    Output("cost-year-slider", "disabled"),

    Output("main-tabs", "value"),
    Input("datafile-dropdown", "value"),
    State("main-tabs", "value"),
)
def sync_controls_for_dataset(nc_path, current_tab):
    """
    Dash-Callback: synchronisiert UI-Controls und Tab-Visibility nach Dataset-Wechsel
    (Jahresfilter, Periodenfilter, Tab enable/disable, Defaults)
    
    Inputs: nc_path: str (aus Dropdown)
            current_tab: aktueller Tab-Wert (State)

    Lädt State per get_dataset_state
    Konfiguriert Jahresfilter im Kapazitäten-Tab (MIP: Jahre, sonst hidden)
    Konfiguriert TS- und Sankey-Periodenfilter (nur sichtbar bei MIP)
    Aktiviert/Deaktiviert Ausbaupfad-Tab (nur MIP) und Kosten-Tab (wenn Kosten oder
    Investitionen vorhanden)                                                      
    Setzt Default-Sektoren
    Konfiguriert Kosten-Slider: Base-Periode und Vergleichsjahre
    Fallback: wenn aktueller Tab jetzt disabled ist, springt auf 'tab-cap'
    
    Outputs: Mehrere Dash-Outputs: Dropdown-Optionen/Values/Styles, Tab-Flags, Slider-Config,
    nächster Tab
    """
    st = get_dataset_state(nc_path) if nc_path else _empty_state("keine Datei gewählt")

    # CAP-Year UI
    if st["has_mip"] and st["years"]:
        cap_year_options = [{"label": str(y), "value": str(y)} for y in st["years"]]
        cap_year_value = [str(y) for y in st["years"]]  # Default: alle Jahre
        cap_year_style = {"display": "block", "marginTop": "12px"}
    else:
        cap_year_options = [{"label": "Single", "value": "Single"}]
        cap_year_value = ["Single"]
        cap_year_style = {"display": "none", "marginTop": "12px"}

    # TS UI
    ts_opts = st["ts_period_options"]
    ts_val = st["default_ts_period"]
    ts_style = {"display": "block"} if st["has_mip"] else {"display": "none"}

    # Sankey UI
    sank_opts = st["sank_period_options"]
    sank_val = st["default_sank_period"]
    sank_style = {"display": "block"} if st["has_mip"] else {"display": "none"}

    # Tabs enabled/disabled
    exp_disabled = not st["has_mip"]
    cost_disabled = (
    ((st["df_cost"] is None) or st["df_cost"].empty)
    and (st.get("df_inv_capex") is None or st["df_inv_capex"].empty)
    )

    # Default sectors
    cap_sector_val = st["default_sector"]
    exp_sector_val = st["default_sector"]

    # Cost slider configuration
    if (not cost_disabled) and st["has_mip_cost"] and st["years_cost"]:
        base_p = st["base_period"]
        compare_years = st["compare_years"]
        base_txt = f"Base Year: {base_p}"
        if compare_years:
            marks = {i: str(y) for i, y in enumerate(compare_years)}
            slider_max = max(len(compare_years) - 1, 0)
            slider_val = 0
            slider_disabled = (len(compare_years) == 0)
            slider_style = {"display": "block"}
        else:
            marks = {}
            slider_max = 0
            slider_val = 0
            slider_disabled = True
            slider_style = {"display": "none"}
    elif not cost_disabled:
        base_txt = "Einjährige Analyse - kein Vergleichswert"
        slider_style = {"display": "none"}
        marks = {}
        slider_max = 0
        slider_val = 0
        slider_disabled = True
    else:
        base_txt = "Keine Kostendaten in dieser Datenbasis."
        slider_style = {"display": "none"}
        marks = {}
        slider_max = 0
        slider_val = 0
        slider_disabled = True

    # Tab fallback, falls aktueller Tab jetzt disabled ist
    next_tab = current_tab or "tab-cap"
    if next_tab == "tab-exp" and exp_disabled:
        next_tab = "tab-cap"
    if next_tab == "tab-cost" and cost_disabled:
        next_tab = "tab-cap"

    return (
        cap_year_options, cap_year_value, cap_year_style,
        ts_opts, ts_val, ts_style,
        sank_opts, sank_val, sank_style,
        exp_disabled, cost_disabled,
        cap_sector_val, exp_sector_val,
        base_txt, slider_style, marks, slider_max, slider_val, slider_disabled,
        next_tab
    )

#%% Variantenvergleich: Sync (Dropdowns + Sichtbarkeit)

@app.callback(
    Output("var-compare-dropdown", "options"),
    Output("var-compare-dropdown", "value"),
    Output("var-year-dropdown", "options"),
    Output("var-year-dropdown", "value"),
    Output("var-year-filter", "style"),
    Output("tab-var", "disabled"),
    Input("datafile-dropdown", "value"),
    State("var-compare-dropdown", "value"),
    State("var-year-dropdown", "value"),
)
def sync_variant_compare_controls(base_path, current_cmp_path, current_year):
    """
    Dash-Callback: synchronisiert die Controls im Variantenvergleich (Vergleichsdatei-Optionen,
    Year-Dropdown Sichtbarkeit)
    
    Inputs: base_path: str (Base-Dataset)
            current_cmp_path: str (bisherige Vergleichsdatei)
            current_year: str (bisheriges Jahr)
    
    Wenn weniger als 2 Dateien: deaktiviert Tab und versteckt Year-Filter
    Filtert file_options so, dass Base nicht als Vergleich wählbar ist
    Validiert/setzt cmp_path (Fallback: erste gültige Option)
    Lädt States für base und compare (Cache warmup)
    Wenn Base MIP: baut Year-Options und wählt default/min-Jahr; sonst versteckt Year-Filter
    
    Outputs: Dash-Outputs: options/value für compare, year options/value/style, tab-var disabled
    """
    if len(file_options) < 2 or base_path is None:
        return [], None, [], None, {"display": "none"}, True

    opts = [o for o in file_options if o.get("value") != base_path]
    valid_values = {o["value"] for o in opts}
    cmp_path = current_cmp_path if current_cmp_path in valid_values else (opts[0]["value"] if opts else None)

    st_base = get_dataset_state(base_path) if base_path else _empty_state("keine Datei")
    st_cmp  = get_dataset_state(cmp_path) if cmp_path else _empty_state("keine Vergleichsdatei")
    _ = st_cmp

    if st_base.get("has_mip", False) and st_base.get("years", []):
        years = st_base["years"]
        year_opts = [{"label": str(y), "value": str(y)} for y in years]
        years_set = {str(y) for y in years}
        y_val = current_year if (current_year in years_set) else str(min(years))
        year_style = {"display": "block", "minWidth": "220px"}
        return opts, cmp_path, year_opts, y_val, year_style, False

    return opts, cmp_path, [], None, {"display": "none"}, False


    # Single-year (kein MIP)
    hint = f"Base: {_basename(base_path)} \n\nVergleich: {_basename(cmp_path)}"
    return opts, cmp_path, [], None, {"display": "none"}, False, hint

@app.callback(
    Output("var-compare-hint", "children"),
    Input("datafile-dropdown", "value"),
    Input("var-compare-dropdown", "value"),
    Input("var-year-dropdown", "value"),
)
def update_variant_compare_hint(base_path, cmp_path, year_val):
    """
    Dash-Callback: aktualisiert den Hinweistext im Variantenvergleich (zeigt Base, Vergleich
    und optional Jahr)
    
    Inputs: base_path: str
            cmp_path: str
            year_val: str|None
            
    Validiert, dass mindestens zwei Dateien existieren und Pfade gesetzt sind
    Lädt base-State, um MIP zu erkennen
    Formatiert Text: 'Base: ... ------ Vergleich: ...' plus Jahr bei MIP
    
    str: Hint-Text
    """
    if len(file_options) < 2 or base_path is None or cmp_path is None:
        return "Mindestens zwei .nc-Dateien erforderlich."

    st_base = get_dataset_state(base_path) if base_path else _empty_state("keine Datei")

    base_txt = _basename(base_path)
    cmp_txt  = _basename(cmp_path)

    if st_base.get("has_mip", False) and st_base.get("years", []):
        y = str(year_val) if year_val is not None else str(min(st_base["years"]))
        return f"Base: {base_txt} \n\nVergleich: {cmp_txt} \n\nJahr: {y}"

    return f"Base: {base_txt} \n\nVergleich: {cmp_txt}"


#%% Graph Callbacks (alle dataset-basiert)

@app.callback(
    Output("cap-power-graph", "figure"),
    Output("cap-energy-graph", "figure"),
    Input("datafile-dropdown", "value"),
    Input("cap-sector-dropdown", "value"),
    Input("cap-year-dropdown", "value"),
)
def update_cap_graphs(nc_path, sector, selected_years):
    """
    Dash-Callback: aktualisiert Nennleistungs- und Speicherkapazitäts-Balkendiagramme für
    einen gewählten Sektor und (optional) Jahre
    
    Inputs: nc_path: str
            sector: str
            selected_years: list[str]
    
    Lädt State
    Validiert selected_years gegen erlaubte Jahre; fallback: alle Jahre
    Filtert df_sector_p/e nach Jahren via _filter_df_sector_years
    Erzeugt Figuren via build_sector_bar (mit subcarrier_color_map)
    
    Outputs: aktualisierte (fig_p, fig_e): go.Figure, go.Figure    
    """
    st = get_dataset_state(nc_path) if nc_path else _empty_state("keine Datei")
    if st["has_mip"] and st["years"]:
        # nur echte Jahre zulassen
        allowed = {str(y) for y in st["years"]}
        selected_years = [y for y in (selected_years or []) if str(y) in allowed]
        if not selected_years:
            selected_years = [str(y) for y in st["years"]]  # fallback: alle
    else:
        selected_years = ["Single"]


    if sector not in SECTORS:
        sector = st["default_sector"]

    dfp = st["by_sector_p"].get(sector)
    dfe = st["by_sector_e"].get(sector)

    dfp_f, years_f = _filter_df_sector_years(dfp, selected_years, st["years"])
    dfe_f, _ = _filter_df_sector_years(dfe, selected_years, st["years"])

    fig_p = build_sector_bar(dfp_f, sector, years_f, value_col="p_nom", unit="kW", title_prefix="Nennleistungen",
                             color_map=st["subcarrier_color_map"])
    fig_e = build_sector_bar(dfe_f, sector, years_f, value_col="e_nom", unit="kWh", title_prefix="Speicherkapazität",
                             color_map=st["subcarrier_color_map"])
    return fig_p, fig_e


@app.callback(
    Output("ts-strom-graph", "figure"),
    Output("ts-waerme-graph", "figure"),
    Output("ts-sonst-graph", "figure"),
    Input("datafile-dropdown", "value"),
    Input("ts-period-dropdown", "value"),
)
def update_timeseries_by_period(nc_path, period_value):
    """
    Dash-Callback: aktualisiert Zeitreihen-Plots für Strom/Wärme/Sonstige für eine gewählte
    Investitionsperiode
    
    Inputs: nc_path: str
            period_value: str
    
    Lädt State und validiert period_value gegen ts_period_options
    Filtert df_dyn_all auf period_value (falls vorhanden)
    Bei MIP: filtert meta auf aktive Assets in der Periode
    Erzeugt drei Figuren via build_sector_timeseries_fig (mit timeseries_color_map)
    
    Outputs: aktualisierte (fig_strom, fig_waerme, fig_sonst): go.Figure    
    """
    st = get_dataset_state(nc_path) if nc_path else _empty_state("keine Datei")

    # period_value validieren (bei Dataset-Wechsel kann ein alter Wert hängen bleiben)
    valid_periods = {o["value"] for o in st.get("ts_period_options", [])}
    if (period_value is None) or (str(period_value) not in {str(v) for v in valid_periods}):
        period_value = st.get("default_ts_period", "Single")

    df = st["df_dyn_all"].copy()
    if "period" in df.columns and period_value is not None:
        df = df[df["period"].astype(str) == str(period_value)]

    if st["has_mip"]:
        active_set = active_assets_in_period(st["df_life"], period_value)
        meta_active = filter_meta_to_active(st["meta_ts"], active_set, st["df_life"])
    else:
        meta_active = st["meta_ts"]

    fig_s = build_sector_timeseries_fig(df, meta_active, "Strom", unit="kW", max_traces=30,
                                        ts_color_map=st["timeseries_color_map"])
    fig_w = build_sector_timeseries_fig(df, meta_active, "Wärme", unit="kW", max_traces=30,
                                        ts_color_map=st["timeseries_color_map"])
    fig_o = build_sector_timeseries_fig(df, meta_active, "Sonstige", unit="kW", max_traces=30,
                                        ts_color_map=st["timeseries_color_map"])
    return fig_s, fig_w, fig_o


@app.callback(
    Output("exp-path-graph", "figure"),
    Output("exp-life-graph", "figure"),
    Input("datafile-dropdown", "value"),
    Input("exp-sector-dropdown", "value"),
)
def update_expansion_tab(nc_path, sector):
    """
    Dash-Callback: aktualisiert Ausbaupfad- und Lifetime-Plots für einen gewählten Sektor 
    (nur MIP)
    
    Inputs: nc_path: str
            sector: str
            
    Lädt State, validiert Sektor
    Wenn kein MIP: gibt Platzhalter-Figuren zurück
    Sonst: build_expansion_path_scatter + build_lifetime_timeline_fig

    Outputs: aktualisierte (fig_exp, fig_life): go.Figure      
    """
    st = get_dataset_state(nc_path) if nc_path else _empty_state("keine Datei")

    if sector not in SECTORS:
        sector = st["default_sector"]

    if not st["has_mip"]:
        fig_exp = go.Figure().update_layout(title="Ausbaupfad (nur bei MIP verfügbar)")
        fig_life = go.Figure().update_layout(title="Lebensdauer (nur bei MIP verfügbar)")
        return fig_exp, fig_life

    fig_exp = build_expansion_path_scatter(st["by_sector_p"], sector, st["years"], value_col="p_nom", unit="kW",
                                           max_series=25, color_map=st["subcarrier_color_map"])
    fig_life = build_lifetime_timeline_fig(st["df_life"], sector, color_map=st["subcarrier_color_map"])
    return fig_exp, fig_life


@app.callback(
    Output("sankey-graph", "figure"),
    Input("datafile-dropdown", "value"),
    Input("sankey-period-dropdown", "value"),
)
def update_sankey(nc_path, period_value):
    """
    Dash-Callback: aktualisiert das Sankey-Diagramm für eine gewählte Periode (bei MIP) bzw
    für alle Snapshots (Single-year)
    
    Inputs: nc_path: str
            period_value: str
            
    Lädt State und validiert period_value gegen sank_period_options
    Wenn kein Netzwerk geladen: gibt Platzhalter zurück
    Wenn MIP: ruft build_sankey_fig mit period_value und Farbinfos auf
    Sonst: ruft build_sankey_fig ohne Periodenfilter auf
    
    Outputs: aktualisiertes Sankey-Diagramm, go.figure
    """
    st = get_dataset_state(nc_path) if nc_path else _empty_state("keine Datei")
    
    valid = {o["value"] for o in st.get("sank_period_options", [])}
    if (period_value is None) or (str(period_value) not in {str(v) for v in valid}):
        period_value = st.get("default_sank_period", "Single")

    if not st["ok"] or st["n"] is None:
        return go.Figure().update_layout(title="Sankey (keine Datenbasis)")

    if st["has_mip"]:
        return build_sankey_fig(st["n"], 
                                df_life=st["df_life"], 
                                period_value=period_value, 
                                max_links=None, 
                                value_unit="kWh",
                                meta_ts=st["meta_ts"],
                                ts_color_map=st["timeseries_color_map"],
                                )
    
    return build_sankey_fig(
        st["n"],
        df_life=st["df_life"],
        period_value=None,
        max_links=None,
        value_unit="kWh",
        meta_ts=st["meta_ts"],                           
        ts_color_map=st["timeseries_color_map"],
    )

@app.callback(
    Output("cost-period-totals-graph", "figure"),
    Output("cost-investment-capex-graph", "figure"),
    Output("cost-composition-graph", "figure"),
    Input("datafile-dropdown", "value"),
    Input("cost-year-slider", "value"),
)

def update_cost_tab(nc_path, slider_idx):
    """
    Dash-Callback: aktualisiert Kostenplots (Totals, Investitions-CAPEX, Kostenverteilung)
    abhängig von MIP/Single und Slider-Index
    
    Inputs: nc_path: str
            slider_idx: int|None
            
    Lädt State
    Baut Investitions-CAPEX-Figur (MIP oder Single) via build_investment_capex_totals_fig
    Wenn df_cost leer: Platzhalter für Kostenfiguren
    Bei MIP: build_cost_totals_fig und build_cost_composition_fig (Base vs Selected) basierend
    auf slider_idx
    Bei Single: build_cost_totals_singleyear_fig und build_cost_single_fig
    
    Outputs: Aktualisierte (fig_tot, fig_inv, fig_comp): go.Figure
    """
    st = get_dataset_state(nc_path) if nc_path else _empty_state("keine Datei")

    df_inv = st.get("df_inv_capex")
    fig_inv = build_investment_capex_totals_fig(df_inv, st["years_cost"]) if st.get("has_mip", False) else \
              build_investment_capex_totals_fig(df_inv, [])

    df_cost = st["df_cost"]
    if df_cost is None or df_cost.empty:
        f1 = go.Figure().update_layout(title="Kosten (keine Daten)")
        f3 = go.Figure().update_layout(title="Kostenverteilung (keine Daten)")
        return f1, fig_inv, f3

    if st["has_mip_cost"]:
        base_period = st["base_period"]
        compare_years = st["compare_years"]

        fig_tot = build_cost_totals_fig(df_cost, st["years_cost"])

        if not compare_years:
            fig_comp = go.Figure().update_layout(
                title=f"Kostenverteilung: kein Vergleichsjahr vorhanden (nur Base Year {base_period})"
            )
            return fig_tot, fig_inv, fig_comp

        if slider_idx is None:
            slider_idx = 0
        slider_idx = int(slider_idx)
        slider_idx = max(0, min(slider_idx, len(compare_years) - 1))
        selected_period = str(compare_years[slider_idx])

        fig_comp = build_cost_composition_fig(df_cost, str(base_period), selected_period, max_components=30)
        return fig_tot, fig_inv, fig_comp

    # Single-year
    single_period = "Single"
    fig_tot = build_cost_totals_singleyear_fig(df_cost, single_period)
    df_single = df_cost[df_cost["period"].astype(str) == single_period].copy()
    fig_comp = build_cost_single_fig(df_single, max_components=30)
    return fig_tot, fig_inv, fig_comp

@app.callback(
    Output("var-cost-compare-graph", "figure"),
    Output("var-capacity-compare-graph", "figure"),
    Output("var-storage-capacity-compare-graph", "figure"),
    Input("datafile-dropdown", "value"),
    Input("var-compare-dropdown", "value"),
    Input("var-year-dropdown", "value"),
)
def update_variant_compare_tab(base_path, cmp_path, year_value):
    """
    Dash-Callback: erstellt die drei Vergleichsplots (Kosten, Leistungen, Speicherkapazitäten)
    zwischen Base- und Vergleichsvariante
    
    Inputs: base_path: str
            cmp_path: str
            year_value: str|None
            
    Initialisiert leere Platzhalterfiguren (verhindert UnboundLocalError)
    Lädt States für base und compare; validiert ok-Flags
    Leitet period_cost/period_cap abhängig von MIP ab
    Erzeugt Vergleichsfiguren via build_variant_cost_compare_fig,
    build_variant_capacity_compare_fig, build_variant_storage_capacity_compare_fig
    
    Outputs: aktualisierte (fig_cost, fig_cap, fig_store): go.Figure
    """
    # Fallbacks für den Fall, dass keine Variante gewählt ist
    f_cost_empty = go.Figure().update_layout(title="Gesamtkostenvergleich (keine Vergleichsvariante gewählt)")
    f_cap_empty  = go.Figure().update_layout(title="Nennleistungen (keine Vergleichsvariante gewählt)")
    f_store_empty = go.Figure().update_layout(title="Speicherkapazität (keine Vergleichsvariante gewählt)")

    if base_path is None or cmp_path is None:
        return f_cost_empty, f_cap_empty, f_store_empty

    st_base = get_dataset_state(base_path) if base_path else _empty_state("keine Datei")
    st_cmp  = get_dataset_state(cmp_path)  if cmp_path  else _empty_state("keine Vergleichsdatei")

    if (not st_base.get("ok", False)) or (not st_cmp.get("ok", False)):
        return f_cost_empty, f_cap_empty, f_store_empty

    base_name = _basename(base_path)
    cmp_name  = _basename(cmp_path)

    # Periodenlogik
    if st_base.get("has_mip", False) and st_base.get("years", []):
        period_cost = str(year_value) if year_value is not None else str(min(st_base["years"]))
        period_cap  = period_cost
    else:
        period_cost = "Single"
        period_cap  = None

    fig_cost = build_variant_cost_compare_fig(
        st_base=st_base,
        st_cmp=st_cmp,
        base_name=base_name,
        cmp_name=cmp_name,
        period_cost=period_cost,
    )

    fig_cap = build_variant_capacity_compare_fig(
        st_base=st_base,
        st_cmp=st_cmp,
        base_name=base_name,
        cmp_name=cmp_name,
        period_value=period_cap,
        top_n=30,
    )

    fig_store = build_variant_storage_capacity_compare_fig(
        st_base=st_base,
        st_cmp=st_cmp,
        base_name=base_name,
        cmp_name=cmp_name,
        period_value=period_cap,
        top_n=30,
    )

    return fig_cost, fig_cap, fig_store

#%% RUN

if __name__ == "__main__":
    if not file_options:
        print(f"[WARN] Keine .nc-Dateien in DATA_DIR gefunden: {DATA_DIR}")
    app.run(debug=False, use_reloader=True, threaded = False)

#%%debug
