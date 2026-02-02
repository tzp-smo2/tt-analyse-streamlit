import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="🏓 Analyse TT", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
ZONE_COLS = ["CD", "Corps", "Revers"]
ZONE_ROWS = ["Court", "Long"]

def normalize_text(s):
    if s is None:
        return ""
    return str(s).strip()

def normalize_lower(s):
    return normalize_text(s).lower()

def count_table(df, index=None, columns=None):
    if df.empty:
        return pd.DataFrame()
    tmp = df.copy()
    tmp["_row_"] = 1
    pt = pd.pivot_table(tmp, index=index, columns=columns, values="_row_", aggfunc="sum", fill_value=0)
    return pt

def safe_str_series(df, col):
    if col not in df.columns:
        return pd.Series([""]*len(df))
    return df[col].astype(str).str.strip()

def zone_to_cell(zone: str):
    z = normalize_lower(zone)
    if not z:
        return None
    if "filet" in z or "hors" in z:
        return None

    if "court" in z:
        row = "Court"
    elif "long" in z:
        row = "Long"
    else:
        row = "Long"

    if "cd" in z:
        col = "CD"
    elif "revers" in z or "rv" in z:
        col = "Revers"
    elif "corps" in z or "milieu" in z:
        col = "Corps"
    else:
        return None

    return (ZONE_ROWS.index(row), ZONE_COLS.index(col))

def compute_heatmap(df, zone_col):
    mat = np.zeros((len(ZONE_ROWS), len(ZONE_COLS)), dtype=int)
    if zone_col not in df.columns:
        return mat
    for z in df[zone_col].dropna().tolist():
        cell = zone_to_cell(z)
        if cell is not None:
            r, c = cell
            mat[r, c] += 1
    return mat

def plot_heatmap(mat, title):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(mat)
    ax.set_xticks(range(len(ZONE_COLS)))
    ax.set_xticklabels(ZONE_COLS)
    ax.set_yticks(range(len(ZONE_ROWS)))
    ax.set_yticklabels(ZONE_ROWS)
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig

def coaching_lines(best_geste, best_g, best_f, worst_geste, worst_g, worst_f, main_win_zone, main_err_zone):
    lines = []
    lines.append(f"✅ **Point fort :** ton geste le plus rentable est **{best_geste}** (gagnants {best_g} / fautes {best_f}).")
    lines.append(f"⚠️ **Point à améliorer :** ton geste le plus coûteux est **{worst_geste}** (gagnants {worst_g} / fautes {worst_f}).")
    if main_win_zone:
        lines.append(f"🎯 **Zone qui marche :** tu marques souvent vers **{main_win_zone}**.")
    if main_err_zone:
        lines.append(f"🧱 **Zone d’erreur :** beaucoup de fautes finissent en **{main_err_zone}**.")
    lines.append("🏁 **Objectif prochain match :** garde ton point fort, et sécurise le geste coûteux (trajet plus haut / plus long).")
    return lines

# -----------------------------
# UI
# -----------------------------
st.title("🏓 Analyse automatique d’un match (Excel)")

st.markdown(
    "Charge ton fichier `.xlsx` (1 ligne = 1 point), et l’app génère automatiquement les tableaux et graphiques.\n"
    "👉 L’app propose un **mapping de colonnes** si tes noms changent."
)

uploaded = st.file_uploader("Importer le fichier Excel (.xlsx)", type=["xlsx"])

with st.sidebar:
    st.header("Paramètres")
    player = st.text_input("Nom du joueur (tel qu’écrit dans le fichier)", value="Théophile")

if uploaded:
    try:
        df_raw = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Impossible de lire le fichier Excel : {e}")
        st.stop()

    df_raw.columns = [str(c).strip() for c in df_raw.columns]
    cols = df_raw.columns.tolist()

    st.subheader("1) Mapping des colonnes")
    st.write("Sélectionne quelle colonne correspond à quel champ (si besoin).")

    def pick_default(candidates):
        for c in candidates:
            if c in cols:
                return c
        return cols[0] if cols else ""

    suggestions = {
        "point_pour": ["point_pour", "Point_pour", "Point pour", "Vainqueur_point"],
        "issue_point": ["issue_point", "Issue_point", "Issue", "Type_point"],
        "Serveur": ["Serveur", "serveur", "server"],
        "geste_technique": ["geste_technique", "Geste", "geste", "coup", "Coup"],
        "Zone_table": ["Zone_table", "Zone", "zone", "placement", "Placement"],
        "auteur_faute": ["auteur_faute", "Auteur_faute", "Faute_par"],
        "manche": ["manche", "Manche", "set", "Set"],
        "effet": ["effet", "Effet", "spin", "Spin"],
    }

    c1, c2 = st.columns(2)
    with c1:
        c_point_pour = st.selectbox("Colonne : point_pour", options=cols, index=cols.index(pick_default(suggestions["point_pour"])) if cols else 0)
        c_issue = st.selectbox("Colonne : issue_point", options=cols, index=cols.index(pick_default(suggestions["issue_point"])) if cols else 0)
        c_serveur = st.selectbox("Colonne : Serveur", options=cols, index=cols.index(pick_default(suggestions["Serveur"])) if cols else 0)
        c_geste = st.selectbox("Colonne : geste_technique", options=cols, index=cols.index(pick_default(suggestions["geste_technique"])) if cols else 0)
    with c2:
        c_zone = st.selectbox("Colonne : Zone_table", options=cols, index=cols.index(pick_default(suggestions["Zone_table"])) if cols else 0)
        c_auteur_faute = st.selectbox("Colonne : auteur_faute (optionnel)", options=["(absent)"] + cols, index=0)
        c_manche = st.selectbox("Colonne : manche (optionnel)", options=["(absent)"] + cols, index=0)
        c_effet = st.selectbox("Colonne : effet (optionnel)", options=["(absent)"] + cols, index=0)

    df = pd.DataFrame()
    df["point_pour"] = safe_str_series(df_raw, c_point_pour)
    df["issue_point"] = safe_str_series(df_raw, c_issue)
    df["Serveur"] = safe_str_series(df_raw, c_serveur)
    df["geste_technique"] = safe_str_series(df_raw, c_geste)
    df["Zone_table"] = safe_str_series(df_raw, c_zone)

    if c_auteur_faute != "(absent)":
        df["auteur_faute"] = safe_str_series(df_raw, c_auteur_faute)
    if c_manche != "(absent)":
        df["manche"] = safe_str_series(df_raw, c_manche)
    if c_effet != "(absent)":
        df["effet"] = safe_str_series(df_raw, c_effet)

    df["issue_norm"] = df["issue_point"].str.lower().str.strip()
    df["point_norm"] = df["point_pour"].str.strip()
    df["serveur_norm"] = df["Serveur"].str.strip()
    df["geste_norm"] = df["geste_technique"].str.strip()
    df["zone_norm"] = df["Zone_table"].str.strip()

    if "manche" in df.columns:
        manches = sorted([m for m in df["manche"].dropna().unique().tolist() if m != ""])
        sel = st.selectbox("Filtrer sur une manche", options=["Toutes"] + manches)
        if sel != "Toutes":
            dfv = df[df["manche"] == sel].copy()
        else:
            dfv = df.copy()
    else:
        dfv = df.copy()

    st.subheader("Aperçu des données")
    st.dataframe(dfv, use_container_width=True, height=240)

    gagnants = dfv[(dfv["point_norm"] == player) & (dfv["issue_norm"] == "gagnant")]

    if "auteur_faute" in dfv.columns:
        fautes_joueur = dfv[(dfv["issue_norm"] == "faute") & (dfv["auteur_faute"].str.strip() == player)]
    else:
        fautes_joueur = dfv[(dfv["issue_norm"] == "faute") & (dfv["point_norm"] != player)]

    st.header("TCD 1 — Résultat global")
    tcd1 = count_table(dfv, index="point_norm")
    st.dataframe(tcd1, use_container_width=True)

    st.header("TCD 2 — Coups efficaces / coups risqués")
    tcd2a = count_table(gagnants, index="geste_norm")
    if not tcd2a.empty:
        tcd2a.columns = ["gagnants"]
    tcd2b = count_table(fautes_joueur, index="geste_norm")
    if not tcd2b.empty:
        tcd2b.columns = ["fautes_joueur"]

    left, right = st.columns(2)
    with left:
        st.markdown("**Gagnants (Théophile)**")
        st.dataframe(tcd2a.sort_values(by="gagnants", ascending=False) if not tcd2a.empty else tcd2a, use_container_width=True)
    with right:
        st.markdown("**Fautes (Théophile)**")
        st.dataframe(tcd2b.sort_values(by="fautes_joueur", ascending=False) if not tcd2b.empty else tcd2b, use_container_width=True)

    merged = pd.DataFrame(index=sorted(set(tcd2a.index).union(set(tcd2b.index))))
    merged["Gagnants"] = tcd2a["gagnants"] if (not tcd2a.empty and "gagnants" in tcd2a.columns) else 0
    merged["Fautes Théophile"] = tcd2b["fautes_joueur"] if (not tcd2b.empty and "fautes_joueur" in tcd2b.columns) else 0
    merged = merged.fillna(0)

    st.markdown("**Graphique : gagnants vs fautes par geste**")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    merged.plot(kind="bar", ax=ax)
    ax.set_xlabel("Geste")
    ax.set_ylabel("Nombre de points")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

    st.header("TCD 3 — Où Théophile fait ses fautes")
    tcd3 = count_table(fautes_joueur, index="zone_norm")
    if not tcd3.empty:
        tcd3.columns = ["fautes_joueur"]
        st.dataframe(tcd3.sort_values(by="fautes_joueur", ascending=False), use_container_width=True)
    else:
        st.info("Aucune faute détectée pour ce filtre.")

    st.header("TCD 4 — Service / Retour")
    tcd4 = count_table(dfv, index="serveur_norm", columns="point_norm")
    st.dataframe(tcd4, use_container_width=True)

    st.header("TCD 5 — Zones qui font marquer Théophile")
    tcd5 = count_table(gagnants, index="zone_norm")
    if not tcd5.empty:
        tcd5.columns = ["gagnants"]
        st.dataframe(tcd5.sort_values(by="gagnants", ascending=False), use_container_width=True)
    else:
        st.info("Aucun gagnant détecté pour ce filtre.")

    st.header("🗺️ Heatmaps des zones")
    cH1, cH2 = st.columns(2)
    mat_win = compute_heatmap(gagnants, zone_col="Zone_table")
    mat_err = compute_heatmap(fautes_joueur, zone_col="Zone_table")
    with cH1:
        st.pyplot(plot_heatmap(mat_win, "Zones où Théophile marque (Gagnants)"), clear_figure=True)
    with cH2:
        st.pyplot(plot_heatmap(mat_err, "Zones associées aux fautes de Théophile"), clear_figure=True)

    st.header("🎯 Bilan coaching automatique (3 constats + 1 objectif)")
    # compute best/worst gesture by (gagnants - fautes)
    g_counts = merged["Gagnants"].to_dict()
    f_counts = merged["Fautes Théophile"].to_dict()
    score = {k: g_counts.get(k, 0) - f_counts.get(k, 0) for k in merged.index.tolist()} if len(merged) else {}

    if score:
        best_geste = max(score, key=lambda k: score[k])
        worst_geste = min(score, key=lambda k: score[k])
        best_g = int(g_counts.get(best_geste, 0))
        best_f = int(f_counts.get(best_geste, 0))
        worst_g = int(g_counts.get(worst_geste, 0))
        worst_f = int(f_counts.get(worst_geste, 0))
    else:
        best_geste, worst_geste = "N/A", "N/A"
        best_g = best_f = worst_g = worst_f = 0

    main_win_zone = tcd5.sort_values(by="gagnants", ascending=False).index[0] if (not tcd5.empty and "gagnants" in tcd5.columns) else None
    main_err_zone = tcd3.sort_values(by="fautes_joueur", ascending=False).index[0] if (not tcd3.empty and "fautes_joueur" in tcd3.columns) else None

    for line in coaching_lines(best_geste, best_g, best_f, worst_geste, worst_g, worst_f, main_win_zone, main_err_zone):
        st.write(line)

else:
    st.info("Importe un fichier Excel pour lancer l’analyse.")
