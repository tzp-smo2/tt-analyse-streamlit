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
        return pd.Series([""] * len(df))
    return df[col].astype(str).str.strip()

def zone_to_cell(zone: str):
    """
    Convertit une étiquette de zone (ex: 'long_CD', 'court_revers', 'corps')
    vers une cellule (row, col) dans une grille 2x3.
    Retourne None si la zone est 'filet' ou 'hors_table' (on les exclut des heatmaps de placement).
    """
    z = normalize_lower(zone)
    if not z:
        return None

    # Exclure erreurs hors-table / filet
    if "filet" in z or "hors" in z:
        return None

    # Row : Court / Long
    if "court" in z:
        row = "Court"
    elif "long" in z:
        row = "Long"
    else:
        # si pas précisé, on met Long par défaut
        row = "Long"

    # Col : CD / Corps / Revers
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
    """
    Retourne une matrice 2x3 de comptage des occurrences.
    Rows: Court, Long
    Cols: CD, Corps, Revers
    """
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
    """
    Affichage 'côté adversaire' :
    - filet en bas => Court en bas, Long en haut
    - CD adversaire à gauche, Revers adversaire à droite
    Donc : long_CD = haut gauche ; court_CD = bas gauche
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # mat colonnes = [CD, Corps, Revers] => CD à gauche, Revers à droite (OK)
    mat_display = mat

    # origin="lower" => Court (ligne 0) affiché en bas, Long (ligne 1) en haut
    im = ax.imshow(mat_display, origin="lower")

    ax.set_xticks(range(3))
    ax.set_xticklabels(["CD (adv)", "Corps", "Revers (adv)"])
    ax.set_yticks(range(2))
    ax.set_yticklabels(["Court", "Long"])

    ax.set_title(title)

    # annotations
    for i in range(mat_display.shape[0]):
        for j in range(mat_display.shape[1]):
            ax.text(j, i, str(mat_display[i, j]), ha="center", va="center")

    ax.set_xlabel("Gauche : CD adversaire     |     Droite : Revers adversaire")
    ax.set_ylabel("Filet ↓                 Profond ↑")

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
    st.caption("Astuce : si ton fichier écrit 'théophile' en minuscule, pas de souci : l’app est maintenant tolérante.")

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
        c_point_pour = st.selectbox("Colonne : point_pour", options=cols,
                                    index=cols.index(pick_default(suggestions["point_pour"])) if cols else 0)
        c_issue = st.selectbox("Colonne : issue_point", options=cols,
                               index=cols.index(pick_default(suggestions["issue_point"])) if cols else 0)
        c_serveur = st.selectbox("Colonne : Serveur", options=cols,
                                 index=cols.index(pick_default(suggestions["Serveur"])) if cols else 0)
        c_geste = st.selectbox("Colonne : geste_technique", options=cols,
                               index=cols.index(pick_default(suggestions["geste_technique"])) if cols else 0)
    with c2:
        c_zone = st.selectbox("Colonne : Zone_table", options=cols,
                              index=cols.index(pick_default(suggestions["Zone_table"])) if cols else 0)
        c_auteur_faute = st.selectbox("Colonne : auteur_faute (optionnel)", options=["(absent)"] + cols, index=0)
        c_manche = st.selectbox("Colonne : manche (optionnel)", options=["(absent)"] + cols, index=0)
        c_effet = st.selectbox("Colonne : effet (optionnel)", options=["(absent)"] + cols, index=0)

    # Normalized dataframe
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

    # Normalized helper columns
    df["issue_norm"] = df["issue_point"].astype(str).str.strip().str.lower()
    df["point_norm"] = df["point_pour"].astype(str).str.strip()
    df["serveur_norm"] = df["Serveur"].astype(str).str.strip()
    df["geste_norm"] = df["geste_technique"].astype(str).str.strip()
    df["zone_norm"] = df["Zone_table"].astype(str).str.strip()

    # Manche filter
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

    # --- Robust filters (case-insensitive)
    player_norm = player.strip().lower()

    point_l = dfv["point_norm"].astype(str).str.strip().str.lower()
    issue_l = dfv["issue_norm"].astype(str).str.strip().str.lower()

    gagnants = dfv[(point_l == player_norm) & (issue_l == "gagnant")]

    if "auteur_faute" in dfv.columns:
        auteur_l = dfv["auteur_faute"].astype(str).str.strip().str.lower()
        fautes_joueur = dfv[(issue_l == "faute") & (auteur_l == player_norm)]
    else:
        fautes_joueur = dfv[(issue_l == "faute") & (point_l != player_norm)]

    # -----------------------------
    # TCD 1
    # -----------------------------
    st.header("TCD 1 — Résultat global")
    tcd1 = count_table(dfv.assign(point_norm_l=point_l), index="point_norm_l")
    st.dataframe(tcd1, use_container_width=True)

    # -----------------------------
    # TCD 2 (geste gagnants / fautes)
    # -----------------------------
    st.header("TCD 2 — Coups efficaces / coups risqués")
    tcd2a = count_table(gagnants, index="geste_norm")
    if not tcd2a.empty:
        tcd2a.columns = ["gagnants"]
    tcd2b = count_table(fautes_joueur, index="geste_norm")
    if not tcd2b.empty:
        tcd2b.columns = ["fautes_joueur"]

    left, right = st.columns(2)
    with left:
        st.markdown("**Gagnants (joueur)**")
        st.dataframe(tcd2a.sort_values(by="gagnants", ascending=False) if not tcd2a.empty else tcd2a,
                     use_container_width=True)
    with right:
        st.markdown("**Fautes (joueur)**")
        st.dataframe(tcd2b.sort_values(by="fautes_joueur", ascending=False) if not tcd2b.empty else tcd2b,
                     use_container_width=True)

    merged = pd.DataFrame(index=sorted(set(tcd2a.index).union(set(tcd2b.index))))
    merged["Gagnants"] = tcd2a["gagnants"] if (not tcd2a.empty and "gagnants" in tcd2a.columns) else 0
    merged["Fautes joueur"] = tcd2b["fautes_joueur"] if (not tcd2b.empty and "fautes_joueur" in tcd2b.columns) else 0
    merged = merged.fillna(0)

    st.markdown("**Graphique : gagnants vs fautes par geste**")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    merged.plot(kind="bar", ax=ax)
    ax.set_xlabel("Geste")
    ax.set_ylabel("Nombre de points")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

    # -----------------------------
    # TCD 3 (zones fautes joueur)
    # -----------------------------
    st.header("TCD 3 — Où le joueur fait ses fautes")
    tcd3 = count_table(fautes_joueur, index="zone_norm")
    if not tcd3.empty:
        tcd3.columns = ["fautes_joueur"]
        st.dataframe(tcd3.sort_values(by="fautes_joueur", ascending=False), use_container_width=True)
    else:
        st.info("Aucune faute détectée pour ce filtre.")

    # -----------------------------
    # TCD 4 (service / retour)
    # -----------------------------
    st.header("TCD 4 — Service / Retour")
    tcd4 = count_table(dfv.assign(point_norm_l=point_l), index="serveur_norm", columns="point_norm_l")
    st.dataframe(tcd4, use_container_width=True)

    # -----------------------------
    # TCD 5 (zones gagnantes)
    # -----------------------------
    st.header("TCD 5 — Zones qui font marquer le joueur")
    tcd5 = count_table(gagnants, index="zone_norm")
    if not tcd5.empty:
        tcd5.columns = ["gagnants"]
        st.dataframe(tcd5.sort_values(by="gagnants", ascending=False), use_container_width=True)
    else:
        st.info("Aucun gagnant détecté pour ce filtre.")

    # -----------------------------
    # Heatmaps
    # -----------------------------
    st.header("🗺️ Heatmaps des zones")
    cH1, cH2 = st.columns(2)

    mat_win = compute_heatmap(gagnants, zone_col="Zone_table")
    mat_err = compute_heatmap(fautes_joueur, zone_col="Zone_table")

    with cH1:
        st.pyplot(plot_heatmap(mat_win, "Zones où le joueur marque (Gagnants)"), clear_figure=True)
    with cH2:
        st.pyplot(plot_heatmap(mat_err, "Zones associées aux fautes du joueur"), clear_figure=True)

    # -----------------------------
    # Coaching summary
    # -----------------------------
    st.header("🎯 Bilan coaching automatique (3 constats + 1 objectif)")
    g_counts = merged["Gagnants"].to_dict()
    f_counts = merged["Fautes joueur"].to_dict()
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

    main_win_zone = (
        tcd5.sort_values(by="gagnants", ascending=False).index[0]
        if (not tcd5.empty and "gagnants" in tcd5.columns) else None
    )
    main_err_zone = (
        tcd3.sort_values(by="fautes_joueur", ascending=False).index[0]
        if (not tcd3.empty and "fautes_joueur" in tcd3.columns) else None
    )

    for line in coaching_lines(best_geste, best_g, best_f, worst_geste, worst_g, worst_f, main_win_zone, main_err_zone):
        st.write(line)

else:
    st.info("Importe un fichier Excel pour lancer l’analyse.")
