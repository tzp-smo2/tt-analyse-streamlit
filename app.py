import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="🏓 Analyse TT", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
def center_df(df):
    """Centre horizontalement et verticalement le contenu d'un DataFrame."""
    if df.empty:
        return df
    return df.style.set_properties(
        **{
            "text-align": "center",
            "vertical-align": "middle"
        }
    )
ZONE_COLS = ["CD", "Corps", "Revers"]
ZONE_ROWS = ["Court", "Long"]

def normalize_text(s):
    if s is None:
        return ""
    return str(s).strip()

def normalize_lower(s):
    return normalize_text(s).lower()

def safe_str_series(df, col):
    if col not in df.columns:
        return pd.Series([""] * len(df))
    return df[col].astype(str).str.strip()

def count_table(df, index=None, columns=None):
    """Pivot count rows."""
    if df.empty:
        return pd.DataFrame()
    tmp = df.copy()
    tmp["_row_"] = 1
    pt = pd.pivot_table(tmp, index=index, columns=columns, values="_row_", aggfunc="sum", fill_value=0)
    return pt

def zone_to_cell(zone: str):
    """
    Convertit une étiquette de zone (ex: 'long_CD', 'court_revers', 'corps')
    vers une cellule (row, col) dans une grille 2x3.
    Exclut 'filet' / 'hors_table' des heatmaps de placement.
    """
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

def compute_heatmap(df, zone_col="Zone_table"):
    mat = np.zeros((len(ZONE_ROWS), len(ZONE_COLS)), dtype=int)
    if zone_col not in df.columns or df.empty:
        return mat
    for z in df[zone_col].dropna().tolist():
        cell = zone_to_cell(z)
        if cell is not None:
            r, c = cell
            mat[r, c] += 1
    return mat

def plot_heatmap_side_opponent(mat, title):
    """
    Vue 'côté adversaire' :
    - Filet en bas -> Court en bas, Long en haut
    - CD adversaire à gauche, Revers adversaire à droite
    => long_CD = haut gauche ; court_CD = bas gauche
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # colonnes = [CD, Corps, Revers] => CD à gauche, Revers à droite (OK)
    mat_display = mat
    im = ax.imshow(mat_display, origin="lower")

    ax.set_xticks(range(3))
    ax.set_xticklabels(["CD (adv)", "Corps", "Revers (adv)"])
    ax.set_yticks(range(2))
    ax.set_yticklabels(["Court", "Long"])
    ax.set_title(title)

    for i in range(mat_display.shape[0]):
        for j in range(mat_display.shape[1]):
            ax.text(j, i, str(mat_display[i, j]), ha="center", va="center")

    ax.set_xlabel("Gauche = CD adversaire   |   Droite = Revers adversaire")
    ax.set_ylabel("Filet ↓               Profond ↑")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig

def bar_from_series(series, xlabel, ylabel, title):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    series.plot(kind="bar", ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()
    return fig

def stacked_bar_from_df(df, xlabel, ylabel, title):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    df.plot(kind="bar", ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig

def top_combos_from_pivot(pivot_df, top_n=10):
    """Transforme une matrice Geste x Zone en liste triée de combos."""
    if pivot_df.empty:
        return pd.Series(dtype=int)
    s = pivot_df.stack().sort_values(ascending=False)
    s = s[s > 0].head(top_n)
    s.index = [f"{idx[0]} → {idx[1]}" for idx in s.index]
    return s

# -----------------------------
# UI
# -----------------------------
st.title("🏓 Analyse automatique d’un match (Excel)")

uploaded = st.file_uploader("Importer le fichier Excel (.xlsx)", type=["xlsx"])

with st.sidebar:
    st.header("Paramètres")
    player = st.text_input("Nom du joueur (tel qu’écrit dans le fichier)", value="Théophile")
    st.caption("La comparaison est insensible à la casse : 'Théophile' = 'théophile'.")

if not uploaded:
    st.info("Importe un fichier Excel pour lancer l’analyse.")
    st.stop()

# --- Load
try:
    df_raw = pd.read_excel(uploaded)
except Exception as e:
    st.error(f"Impossible de lire le fichier Excel : {e}")
    st.stop()

df_raw.columns = [str(c).strip() for c in df_raw.columns]
cols = df_raw.columns.tolist()

# --- Mapping
st.subheader("Mapping des colonnes")
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
    "num_point": ["num_point", "Num_point", "Point", "point"],
}

m1, m2 = st.columns(2)
with m1:
    c_point_pour = st.selectbox("Colonne : point_pour", options=cols,
                                index=cols.index(pick_default(suggestions["point_pour"])) if cols else 0)
    c_issue = st.selectbox("Colonne : issue_point", options=cols,
                           index=cols.index(pick_default(suggestions["issue_point"])) if cols else 0)
    c_serveur = st.selectbox("Colonne : Serveur", options=cols,
                             index=cols.index(pick_default(suggestions["Serveur"])) if cols else 0)
    c_geste = st.selectbox("Colonne : geste_technique", options=cols,
                           index=cols.index(pick_default(suggestions["geste_technique"])) if cols else 0)

with m2:
    c_zone = st.selectbox("Colonne : Zone_table", options=cols,
                          index=cols.index(pick_default(suggestions["Zone_table"])) if cols else 0)
    c_auteur_faute = st.selectbox("Colonne : auteur_faute (optionnel)", options=["(absent)"] + cols, index=0)
    c_manche = st.selectbox("Colonne : manche (optionnel)", options=["(absent)"] + cols, index=0)
    c_effet = st.selectbox("Colonne : effet (optionnel)", options=["(absent)"] + cols, index=0)

# --- Normalized DF
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

# helper columns
df["issue_norm"] = df["issue_point"].astype(str).str.strip().str.lower()
df["point_norm"] = df["point_pour"].astype(str).str.strip()
df["serveur_norm"] = df["Serveur"].astype(str).str.strip()
df["geste_norm"] = df["geste_technique"].astype(str).str.strip()
df["zone_norm"] = df["Zone_table"].astype(str).str.strip()

# manche filter
if "manche" in df.columns:
    manches = sorted([m for m in df["manche"].dropna().unique().tolist() if m != ""])
    sel = st.selectbox("Filtrer sur une manche", options=["Toutes"] + manches)
    dfv = df[df["manche"] == sel].copy() if sel != "Toutes" else df.copy()
else:
    dfv = df.copy()

# robust matching (case-insensitive)
player_norm = player.strip().lower()
point_l = dfv["point_norm"].astype(str).str.strip().str.lower()
issue_l = dfv["issue_norm"].astype(str).str.strip().str.lower()
serveur_l = dfv["serveur_norm"].astype(str).str.strip().str.lower()

# filters
gagnants = dfv[(point_l == player_norm) & (issue_l == "gagnant")]
if "auteur_faute" in dfv.columns:
    auteur_l = dfv["auteur_faute"].astype(str).str.strip().str.lower()
    fautes_joueur = dfv[(issue_l == "faute") & (auteur_l == player_norm)]
else:
    # fallback
    fautes_joueur = dfv[(issue_l == "faute") & (point_l != player_norm)]

# -----------------------------
# Tabs
# -----------------------------
tab_bilan, tab_coach, tab_data = st.tabs(["👦 Bilan Théophile", "🧠 Analyse coach (avancée)", "📄 Données"])

# -----------------------------
# DATA tab
# -----------------------------
with tab_data:
    st.subheader("Aperçu des données normalisées (filtrées)")
    st.dataframe(dfv, use_container_width=True, height=320)

# -----------------------------
# BILAN tab (simple)
# -----------------------------
with tab_bilan:
    st.subheader("Ce que tu peux expliquer facilement à Théophile")

    # Heatmaps
    colH1, colH2 = st.columns(2)
    mat_win = compute_heatmap(gagnants, zone_col="Zone_table")
    mat_err = compute_heatmap(fautes_joueur, zone_col="Zone_table")

    with colH1:
        st.pyplot(plot_heatmap_side_opponent(mat_win, "Où tu marques (zones gagnantes)"), clear_figure=True)
    with colH2:
        st.pyplot(plot_heatmap_side_opponent(mat_err, "Où tu fais des fautes (zones à risque)"), clear_figure=True)

    # Gesture efficiency graph (gagnants vs fautes)
    st.markdown("### Coups : gagnants vs fautes")
    t_g = count_table(gagnants, index="geste_norm")
    if not t_g.empty:
        t_g.columns = ["Gagnants"]
    t_f = count_table(fautes_joueur, index="geste_norm")
    if not t_f.empty:
        t_f.columns = ["Fautes"]

    merged = pd.DataFrame(index=sorted(set(t_g.index).union(set(t_f.index))))
    merged["Gagnants"] = t_g["Gagnants"] if (not t_g.empty and "Gagnants" in t_g.columns) else 0
    merged["Fautes"] = t_f["Fautes"] if (not t_f.empty and "Fautes" in t_f.columns) else 0
    merged = merged.fillna(0)

    st.pyplot(stacked_bar_from_df(merged, "Geste", "Nb points", "Gagnants vs Fautes par geste"), clear_figure=True)

    # 3+1 coaching lines
    st.markdown("### Bilan (3 constats + 1 objectif)")
    g_counts = merged["Gagnants"].to_dict()
    f_counts = merged["Fautes"].to_dict()
    score = {k: g_counts.get(k, 0) - f_counts.get(k, 0) for k in merged.index.tolist()} if len(merged) else {}

    if score:
        best_geste = max(score, key=lambda k: score[k])
        worst_geste = min(score, key=lambda k: score[k])
        best_g = int(g_counts.get(best_geste, 0))
        best_f = int(f_counts.get(best_geste, 0))
        worst_g = int(g_counts.get(worst_geste, 0))
        worst_f = int(f_counts.get(worst_geste, 0))
    else:
        best_geste = worst_geste = "N/A"
        best_g = best_f = worst_g = worst_f = 0

    # main zones
    t_z_win = count_table(gagnants, index="zone_norm")
    t_z_err = count_table(fautes_joueur, index="zone_norm")
    main_win_zone = t_z_win.sort_values(by=t_z_win.columns[0], ascending=False).index[0] if not t_z_win.empty else None
    main_err_zone = t_z_err.sort_values(by=t_z_err.columns[0], ascending=False).index[0] if not t_z_err.empty else None

    st.write(f"✅ **Point fort :** ton geste le plus rentable est **{best_geste}** (gagnants {best_g} / fautes {best_f}).")
    st.write(f"⚠️ **Point à améliorer :** ton geste le plus coûteux est **{worst_geste}** (gagnants {worst_g} / fautes {worst_f}).")
    if main_win_zone:
        st.write(f"🎯 **Zone qui marche :** tu marques souvent vers **{main_win_zone}**.")
    if main_err_zone:
        st.write(f"🧱 **Zone à risque :** beaucoup de fautes finissent en **{main_err_zone}**.")
    st.write("🏁 **Objectif prochain match :** garde ton point fort, et sécurise le geste coûteux (plus haut / plus long).")

# -----------------------------
# COACH tab (advanced)
# -----------------------------
with tab_coach:
    st.subheader("Analyse approfondie (pour toi)")

    # -----------------------------
    # TCD 1 — Résultat global
    # -----------------------------
    st.markdown("## TCD 1 — Résultat global")
    tcd1 = count_table(dfv.assign(point_l=point_l), index="point_l")
    st.dataframe(tcd1, use_container_width=True)

    # -----------------------------
    # TCD 4 — Service/Retour (global)
    # -----------------------------
    st.markdown("## TCD 4 — Service / Retour (global)")
    tcd4 = count_table(dfv.assign(point_l=point_l, srv_l=serveur_l), index="srv_l", columns="point_l")
    st.dataframe(tcd4, use_container_width=True)

    # -----------------------------
    # TCD 5 — Zones gagnantes
    # -----------------------------
    st.markdown("## TCD 5 — Zones gagnantes (joueur)")
    tcd5 = count_table(gagnants, index="zone_norm")
    if not tcd5.empty:
        tcd5.columns = ["Gagnants"]
        st.dataframe(tcd5.sort_values(by="Gagnants", ascending=False), use_container_width=True)
        st.pyplot(bar_from_series(tcd5["Gagnants"].sort_values(ascending=False), "Zone", "Nb gagnants", "Gagnants par zone"), clear_figure=True)
    else:
        st.info("Aucun gagnant détecté pour ce filtre.")

    # -----------------------------
    # TCD 3 — Zones fautes joueur
    # -----------------------------
    st.markdown("## TCD 3 — Zones des fautes (joueur)")
    tcd3 = count_table(fautes_joueur, index="zone_norm")
    if not tcd3.empty:
        tcd3.columns = ["Fautes"]
        st.dataframe(tcd3.sort_values(by="Fautes", ascending=False), use_container_width=True)
        st.pyplot(bar_from_series(tcd3["Fautes"].sort_values(ascending=False), "Zone", "Nb fautes", "Fautes par zone"), clear_figure=True)
    else:
        st.info("Aucune faute détectée pour ce filtre.")

    # -----------------------------
    # TCD 6 — Geste × Zone (gagnants / fautes)
    # -----------------------------
    st.markdown("## TCD 6 — Geste × Zone (gagnants / fautes)")
    col6a, col6b = st.columns(2)

    tcd6a = count_table(gagnants, index="geste_norm", columns="zone_norm")
    tcd6b = count_table(fautes_joueur, index="geste_norm", columns="zone_norm")

    with col6a:
        st.markdown("### 6A — Gagnants : Geste × Zone")
        st.dataframe(tcd6a, use_container_width=True)
        topA = top_combos_from_pivot(tcd6a, top_n=10)
        if not topA.empty:
            st.pyplot(bar_from_series(topA, "Combo", "Nb gagnants", "Top 10 combos gagnants"), clear_figure=True)

    with col6b:
        st.markdown("### 6B — Fautes : Geste × Zone")
        st.dataframe(tcd6b, use_container_width=True)
        topB = top_combos_from_pivot(tcd6b, top_n=10)
        if not topB.empty:
            st.pyplot(bar_from_series(topB, "Combo", "Nb fautes", "Top 10 combos fautes"), clear_figure=True)

    # -----------------------------
    # TCD 7 — Zones jouées (tous points)
    # -----------------------------
    st.markdown("## TCD 7 — Zones jouées (tous points, indépendamment du résultat)")
    tcd7 = count_table(dfv, index="zone_norm")
    if not tcd7.empty:
        # tcd7 a une seule colonne "_row_" (nom dépend du pivot). On récupère la 1ère.
        colname = tcd7.columns[0]
        tcd7 = tcd7.sort_values(by=colname, ascending=False)
        st.dataframe(tcd7, use_container_width=True)
        st.pyplot(bar_from_series(tcd7[colname], "Zone", "Nb points", "Fréquence de jeu par zone"), clear_figure=True)

    # -----------------------------
    # TCD 8 — Profil des fautes (filet / hors table / autre) + geste
    # -----------------------------
    st.markdown("## TCD 8 — Profil des fautes (filet / hors table / autre) + geste")
    err_df = fautes_joueur.copy()
    if not err_df.empty:
        err_df["err_type"] = err_df["zone_norm"].str.lower().apply(
            lambda z: "filet" if "filet" in z else ("hors_table" if "hors" in z else "autre_zone")
        )

        tcd8a = count_table(err_df, index="err_type")
        tcd8b = count_table(err_df, index="geste_norm", columns="err_type")

        c8a, c8b = st.columns(2)
        with c8a:
            st.markdown("### 8A — Répartition des fautes")
            st.dataframe(tcd8a, use_container_width=True)
            s = tcd8a[tcd8a.columns[0]].sort_values(ascending=False)
            st.pyplot(bar_from_series(s, "Type", "Nb fautes", "Filet vs Hors table vs Autres"), clear_figure=True)

        with c8b:
            st.markdown("### 8B — Geste × type de faute")
            st.dataframe(tcd8b, use_container_width=True)
            st.pyplot(stacked_bar_from_df(tcd8b, "Geste", "Nb fautes", "Répartition des fautes par geste"), clear_figure=True)
    else:
        st.info("Aucune faute joueur détectée, impossible de profiler les erreurs.")

    # -----------------------------
    # TCD 9 — Zoom Service / Retour (issue × point pour)
    # -----------------------------
    st.markdown("## TCD 9 — Zoom Service / Retour (issue × point pour)")

    df_serve = dfv[serveur_l == player_norm].copy()
    df_return = dfv[serveur_l != player_norm].copy()

    # pour pivots : colonnes normalisées
    df_serve = df_serve.assign(point_l=point_l[df_serve.index], issue_l=issue_l[df_serve.index])
    df_return = df_return.assign(point_l=point_l[df_return.index], issue_l=issue_l[df_return.index])

    tcd9a = count_table(df_serve, index="issue_l", columns="point_l")
    tcd9b = count_table(df_return, index="issue_l", columns="point_l")

    c9a, c9b = st.columns(2)
    with c9a:
        st.markdown("### 9A — Quand le joueur sert")
        st.dataframe(tcd9a, use_container_width=True)
        if not tcd9a.empty:
            st.pyplot(stacked_bar_from_df(tcd9a, "Issue", "Nb points", "Service : issue → points"), clear_figure=True)

    with c9b:
        st.markdown("### 9B — Quand l’adversaire sert (retour)")
        st.dataframe(tcd9b, use_container_width=True)
        if not tcd9b.empty:
            st.pyplot(stacked_bar_from_df(tcd9b, "Issue", "Nb points", "Retour : issue → points"), clear_figure=True)

    # -----------------------------
    # Bonus — Ratio d'efficacité par geste
    # -----------------------------
    st.markdown("## Bonus — Ratio efficacité par geste (Gagnants / (Fautes+1))")
    t_g2 = count_table(gagnants, index="geste_norm")
    t_f2 = count_table(fautes_joueur, index="geste_norm")

    idx = sorted(set(t_g2.index).union(set(t_f2.index)))
    eff = pd.DataFrame(index=idx)
    eff["Gagnants"] = t_g2[t_g2.columns[0]] if not t_g2.empty else 0
    eff["Fautes"] = t_f2[t_f2.columns[0]] if not t_f2.empty else 0
    eff = eff.fillna(0)
    eff["Ratio"] = eff["Gagnants"] / (eff["Fautes"] + 1)
    st.dataframe(eff.sort_values(by="Ratio", ascending=False), use_container_width=True)
    st.pyplot(bar_from_series(eff["Ratio"].sort_values(ascending=False), "Geste", "Ratio", "Efficacité relative (G/(F+1))"), clear_figure=True)

    st.info("Astuce : utilise cette page pour choisir 2–3 messages simples à transmettre à Théophile.")
