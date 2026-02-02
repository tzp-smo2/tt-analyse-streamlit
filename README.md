# 🏓 TT Analyse (Streamlit)

App Streamlit pour analyser un match de tennis de table à partir d'un fichier Excel (1 ligne = 1 point).

## Contenu
- `app.py` : l'application
- `requirements.txt` : dépendances
- `.gitignore` : ignore venv/cache
- `README.md` : guide

## 1) Lancer en local
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 2) Format attendu
L'app te propose un **mapping de colonnes** au démarrage.
Champs nécessaires :
- point_pour
- issue_point (valeurs attendues : `Gagnant` / `Faute`)
- Serveur
- geste_technique
- Zone_table

Optionnels :
- auteur_faute
- manche
- effet

## 3) Déployer avec GitHub + Streamlit Community Cloud

### A) Créer un repo GitHub
1. GitHub → New repository
2. Nom : `tt-analyse-streamlit` (ou autre)

### B) Pousser le code
Dans le dossier du projet :
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<TON_USER>/<TON_REPO>.git
git push -u origin main
```

### C) Déployer
1. https://share.streamlit.io
2. Login GitHub
3. Create app
4. Repository : ton repo
5. Branch : `main`
6. Main file path : `app.py`
7. Deploy
