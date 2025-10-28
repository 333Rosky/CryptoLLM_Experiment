# Fix Git Push Issues

## Problème 1 : Virtualenv versionné (RÉSOLU)
Un fichier `.gitignore` a été créé pour exclure `.venv/` et autres fichiers inutiles.

**Prochaine étape** :
```bash
# Retirer .venv/ du git (sans le supprimer de votre disque)
git rm -r --cached .venv/

# Vérifier ce qui sera committé
git status

# Committer le .gitignore et la suppression de .venv/
git add .gitignore
git commit -m "Add .gitignore and remove .venv/"
```

## Problème 2 : Scope manquant (À FAIRE)

Le token GitHub manque le scope `workflow`. Solution :

### Option A : Créer un nouveau token avec scope workflow
1. Allez sur https://github.com/settings/tokens/new
2. Cochez **`workflow`** dans les scopes
3. Créez le token
4. Ajoutez-le à votre config git :
```bash
# Voir vos credentials actuels
git config --list | grep credential

# Changer l'URL pour utiliser le token
git remote set-url origin https://VOTRE_TOKEN@github.com/333Rosky/CryptoLLM_Experiment.git
```

### Option B : Supprimer les workflows pour l'instant
Si vous ne voulez pas modifier le workflow :
```bash
rm .github/workflows/ci.yml  # Supprimer temporairement
git add -A
git commit -m "Remove workflow temporarily"
git push origin main
```

## Après avoir résolu les 2 problèmes

```bash
# 1. Nettoie .venv/
git rm -r --cached .venv/
git add .gitignore
git commit -m "Add .gitignore and remove .venv/"

# 2. Push (avec le bon scope)
git push origin main
```

