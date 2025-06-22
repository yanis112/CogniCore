# Workflow de Publication sur PyPI avec Git

Ce document décrit la procédure à suivre pour publier une nouvelle version du package `cognicore` sur PyPI. Le processus est automatisé grâce à une action GitHub qui se déclenche lors de la création et de la poussée (push) d'un nouveau tag Git.

## Prérequis

- Avoir les droits de push sur le dépôt GitHub.
- S'assurer que les modifications à publier sont bien présentes sur votre branche principale (ex: `main` ou `master`).

## Étapes pour publier une nouvelle version

Suivez ces étapes dans l'ordre pour garantir une publication réussie.

### 1. Finaliser les modifications du code

Assurez-vous que tout le code que vous souhaitez inclure dans la nouvelle version est complet, testé et mergé sur votre branche principale.

### 2. Mettre les changements en attente et les commiter

Une fois vos modifications prêtes, utilisez les commandes Git habituelles pour les commiter.

```bash
# 1. Ajoutez tous les fichiers modifiés à la zone de staging
git add .

# 2. Commitez vos changements avec un message clair et descriptif
# (ex: "feat: Ajout du support pour le modèle X" ou "fix: Correction d'un bug dans l'analyse d'image")
git commit -m "feat: Description de vos changements"
```

### 3. Créer un nouveau tag Git

La version du package est directement liée au tag Git. Il est crucial de suivre le versionnement sémantique (MAJOR.MINOR.PATCH).

- **MAJOR** (ex: `v1.0.0` -> `v2.0.0`): Pour des changements non rétrocompatibles.
- **MINOR** (ex: `v1.1.0` -> `v1.2.0`): Pour l'ajout de nouvelles fonctionnalités rétrocompatibles.
- **PATCH** (ex: `v1.1.1` -> `v1.1.2`): Pour des corrections de bugs rétrocompatibles.

Créez le tag avec la commande suivante (le `v` au début est important) :

```bash
# Exemple pour une nouvelle version mineure
git tag v0.2.0
```

### 4. Pousser (push) les commits et le tag sur GitHub

Pour déclencher le workflow de publication, vous devez pousser vos commits ainsi que le nouveau tag sur le dépôt distant.

```bash
# 1. Poussez d'abord vos commits
git push

# 2. Ensuite, poussez le tag que vous venez de créer
git push --tags
```

### 5. Vérifier la publication

Une fois le tag poussé, l'action GitHub va automatiquement démarrer.

1.  Allez dans l'onglet **"Actions"** de votre dépôt GitHub pour suivre l'avancement du workflow nommé `Publish Python Package to PyPI with UV`.
2.  Si le workflow se termine avec succès, votre nouvelle version sera disponible sur [PyPI](https://pypi.org/project/cognicore/).
3.  Vous pouvez vérifier que la nouvelle version est en ligne et l'installer avec `uv pip install cognicore==<votre-nouvelle-version>`.

--- 

En suivant ce processus, vous assurez une gestion de version centralisée et une publication automatisée et fiable de votre package.
