# git_push.ps1
# Run this script once from the "Scene Decomposition" folder to push to GitHub.
# Usage:  .\git_push.ps1

Set-Location -LiteralPath $PSScriptRoot

# 1. Init repo (safe to re-run on an already-initialised repo)
git init

# 2. Set remote (remove existing 'origin' first if it already exists)
git remote remove origin 2>$null
git remote add origin https://github.com/gangu9890/Scene-Graph-Decomposition.git

# 3. Stage all tracked files (respects .gitignore)
git add .

# 4. Show what will be committed
Write-Host "`n--- Files staged for commit ---"
git status --short

# 5. Commit
git commit -m "Initial commit: GNN scene-graph decomposition model"

# 6. Rename branch to main (GitHub default)
git branch -M main

# 7. Push
git push -u origin main

Write-Host "`nDone! Check https://github.com/gangu9890/Scene-Graph-Decomposition"
