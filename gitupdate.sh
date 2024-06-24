git checkout --orphan latest_branch
git add -A
git commit -am "first"
git branch -D page
git branch -m page
git push -f origin page:page