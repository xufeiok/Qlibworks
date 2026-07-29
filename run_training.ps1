$old = Get-Item "scripts/training/score_tree_selected.csv" -ErrorAction SilentlyContinue
if ($old) { Write-Host "old_score_mtime=$($old.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))" }
& "E:/Conda_env/Qlib_env/python.exe" scripts/training/train_from_selected.py 2>&1
$new = Get-Item "scripts/training/score_tree_selected.csv" -ErrorAction SilentlyContinue
if ($new) { Write-Host "new_score_mtime=$($new.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))" }
