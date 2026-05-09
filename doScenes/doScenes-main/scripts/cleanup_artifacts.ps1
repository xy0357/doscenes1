param(
    [switch]$DryRun = $false
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Assert-InRepo {
    param(
        [string]$RepoRoot,
        [string]$TargetPath
    )
    $repo = [System.IO.Path]::GetFullPath($RepoRoot)
    $target = [System.IO.Path]::GetFullPath($TargetPath)
    if (-not $target.StartsWith($repo, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refuse to operate outside repo. target=$target repo=$repo"
    }
}

function Remove-IfExists {
    param(
        [string]$Path,
        [switch]$Recurse
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }
    if ($DryRun) {
        Write-Host "[DryRun] Remove: $Path"
        return
    }
    if ($Recurse) {
        Remove-Item -LiteralPath $Path -Recurse -Force
    } else {
        Remove-Item -LiteralPath $Path -Force
    }
    Write-Host "[Done] Remove: $Path"
}

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
Assert-InRepo -RepoRoot $repoRoot -TargetPath $repoRoot

$targets = @(
    (Join-Path $repoRoot "__pycache__"),
    (Join-Path $repoRoot "train_full.log"),
    (Join-Path $repoRoot "smoke_train.log"),
    (Join-Path $repoRoot "evaluate_after_train.log")
)

foreach ($t in $targets) {
    Assert-InRepo -RepoRoot $repoRoot -TargetPath $t
    if ($t.EndsWith("__pycache__", [System.StringComparison]::OrdinalIgnoreCase)) {
        Remove-IfExists -Path $t -Recurse
    } else {
        Remove-IfExists -Path $t
    }
}

$ckptDir = Join-Path $repoRoot "artifacts\checkpoints"
Assert-InRepo -RepoRoot $repoRoot -TargetPath $ckptDir
if (Test-Path -LiteralPath $ckptDir) {
    $smokeFiles = Get-ChildItem -LiteralPath $ckptDir -File -Filter "*smoke*.pth" -ErrorAction SilentlyContinue
    foreach ($f in $smokeFiles) {
        Assert-InRepo -RepoRoot $repoRoot -TargetPath $f.FullName
        Remove-IfExists -Path $f.FullName
    }
}

Write-Host ""
if ($DryRun) {
    Write-Host "保守清理 DryRun 完成。未实际删除文件。"
} else {
    Write-Host "保守清理完成。"
}
