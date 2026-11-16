<#
.SYNOPSIS
  Sets up the meshsolid engine (its own Python venv) and installs the MeshSolid Fusion add-in.

.DESCRIPTION
  1. Finds Python 3.12 or 3.13 (the versions all engine wheels exist for on Windows).
  2. Creates a venv (default %LOCALAPPDATA%\meshsolid\venv) and installs requirements.txt.
  3. Runs the engine self-check: imports, then a synthetic box meshed and rebuilt to STEP.
  4. Copies fusion_addin\MeshSolid into Fusion's AddIns folder and writes its settings.json.
  5. With -FreeCAD: copies freecad_addin\MeshSolid (plus the shared mslib) into FreeCAD's user Mod
     folder (%APPDATA%\FreeCAD\Mod for FreeCAD 1.0, %APPDATA%\FreeCAD\v1-x\Mod for 1.1+) and writes its
     settings.json.  -NoFusion skips the Fusion copy.
  6. With -Gpu: installs the optional GPU extras (requirements-gpu.txt: NVIDIA Warp) and checks that
     CUDA is usable.  Without a usable GPU it warns and the engine runs on the CPU (--device auto).
  Stops at the first failing step with that step's error; it never reports a step it did not run.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File fusion_addin\install_windows.ps1
  powershell -ExecutionPolicy Bypass -File fusion_addin\install_windows.ps1 -CheckOnly
  powershell -ExecutionPolicy Bypass -File fusion_addin\install_windows.ps1 -FreeCAD -NoFusion
  powershell -ExecutionPolicy Bypass -File fusion_addin\install_windows.ps1 -Gpu
#>
param(
    [string]$Python = "",
    [string]$EngineDir = "",
    [string]$VenvDir = "",
    [string]$AddinsDir = "",
    [switch]$CheckOnly,
    [switch]$QuickCheck,
    [switch]$NoAddinCopy,
    [switch]$FreeCAD,
    [switch]$NoFusion,
    [switch]$Gpu,
    [string]$FreeCADModDir = ""
)
$ErrorActionPreference = "Stop"
$onWindows = ($env:OS -eq "Windows_NT")
if (-not $EngineDir) { $EngineDir = Split-Path -Parent $PSScriptRoot }
$EngineDir = (Resolve-Path $EngineDir).Path
if (-not $VenvDir) {
    $base = if ($env:LOCALAPPDATA) { $env:LOCALAPPDATA } else { Join-Path $HOME ".local/share" }
    $VenvDir = Join-Path (Join-Path $base "meshsolid") "venv"
}
if (-not $AddinsDir) { $AddinsDir = Join-Path $env:APPDATA "Autodesk\Autodesk Fusion 360\API\AddIns" }
$venvPy = if ($onWindows) { Join-Path $VenvDir "Scripts\python.exe" } else { Join-Path $VenvDir "bin/python" }

function Fail([string]$msg) { Write-Host "FAILED: $msg" -ForegroundColor Red; exit 1 }
function Step([string]$msg) { Write-Host "== $msg" -ForegroundColor Cyan }

if (-not (Test-Path (Join-Path $EngineDir "meshsolid\__init__.py"))) {
    Fail "no meshsolid package under $EngineDir (run this script from the repo's fusion_addin folder)"
}

function Find-Python {
    if ($Python) { return $Python }
    foreach ($v in @("3.12", "3.13")) {
        try {
            $exe = & py "-$v" -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $exe) { return $exe.Trim() }
        } catch { }
    }
    foreach ($c in @("python", "python3")) {
        $cmd = Get-Command $c -ErrorAction SilentlyContinue
        if ($cmd) { return $cmd.Source }
    }
    return $null
}

if (-not $CheckOnly) {
    Step "Python"
    $py = Find-Python
    if (-not $py) { Fail "Python 3.12 or 3.13 not found. Install it from python.org (tick 'Add to PATH' or use the py launcher) and rerun." }
    $ver = & $py -c "import sys; print('%d.%d' % sys.version_info[:2])"
    if ($LASTEXITCODE -ne 0) { Fail "could not run $py" }
    if (@("3.12", "3.13") -notcontains $ver.Trim()) {
        Fail "Python ${ver} at ${py}: the engine's wheels (cadquery-ocp, pymeshlab, libigl) are checked for 3.12 and 3.13. Pass -Python <path to python.exe 3.12/3.13>."
    }
    Write-Host "using $py (Python $ver)"

    Step "engine venv at $VenvDir"
    if (-not (Test-Path $venvPy)) {
        & $py -m venv $VenvDir
        if ($LASTEXITCODE -ne 0) { Fail "python -m venv $VenvDir failed" }
    }
    & $venvPy -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { Fail "pip upgrade failed (network or proxy?)" }
    & $venvPy -m pip install -r (Join-Path $EngineDir "requirements.txt")
    if ($LASTEXITCODE -ne 0) { Fail "pip install -r requirements.txt failed; the pip output above names the package" }
    if ($Gpu) {
        Step "GPU extras (NVIDIA Warp)"
        & $venvPy -m pip install -r (Join-Path $EngineDir "requirements-gpu.txt")
        if ($LASTEXITCODE -ne 0) { Fail "pip install -r requirements-gpu.txt failed; the pip output above names the package" }
    }
}
if ($Gpu) {
    Step "GPU check"
    $old = $env:PYTHONPATH
    $env:PYTHONPATH = $EngineDir
    $desc = & $venvPy -c "from meshsolid import device; ok, d = device.cuda_info(); print(d); raise SystemExit(0 if ok else 3)"
    $grc = $LASTEXITCODE
    $env:PYTHONPATH = $old
    if ($grc -eq 0) { Write-Host "GPU: $desc (the engine uses it with --device auto)" -ForegroundColor Green }
    else { Write-Host "warning: GPU extras installed but CUDA is not usable: $desc. The engine will run on the CPU." -ForegroundColor Yellow }
}

Step "engine self-check"
if (-not (Test-Path $venvPy)) { Fail "engine Python not found at $venvPy (run without -CheckOnly first)" }
$json = Join-Path ([IO.Path]::GetTempPath()) ("meshsolid_selfcheck_{0}.json" -f [guid]::NewGuid())
$old = $env:PYTHONPATH
$env:PYTHONPATH = $EngineDir
$args2 = @("-m", "meshsolid.selfcheck", "--json", $json)
if ($QuickCheck) { $args2 += "--quick" }
& $venvPy @args2 | Out-Null
$rc = $LASTEXITCODE
$env:PYTHONPATH = $old
if (-not (Test-Path $json)) { Fail "self-check wrote no result (exit $rc); run: `"$venvPy`" -m meshsolid.selfcheck" }
$res = Get-Content $json -Raw | ConvertFrom-Json
Remove-Item $json
foreach ($w in $res.warnings) { Write-Host "warning: $w" -ForegroundColor Yellow }
if (-not $res.ok) {
    foreach ($e in $res.errors) { Write-Host "error: $e" -ForegroundColor Red }
    Fail "engine self-check failed"
}
if ($res.roundtrip) {
    Write-Host ("synthetic box: status {0}, {1} solid, {2} s" -f $res.roundtrip.status, $res.roundtrip.n_solids, $res.roundtrip.seconds)
} else {
    Write-Host "imports ok (quick check: no round trip run)"
}

if ($CheckOnly -or $NoAddinCopy) { Write-Host "engine ready; add-in not copied"; exit 0 }

$settings = [ordered]@{ python = $venvPy; engine_dir = $EngineDir; timeout_s = 5400; extra_args = @() }
$shared = Join-Path (Split-Path -Parent $VenvDir) "settings.json"      # %LOCALAPPDATA%\meshsolid\settings.json
[IO.File]::WriteAllText($shared, ($settings | ConvertTo-Json))
Write-Host "engine settings: $shared"

if (-not $NoFusion) {
    Step "Fusion add-in -> $AddinsDir\MeshSolid"
    $dst = Join-Path $AddinsDir "MeshSolid"
    New-Item -ItemType Directory -Force -Path $dst | Out-Null
    Copy-Item -Recurse -Force (Join-Path $EngineDir "fusion_addin\MeshSolid\*") $dst
    Get-ChildItem -Recurse -Directory -Filter "__pycache__" $dst | Remove-Item -Recurse -Force
    [IO.File]::WriteAllText((Join-Path $dst "settings.json"), ($settings | ConvertTo-Json))
    Write-Host "settings.json written (python = $venvPy, engine_dir = $EngineDir)"
    Write-Host "Fusion: Utilities > Add-Ins (Shift+S) > Add-Ins tab > MeshSolid > Run. The command 'Mesh to Solid (meshsolid)' is in Utilities > Add-Ins." -ForegroundColor Green
}

if ($FreeCAD) {
    if (-not $FreeCADModDir) {
        $fcRoot = Join-Path $env:APPDATA "FreeCAD"
        $versioned = @()
        if (Test-Path $fcRoot) { $versioned = @(Get-ChildItem -Directory $fcRoot -Filter "v1-*" | Sort-Object Name) }
        if ($versioned.Count -gt 0) { $FreeCADModDir = Join-Path $versioned[-1].FullName "Mod" }
        else { $FreeCADModDir = Join-Path $fcRoot "Mod" }
    }
    Step "FreeCAD workbench -> $FreeCADModDir\MeshSolid"
    $fdst = Join-Path $FreeCADModDir "MeshSolid"
    New-Item -ItemType Directory -Force -Path $fdst | Out-Null
    Copy-Item -Recurse -Force (Join-Path $EngineDir "freecad_addin\MeshSolid\*") $fdst
    Copy-Item -Recurse -Force (Join-Path $EngineDir "fusion_addin\MeshSolid\mslib") $fdst
    Get-ChildItem -Recurse -Directory -Filter "__pycache__" $fdst | Remove-Item -Recurse -Force
    [IO.File]::WriteAllText((Join-Path $fdst "settings.json"), ($settings | ConvertTo-Json))
    Write-Host "settings.json written; shared mslib copied"
    Write-Host "FreeCAD: restart FreeCAD, pick the MeshSolid workbench, select a Mesh object, click Mesh to Solid." -ForegroundColor Green
    Write-Host "(If FreeCAD shows no MeshSolid workbench, check Edit > Preferences > General for the user data folder and pass -FreeCADModDir <that folder>\Mod.)"
}
Write-Host "Job files go to %LOCALAPPDATA%\meshsolid\jobs."
