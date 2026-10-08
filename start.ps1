# Start open-omnivoice-tts on Windows. Creates the local settings on the first run, builds the image if it
# does not exist yet (or was built for another GPU architecture), starts the container and waits until
# the model is loaded. The first start downloads the model, about 3 GB.
# -Build rebuilds the image even if it exists.
param([switch]$Build)
$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

function Get-EnvValue([string]$name, [string]$default) {
    $line = Get-Content .env | Where-Object { $_ -match "^\s*$name\s*=" } | Select-Object -First 1
    if ($line -match "^\s*$name\s*=\s*([^#]*)") { $v = $Matches[1].Trim(); if ($v) { return $v } }
    return $default
}

function Set-EnvValue([string]$name, [string]$value) {
    $found = $false
    $lines = @(Get-Content .env | ForEach-Object {
        if ($_ -match "^\s*$name\s*=") { $found = $true; "$name=$value" } else { $_ }
    })
    if (-not $found) { $lines += "$name=$value" }
    Set-Content -Path .env -Value $lines -Encoding ascii
}

function Invoke-Quiet([scriptblock]$block) {
    # Run a native command whose failure is handled by the caller (stderr is not an error here).
    $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    try { & $block } finally { $ErrorActionPreference = $prev }
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error 'Docker is not installed. See README.md -> Requirements.'
    exit 1
}
Invoke-Quiet { docker info 2>$null | Out-Null }
if ($LASTEXITCODE -ne 0) {
    Write-Error 'Docker is not running. Start Docker Desktop and try again.'
    exit 1
}

# OmniVoice's source is a git submodule: a clone without --recursive leaves its folder empty.
if (-not (Test-Path vendor/omnivoice-src/pyproject.toml)) {
    if ((Get-Command git -ErrorAction SilentlyContinue) -and (Test-Path .git)) {
        Write-Host '==> fetching the OmniVoice source (git submodule)'
        git submodule update --init --recursive
        if ($LASTEXITCODE -ne 0) { Write-Error 'git submodule update failed'; exit 1 }
    } else {
        Write-Error 'vendor/omnivoice-src is empty. Clone the repository with: git clone --recursive https://github.com/hypersniper05/open-omnivoice-tts.git'
        exit 1
    }
}

# Local settings (not tracked by git): create them from the tracked examples on the first run.
if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host '==> created .env from .env.example (GPU, port, ...)'
}
if (-not (Test-Path app/openai_voices.json)) {
    Copy-Item app/openai_voices.example.json app/openai_voices.json
    Write-Host '==> created app/openai_voices.json from the example (voice names, API key, ...)'
}
foreach ($d in 'model', 'Voices/custom', 'output/prompt_cache', 'output/voice_cache', 'output/inductor_cache') {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d | Out-Null }
}

$port = Get-EnvValue 'OMNIVOICE_PORT' '8008'
$gpu = Get-EnvValue 'OMNIVOICE_GPU' '0'
$device = Get-EnvValue 'OMNIVOICE_DEVICE' 'cuda:0'
$arch = Get-EnvValue 'FLASHINFER_CUDA_ARCH_LIST' ''

# The GPU and its architecture: the FlashInfer kernels are built for exactly one.
if ($device -ne 'cpu') {
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $info = Invoke-Quiet { nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader -i $gpu 2>$null }
        if ($LASTEXITCODE -eq 0 -and $info) {
            $parts = ($info | Select-Object -First 1).Split(',')
            $name = $parts[0].Trim(); $cap = $parts[1].Trim()
            Write-Host "==> GPU ${gpu}: $name (compute capability $cap)"
            if (-not $arch) {
                $arch = $cap
                Set-EnvValue 'FLASHINFER_CUDA_ARCH_LIST' $arch
                Write-Host "    set FLASHINFER_CUDA_ARCH_LIST=$arch in .env"
            }
        } else {
            Write-Host "    warning: nvidia-smi does not know GPU '$gpu' (OMNIVOICE_GPU in .env)" -ForegroundColor Yellow
        }
    } else {
        Write-Host '    warning: nvidia-smi was not found. Is the NVIDIA driver installed?' -ForegroundColor Yellow
    }
} else {
    Write-Host '==> CPU mode (OMNIVOICE_DEVICE=cpu): more than 100 times slower than a GPU'
}
if (-not $arch) { $arch = '8.6' }

# Build first, while any running instance keeps serving.
$image = 'omnivoice-tts:latest'
$labels = Invoke-Quiet { docker image inspect -f '{{json .Config.Labels}}' $image 2>$null }
$haveImage = ($LASTEXITCODE -eq 0)
$builtArch = ''
if ($haveImage -and $labels) { $builtArch = [string](($labels | ConvertFrom-Json).'omnivoice.flashinfer_arch') }
if ($Build -or -not $haveImage -or $builtArch -ne $arch) {
    if ($haveImage -and -not $Build) { Write-Host "==> the image has kernels for '$builtArch', this GPU needs $arch" }
    Write-Host '==> building the image (the first build downloads about 15 GB and takes 10 to 30 minutes)'
    docker compose build
    if ($LASTEXITCODE -ne 0) { Write-Error 'docker compose build failed'; exit 1 }
} else {
    Write-Host "==> using the existing image $image (start.cmd -Build rebuilds it)"
}

# Replace the running container with the new one.
Invoke-Quiet { docker compose down --remove-orphans 2>$null | Out-Null }
Invoke-Quiet { docker rm -f omnivoice-tts 2>$null | Out-Null }
docker compose up -d --no-build
if ($LASTEXITCODE -ne 0) { Write-Error 'docker compose up failed'; exit 1 }

$fromHub = -not (Test-Path model/config.json)   # no local copy: the model comes from Hugging Face
if ($fromHub) { Write-Host '==> waiting for the model (the first start downloads about 3 GB)' }
else { Write-Host '==> waiting for the model (from ./model)' }
$t0 = Get-Date
$lastNote = 0
while ($true) {
    try { $meta = Invoke-RestMethod -Uri "http://localhost:$port/api/meta" -TimeoutSec 5 -ErrorAction Stop } catch { $meta = $null }
    if ($meta) {
        if ($meta.ready) { break }
        if ($meta.error) {
            Write-Host "Startup failed: $($meta.error)" -ForegroundColor Red
            Write-Host 'Details: docker compose logs --tail 80'
            exit 1
        }
    }
    $running = Invoke-Quiet { docker ps --format '{{.Names}}' } | Where-Object { $_ -eq 'omnivoice-tts' }
    if (-not $running) {
        Write-Host 'The container stopped. Details: docker compose logs --tail 80' -ForegroundColor Red
        exit 1
    }
    $elapsed = [int]((Get-Date) - $t0).TotalSeconds
    if ($elapsed - $lastNote -ge 30) {
        $note = ''
        if ($fromHub) {
            $size = Invoke-Quiet { docker exec omnivoice-tts du -sh /app/.hf_cache/hub 2>$null }
            if ($size) { $note = ", downloaded $((([string]$size) -split '\s+')[0]) of about 3.1G" }
        }
        Write-Host "    still loading ($elapsed s$note)"
        $lastNote = $elapsed
    }
    Start-Sleep -Seconds 3
}

$secs = [int]((Get-Date) - $t0).TotalSeconds
$key = (Get-Content app/openai_voices.json -Raw | ConvertFrom-Json).api_key
$auth = if ($key) { 'API key required' } else { 'no API key' }
Write-Host ''
Write-Host "Ready after $secs s. OpenAI-compatible API ($auth):" -ForegroundColor Green
Write-Host "    base URL   http://localhost:$port/v1"
Write-Host "    API docs   http://localhost:$port/docs   (try POST /v1/audio/speech there)"
Write-Host "From other machines use this computer's IP address instead of localhost."
Write-Host 'Logs: docker compose logs -f     Stop: stop.cmd'
