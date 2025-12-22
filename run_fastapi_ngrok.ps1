param(
  [int]$Port = 8000,
  [string]$App = "luxwatcher.main:app",
  [int]$NgrokInspectPort = 4040,
  [switch]$SkipInstall,
  [switch]$DisablePipeline,
  [switch]$ExitAfterUrl,
  [string]$Domain
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

function Get-DotEnvValue([string]$path, [string]$key) {
  if (!(Test-Path $path)) { return $null }
  foreach ($line in Get-Content -LiteralPath $path) {
    $t = $line.Trim()
    if (!$t -or $t.StartsWith("#")) { continue }
    $idx = $t.IndexOf("=")
    if ($idx -lt 1) { continue }
    $k = $t.Substring(0, $idx).Trim()
    if ($k -ne $key) { continue }
    $v = $t.Substring($idx + 1).Trim()
    if (($v.StartsWith('"') -and $v.EndsWith('"')) -or ($v.StartsWith("'") -and $v.EndsWith("'"))) {
      $v = $v.Substring(1, $v.Length - 2)
    }
    return $v
  }
  return $null
}

function Resolve-NgrokExe() {
  $cmd = Get-Command ngrok -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Source }

  $local = Join-Path $env:LOCALAPPDATA "ngrok\\ngrok.exe"
  if (Test-Path $local) { return $local }

  $installDir = Split-Path -Parent $local
  New-Item -ItemType Directory -Force -Path $installDir | Out-Null

  $zipPath = Join-Path $env:TEMP "ngrok.zip"
  $url = "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-windows-amd64.zip"
  Invoke-WebRequest -Uri $url -OutFile $zipPath
  Expand-Archive -LiteralPath $zipPath -DestinationPath $installDir -Force
  Remove-Item -LiteralPath $zipPath -Force -ErrorAction SilentlyContinue

  if (!(Test-Path $local)) {
    throw "ngrok.exe not found after download. Install ngrok and ensure it is on PATH."
  }
  return $local
}

if (!(Test-Path ".\\.venv")) {
  python -m venv .venv
}

.\\.venv\\Scripts\\Activate.ps1

if (!$SkipInstall) {
  pip install -r requirements.fastapi.txt --upgrade
}

if (!(Test-Path ".\\.env")) {
  Copy-Item .env.example .env
  Write-Host "Created .env from .env.example. Please set FIRECRAWL_API_KEY and NGROK_AUTH_TOKEN."
  exit 1
}

$ngrokToken = Get-DotEnvValue ".\\.env" "NGROK_AUTH_TOKEN"
if (!$ngrokToken) { $ngrokToken = Get-DotEnvValue ".\\.env" "NGROK_AUTHTOKEN" }
if (!$ngrokToken) { throw "Missing NGROK_AUTH_TOKEN in .env" }

$envDomain = Get-DotEnvValue ".\\.env" "NGROK_DOMAIN"
if (!$Domain -and $envDomain) { $Domain = $envDomain }

$ngrokExe = Resolve-NgrokExe

& $ngrokExe "config" "add-authtoken" $ngrokToken | Out-Null
if ($LASTEXITCODE -ne 0) {
  & $ngrokExe "authtoken" $ngrokToken | Out-Null
  if ($LASTEXITCODE -ne 0) { throw "Failed to configure ngrok authtoken." }
}

$pythonExe = Resolve-Path ".\\.venv\\Scripts\\python.exe"

$uvicornArgs = @(
  "-m", "uvicorn", $App,
  "--host", "127.0.0.1",
  "--port", $Port.ToString()
)
if ($DisablePipeline) {
  $uvicornArgs += @("--lifespan", "off")
}

$uvicornProc = Start-Process -FilePath $pythonExe -ArgumentList @(
  $uvicornArgs
) -PassThru -NoNewWindow

$ngrokArgs = @("http", $Port.ToString(), "--inspect=true")
if ($Domain) { $ngrokArgs += @("--domain", $Domain) }
$ngrokProc = Start-Process -FilePath $ngrokExe -ArgumentList $ngrokArgs -PassThru -NoNewWindow

try {
  $deadline = (Get-Date).AddSeconds(30)
  $publicUrl = $null
  while ((Get-Date) -lt $deadline -and !$publicUrl) {
    try {
      $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$NgrokInspectPort/api/tunnels" -TimeoutSec 2
      $publicUrl = ($resp.tunnels | Where-Object { $_.public_url -like "https://*" } | Select-Object -First 1).public_url
    } catch {
      Start-Sleep -Seconds 1
    }
  }

  if ($publicUrl) {
    Write-Host ""
    Write-Host "ngrok public URL: $publicUrl"
    Write-Host "Local URL: http://127.0.0.1:$Port"
    if ($DisablePipeline) {
      Write-Host "Note: started with --lifespan off (pipeline disabled)."
    }
    if ($ExitAfterUrl) {
      Write-Host ""
      return
    }
    Write-Host "Press Ctrl+C to stop."
    Write-Host ""
  } else {
    if ($ExitAfterUrl) { throw "ngrok started but no public URL found yet." }
    Write-Host "ngrok started but no public URL found yet. Check ngrok logs or http://127.0.0.1:$NgrokInspectPort"
  }

  Wait-Process -Id $uvicornProc.Id
} finally {
  if ($ngrokProc -and -not $ngrokProc.HasExited) {
    Stop-Process -Id $ngrokProc.Id -Force -ErrorAction SilentlyContinue
  }
  if ($uvicornProc -and -not $uvicornProc.HasExited) {
    Stop-Process -Id $uvicornProc.Id -Force -ErrorAction SilentlyContinue
  }
}
