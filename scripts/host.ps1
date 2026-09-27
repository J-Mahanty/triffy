<#
.SYNOPSIS
  Host Triffy from this laptop, with a public link anyone can open.

.DESCRIPTION
  Four long-running processes:

    collector A   92 London cameras, owns the GPU
    collector B   80 more cameras, CPU only
    website       the API and the site, on 127.0.0.1:8000, read-only
    tunnel        a Cloudflare quick tunnel: a public https link to the website

  The tunnel connects outwards from this laptop, so no router or firewall
  change is needed and the website itself is never opened to the network.
  A quick tunnel's link changes every time the tunnel starts; `status` prints
  the current one.

  Rules carried over from the old demo script, both learned the hard way:
  * Only one process may use CUDA: two at once hang the GTX 1650 and nothing
    is collected. Collector A gets the GPU; everything else has it hidden.
  * Never start a second copy: duplicate collectors double-write the data.
    Anything already running is left alone.

  Keep the laptop plugged in and awake. A sleeping laptop stops collecting and
  takes the link down, and neither comes back by itself.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\host.ps1 start
#>
param(
  [ValidateSet('start', 'stop', 'status', 'restart')]
  [string]$Action = 'status'
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Port = 8000
$TunnelLog = Join-Path $Root 'data\tunnel.log'

$Python = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $Python) { Write-Host "No python on PATH." -ForegroundColor Red; exit 1 }
$Cloudflared = (Get-Command cloudflared -ErrorAction SilentlyContinue).Source
if (-not $Cloudflared) { $Cloudflared = 'C:\Program Files (x86)\cloudflared\cloudflared.exe' }

$Collectors = @(
  @{ Name = 'collector A'; Match = 'collector_ids_A'; Gpu = $true
     Args = @('-m', 'route_engine.collector', '--city', 'lon', '--ids', 'data\collector_ids_A.txt',
              '--cameras', '92', '--interval', '360')
     Out = 'data\collector_A.log'; Err = 'data\collector_A.err' }
  @{ Name = 'collector B'; Match = 'collector_ids_B'; Gpu = $false
     Args = @('-m', 'route_engine.collector', '--city', 'lon', '--ids', 'data\collector_ids_B.txt',
              '--cameras', '80', '--interval', '360', '--device', 'cpu')
     Out = 'data\collector_B.log'; Err = 'data\collector_B.err' }
)

function Get-Running($match, $name = 'python%') {
  Get-CimInstance Win32_Process -Filter "Name like '$name'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -and $_.CommandLine -like "*$match*" }
}
function Get-Website { Get-Running 'route_engine.api' }
function Get-Tunnel  { Get-Running "127.0.0.1:$Port" 'cloudflared%' }

function Start-Python($name, $argList, $out, $err, [bool]$gpu) {
  # Only collector A may see the GPU; hiding it is more reliable than trusting
  # every code path to honour a --device flag.
  if ($gpu) { Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue }
  else      { $env:CUDA_VISIBLE_DEVICES = '-1' }
  $env:TRIFFY_DASH_DEVICE = 'cpu'
  Start-Process -FilePath $Python -ArgumentList $argList -WorkingDirectory $Root `
                -WindowStyle Hidden -RedirectStandardOutput $out -RedirectStandardError $err
  Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
  Write-Host ("  {0,-12} started" -f $name) -ForegroundColor Green
}

# The quick tunnel prints its link a few seconds after starting.
function Get-PublicUrl {
  if (-not (Test-Path $TunnelLog)) { return $null }
  $m = Select-String -Path $TunnelLog -Pattern 'https://[a-z0-9-]+\.trycloudflare\.com' |
       Select-Object -Last 1
  if ($m) { return $m.Matches[0].Value }
  return $null
}

function Start-All {
  Write-Host "`nSTARTING" -ForegroundColor Cyan
  foreach ($c in $Collectors) {
    if (Get-Running $c.Match) { Write-Host ("  {0,-12} already running" -f $c.Name) -ForegroundColor DarkGray }
    else { Start-Python $c.Name $c.Args $c.Out $c.Err $c.Gpu }
  }

  # The tunnel first: the website needs its link, for the map links the chat
  # hands out.
  if (Get-Tunnel) { Write-Host "  tunnel       already running" -ForegroundColor DarkGray }
  else {
    Remove-Item $TunnelLog -ErrorAction SilentlyContinue
    Start-Process -FilePath $Cloudflared -WindowStyle Hidden `
      -ArgumentList @('tunnel', '--no-autoupdate', '--url', "http://127.0.0.1:$Port") `
      -RedirectStandardError $TunnelLog -RedirectStandardOutput 'data\tunnel.out'
    Write-Host "  tunnel       started" -ForegroundColor Green
  }
  $url = $null
  for ($i = 0; $i -lt 30 -and -not $url; $i++) { Start-Sleep -Seconds 1; $url = Get-PublicUrl }
  if (-not $url) { Write-Host "  tunnel gave no link yet; see data\tunnel.log" -ForegroundColor Yellow }

  if (Get-Website) { Write-Host "  website      already running" -ForegroundColor DarkGray }
  else {
    # Read-only: the public cannot change the clock, incidents or feedback.
    $env:TRIFFY_READONLY = '1'
    $env:TRIFFY_PORT = "$Port"
    if ($url) { $env:TRIFFY_MAP_BASE = $url }
    Start-Python 'website' @('-m', 'route_engine.api') 'data\api.log' 'data\api.err' $false
    Remove-Item Env:TRIFFY_READONLY, Env:TRIFFY_MAP_BASE, Env:TRIFFY_PORT -ErrorAction SilentlyContinue
  }
  Wait-ForWebsite | Out-Null
  Show-Status
}

function Stop-All {
  Write-Host "`nSTOPPING" -ForegroundColor Cyan
  $procs = @()
  foreach ($c in $Collectors) { $procs += @(Get-Running $c.Match) }
  $procs += @(Get-Website) + @(Get-Tunnel)
  $procs = $procs | Where-Object { $_ }
  foreach ($p in $procs) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
  Write-Host ("  stopped {0} process(es)" -f @($procs).Count) -ForegroundColor Yellow
}

function Wait-ForWebsite([int]$seconds = 120) {
  Write-Host "  waiting for the website..." -NoNewline
  for ($i = 0; $i -lt $seconds; $i += 3) {
    try {
      Invoke-WebRequest -Uri "http://127.0.0.1:$Port/api/profiles" -TimeoutSec 4 -UseBasicParsing | Out-Null
      Write-Host " ready" -ForegroundColor Green; return $true
    } catch { Start-Sleep -Seconds 3; Write-Host "." -NoNewline }
  }
  Write-Host " not yet (see data\api.err)" -ForegroundColor Yellow; return $false
}

function Show-Status {
  Write-Host "`nPROCESSES" -ForegroundColor Cyan
  foreach ($c in $Collectors) {
    $p = Get-Running $c.Match
    if ($p) { Write-Host ("  {0,-12} up    PID {1}" -f $c.Name, $p[0].ProcessId) -ForegroundColor Green }
    else    { Write-Host ("  {0,-12} DOWN" -f $c.Name) -ForegroundColor Red }
  }
  foreach ($s in @(@{ N = 'website'; P = (Get-Website) }, @{ N = 'tunnel'; P = (Get-Tunnel) })) {
    if ($s.P) { Write-Host ("  {0,-12} up    PID {1}" -f $s.N, @($s.P)[0].ProcessId) -ForegroundColor Green }
    else      { Write-Host ("  {0,-12} DOWN" -f $s.N) -ForegroundColor Red }
  }

  # Freshness matters more than liveness: a laptop that slept leaves processes
  # that look fine while nothing has been collected.
  Write-Host "`nDATA" -ForegroundColor Cyan
  $obs = Join-Path $Root 'data\live_observations.jsonl'
  if (Test-Path $obs) {
    $age = (New-TimeSpan -Start (Get-Item $obs).LastWriteTime -End (Get-Date)).TotalMinutes
    if ($age -lt 15) { Write-Host ("  camera readings  fresh, last written {0:N0} min ago" -f $age) -ForegroundColor Green }
    else { Write-Host ("  camera readings  STALE: nothing written for {0:N0} min. Did the laptop sleep?" -f $age) -ForegroundColor Red }
  }

  Write-Host "`nLINK" -ForegroundColor Cyan
  $url = Get-PublicUrl
  if ($url -and (Get-Tunnel)) { Write-Host "  $url" -ForegroundColor Green }
  else { Write-Host "  no public link (tunnel down)" -ForegroundColor Red }
  Write-Host ""
}

switch ($Action) {
  'start'   { Start-All }
  'stop'    { Stop-All }
  'restart' { Stop-All; Start-Sleep -Seconds 3; Start-All }
  'status'  { Show-Status }
}
