$ErrorActionPreference = "Stop"

$RootDir = Resolve-Path (Join-Path $PSScriptRoot "..")
$Topic = "infernet/grid-demo/1"
$WorkerExe = Join-Path $RootDir "target/debug/infernet-worker.exe"
$UiDir = Join-Path $RootDir "infernet-ui"
$Processes = @()

Set-Location $RootDir

cargo build -p infernet-worker
if ($LASTEXITCODE -ne 0) {
  throw "cargo build failed with exit code $LASTEXITCODE"
}

try {
  $Peers = @(
    @{ Name = "a"; Layers = "0:3" },
    @{ Name = "b"; Layers = "3:6" },
    @{ Name = "c"; Layers = "6:9" },
    @{ Name = "d"; Layers = "9:12" }
  )

  foreach ($Peer in $Peers) {
    $Processes += Start-Process `
      -FilePath $WorkerExe `
      -ArgumentList @("serve", "--model", "grid-demo-12", "--layers", $Peer.Layers, "--topic", $Topic) `
      -RedirectStandardOutput (Join-Path $RootDir "target/infernet-ui-peer-$($Peer.Name).log") `
      -RedirectStandardError (Join-Path $RootDir "target/infernet-ui-peer-$($Peer.Name).err.log") `
      -PassThru `
      -WindowStyle Hidden
  }

  Start-Sleep -Seconds 2

  if (-not (Test-Path (Join-Path $UiDir "node_modules"))) {
    npm --prefix $UiDir install
    if ($LASTEXITCODE -ne 0) {
      throw "npm install failed with exit code $LASTEXITCODE"
    }
  }

  npm --prefix $UiDir run tauri dev
  if ($LASTEXITCODE -ne 0) {
    throw "npm run tauri dev failed with exit code $LASTEXITCODE"
  }
}
finally {
  foreach ($Process in $Processes) {
    Stop-Process -Id $Process.Id -ErrorAction SilentlyContinue
  }
}
