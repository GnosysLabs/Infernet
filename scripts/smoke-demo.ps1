param(
  [string]$Prompt = "hello infernet"
)

$ErrorActionPreference = "Stop"

$RootDir = Resolve-Path (Join-Path $PSScriptRoot "..")
$Topic = "infernet/smoke/$PID"
$WorkerExe = Join-Path $RootDir "target/debug/infernet-worker.exe"
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
      -RedirectStandardOutput (Join-Path $RootDir "target/infernet-peer-$($Peer.Name).log") `
      -RedirectStandardError (Join-Path $RootDir "target/infernet-peer-$($Peer.Name).err.log") `
      -PassThru `
      -WindowStyle Hidden
  }

  Start-Sleep -Seconds 2

  & $WorkerExe infer `
    --model grid-demo-12 `
    --prompt $Prompt `
    --topic $Topic `
    --discovery-timeout-ms 6000
  if ($LASTEXITCODE -ne 0) {
    throw "infernet-worker infer failed with exit code $LASTEXITCODE"
  }
}
finally {
  foreach ($Process in $Processes) {
    Stop-Process -Id $Process.Id -ErrorAction SilentlyContinue
  }
}
