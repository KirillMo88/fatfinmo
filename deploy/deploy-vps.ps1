[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$SshKey,
    [string]$VpsHost = "root@167.233.42.33",
    [string]$RemoteRoot = "/opt/fatfinmo",
    [string]$HealthUrl = "https://finxmo.com/_stcore/health"
)

$ErrorActionPreference = "Stop"

function Invoke-Native {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Command failed with exit code $LASTEXITCODE"
    }
}

function Get-NativeOutput {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    $output = & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Command failed with exit code $LASTEXITCODE"
    }
    return ($output | Out-String).Trim()
}

if (-not (Test-Path -LiteralPath $SshKey -PathType Leaf)) {
    throw "SSH key not found: $SshKey"
}
if ($VpsHost -notmatch '^[A-Za-z0-9_.@-]+$') {
    throw "VpsHost contains unsupported characters."
}
if ($RemoteRoot -notmatch '^/[A-Za-z0-9._/-]+$') {
    throw "RemoteRoot must be an absolute Linux path without spaces."
}

$repoRoot = Get-NativeOutput -Command "git" -Arguments @("rev-parse", "--show-toplevel")
Push-Location $repoRoot
$tempDir = $null
try {
    $dirty = Get-NativeOutput -Command "git" -Arguments @("status", "--porcelain")
    if ($dirty) {
        throw "Working tree is not clean. Commit all changes before deployment."
    }

    Invoke-Native -Command "git" -Arguments @("fetch", "--prune", "origin", "main")
    $head = Get-NativeOutput -Command "git" -Arguments @("rev-parse", "HEAD")
    $originMain = Get-NativeOutput -Command "git" -Arguments @("rev-parse", "origin/main")
    if ($head -ne $originMain) {
        throw "HEAD ($head) is not origin/main ($originMain). Merge and push to GitHub before deployment."
    }

    $shortSha = $head.Substring(0, 7)
    $timestamp = (Get-Date).ToUniversalTime().ToString("yyyyMMdd-HHmmss")
    $releaseName = "github-main-$timestamp-$shortSha"
    $releasePath = "$RemoteRoot/releases/$releaseName"
    $remoteArchive = "/tmp/$releaseName.tar.gz"

    $tempDir = Join-Path ([System.IO.Path]::GetTempPath()) $releaseName
    New-Item -ItemType Directory -Path $tempDir | Out-Null
    $archivePath = Join-Path $tempDir "$releaseName.tar.gz"
    Invoke-Native -Command "git" -Arguments @("archive", "--format=tar.gz", "--output=$archivePath", $head)

    $sshArgs = @(
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=yes",
        "-i", $SshKey
    )
    $scpArgs = @(
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=yes",
        "-i", $SshKey,
        $archivePath,
        "${VpsHost}:$remoteArchive"
    )
    Invoke-Native -Command "scp" -Arguments $scpArgs

    $prepare = "set -eu; test -f '$RemoteRoot/.env'; test -d '$RemoteRoot/persistent'; test ! -e '$releasePath'; mkdir -p '$releasePath'; tar -xzf '$remoteArchive' -C '$releasePath'; ln -s '$RemoteRoot/.env' '$releasePath/.env'; ln -s '$RemoteRoot/persistent' '$releasePath/persistent'; rm -f '$remoteArchive'; docker compose -p fatfinmo -f '$releasePath/docker-compose.yml' --env-file '$RemoteRoot/.env' config -q"
    Invoke-Native -Command "ssh" -Arguments ($sshArgs + @($VpsHost, $prepare))

    $deploy = "set -eu; docker compose -p fatfinmo -f '$releasePath/docker-compose.yml' --env-file '$RemoteRoot/.env' build screener screener-jobs; docker compose -p fatfinmo -f '$releasePath/docker-compose.yml' --env-file '$RemoteRoot/.env' up -d --no-deps screener screener-jobs; curl -fsS '$HealthUrl'; ln -sfn '$releasePath' '$RemoteRoot/current'"
    Invoke-Native -Command "ssh" -Arguments ($sshArgs + @($VpsHost, $deploy))

    Write-Host "Deployed GitHub main $head"
    Write-Host "Release: $releasePath"
}
finally {
    Pop-Location
    if ($tempDir -and (Test-Path -LiteralPath $tempDir)) {
        Remove-Item -LiteralPath $tempDir -Recurse -Force
    }
}
