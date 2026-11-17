$ErrorActionPreference = 'Stop'
$workspaceRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))

function Move-AuditPath([string]$sourceRelative, [string]$targetRelative) {
    $sourcePath = [IO.Path]::GetFullPath((Join-Path $workspaceRoot $sourceRelative))
    $targetPath = [IO.Path]::GetFullPath((Join-Path $workspaceRoot $targetRelative))
    $prefix = $workspaceRoot.TrimEnd('\') + '\'
    if (-not $sourcePath.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase) -or
        -not $targetPath.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Migration path escaped the project workspace'
    }
    if (-not (Test-Path -LiteralPath $sourcePath)) { return }
    if (Test-Path -LiteralPath $targetPath) { throw "Destination already exists: $targetPath" }
    New-Item -ItemType Directory -Path (Split-Path -Parent $targetPath) -Force | Out-Null
    Move-Item -LiteralPath $sourcePath -Destination $targetPath
    Write-Output "$sourceRelative -> $targetRelative"
}

# Stop only the audit service identified on its own loopback port.
$listeners = Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue
foreach ($auditProcessId in ($listeners.OwningProcess | Sort-Object -Unique)) {
    $auditProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$auditProcessId"
    if ($auditProcess.CommandLine -notmatch 'dice_audit\.py\s+serve') {
        throw "Port 8765 belongs to another service: $auditProcessId"
    }
    $result = Invoke-CimMethod -InputObject $auditProcess -MethodName Terminate
    if ($result.ReturnValue -ne 0) { throw "Cannot stop legacy audit: $($result.ReturnValue)" }
    Write-Output "Stopped legacy audit process $auditProcessId"
}

Move-AuditPath 'audit-ui' 'audit\audit-ui'
Move-AuditPath 'src\dice_viewer\audit' 'audit\audit-back\src\dice_audit'
Move-AuditPath 'src\dice_viewer\data\audit_import.py' 'audit\audit-back\src\dice_audit\importers.py'
Move-AuditPath 'dice_audit.py' 'audit\audit-back\run.py'
Move-AuditPath 'audit-requirements.txt' 'audit\audit-back\requirements.txt'
Move-AuditPath 'start-audit.cmd' 'audit\start-audit.cmd'
Move-AuditPath 'README_AUDITORIA.md' 'audit\README.md'
Move-AuditPath 'skills\review-dice-images' 'audit\skills\review-dice-images'
Move-AuditPath 'runtime\audit' 'audit\audit-back\runtime'
Move-AuditPath 'runtime\audit-manual' 'audit\audit-back\runtime-manual'
Move-AuditPath '.venv-audit' 'audit\audit-back\runtime\legacy-env'
Move-AuditPath 'runtime\audit-dependencies.py' 'audit\audit-back\runtime\audit-dependencies.py'

Move-AuditPath 'src\dice_viewer\data\__pycache__' 'audit\audit-back\runtime\legacy-import-cache'
Move-AuditPath '__pycache__\dice_audit.cpython-311.pyc' 'audit\audit-back\runtime\legacy-bootstrap-cache\dice_audit.cpython-311.pyc'

foreach ($relative in @('src\dice_viewer\data', 'skills')) {
    $oldDirectory = [IO.Path]::GetFullPath((Join-Path $workspaceRoot $relative))
    if (-not $oldDirectory.StartsWith($workspaceRoot.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Path outside workspace'
    }
    if ((Test-Path -LiteralPath $oldDirectory -PathType Container) -and
        @(Get-ChildItem -LiteralPath $oldDirectory -Force).Count -eq 0) {
        Remove-Item -LiteralPath $oldDirectory
    }
}
