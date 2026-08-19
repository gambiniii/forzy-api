param()

$TaskName   = "ForzyCollector"
# Usa o caminho real do script atual para garantir a acentuacao correta
$ScriptFile = $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path (Split-Path $ScriptFile -Parent) -Parent
$BatchFile  = Join-Path $ProjectDir "scripts\run_collector.bat"

Write-Host "Projeto: $ProjectDir"
Write-Host "Batch:   $BatchFile"

if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Tarefa anterior removida."
}

$Action = New-ScheduledTaskAction `
    -Execute "cmd.exe" `
    -Argument "/c `"$BatchFile`"" `
    -WorkingDirectory $ProjectDir

$TriggerSeg = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday    -At "11:55"
$TriggerTer = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Tuesday   -At "11:55"
$TriggerQua = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Wednesday -At "11:55"

$Settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 3) `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable `
    -MultipleInstances IgnoreNew

$Principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Limited

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $TriggerSeg, $TriggerTer, $TriggerQua `
    -Settings $Settings `
    -Principal $Principal `
    -Description "Coleta API Forzy (WEG W22) -> PostgreSQL RDS. Seg-Qua 12h-14h BRT." `
    -Force

Write-Host ""
Write-Host "Tarefa ForzyCollector criada!" -ForegroundColor Green
Write-Host "Agenda: Segunda, Terca, Quarta - 11h55 (coleta 12h-14h BRT)"
Write-Host ""
Write-Host "Para verificar: schtasks /query /tn ForzyCollector /fo list /v"
Write-Host "Para testar agora: Start-ScheduledTask -TaskName ForzyCollector"
$today = Get-Date -Format "yyyyMMdd"
$logPath = Join-Path $ProjectDir "logs\collector_$today.log"
Write-Host "Para ver logs: Get-Content '$logPath' -Wait"
