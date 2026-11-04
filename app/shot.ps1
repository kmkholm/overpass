# Launches dist\Ayna.exe, waits for its window, captures it to results\ayna_shot_<lang>.png, closes it.
param([string]$Out = "..\results\ayna_shot.png", [int]$Wait = 45)
Add-Type -AssemblyName System.Drawing
Add-Type @"
using System; using System.Runtime.InteropServices;
public class W32 { [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [StructLayout(LayoutKind.Sequential)] public struct RECT { public int L, T, R, B; } }
"@
$exe = Join-Path $PSScriptRoot "dist_v2\Ayna.exe"
$t0 = Get-Date
$p = Start-Process -FilePath $exe -PassThru
Start-Sleep -Seconds $Wait
# only the instance started here (the user may have another copy open)
$mine = Get-Process -Name Ayna -ErrorAction SilentlyContinue | Where-Object { $_.StartTime -ge $t0.AddSeconds(-2) }
$w = $mine | Where-Object { $_.MainWindowHandle -ne 0 } | Select-Object -First 1
if (-not $w) { "no window"; $mine | Stop-Process -Force -ErrorAction SilentlyContinue; exit 1 }
[W32]::SetForegroundWindow($w.MainWindowHandle) | Out-Null; Start-Sleep -Milliseconds 800
$r = New-Object W32+RECT; [W32]::GetWindowRect($w.MainWindowHandle, [ref]$r) | Out-Null
$bw = $r.R - $r.L; $bh = $r.B - $r.T
$bmp = New-Object System.Drawing.Bitmap $bw, $bh
$g = [System.Drawing.Graphics]::FromImage($bmp); $g.CopyFromScreen($r.L, $r.T, 0, 0, $bmp.Size); $g.Dispose()
$full = Join-Path $PSScriptRoot $Out
$bmp.Save($full, [System.Drawing.Imaging.ImageFormat]::Png); $bmp.Dispose()
"saved $full ($bw x $bh) title: $($w.MainWindowTitle)"
$mine | Stop-Process -Force -ErrorAction SilentlyContinue
