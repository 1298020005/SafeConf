# Windows clipboard sync for Grok /copy bridge.
# Requires SSH LocalForward 18765 127.0.0.1:18765
# Uses Unicode clipboard (CF_UNICODETEXT), not raw bytes.

$ErrorActionPreference = "SilentlyContinue"
$uri = "http://127.0.0.1:18765/copy"
$lastHash = ""

Add-Type -AssemblyName System.Windows.Forms

function Set-UnicodeClipboard([string]$text) {
    # Prefer WinForms: reliable Unicode on Chinese Windows
    [System.Windows.Forms.Clipboard]::SetText($text)
}

Write-Host "clip_sync Unicode mode. Server /copy -> Windows clipboard. Ctrl+C to stop."

while ($true) {
    try {
        $wc = New-Object System.Net.WebClient
        $wc.Headers.Add("Cache-Control", "no-cache")
        $wc.Encoding = [System.Text.Encoding]::UTF8
        $bytes = $wc.DownloadData($uri)
        # Strip UTF-8 BOM if present
        if ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) {
            $text = [System.Text.Encoding]::UTF8.GetString($bytes, 3, $bytes.Length - 3)
        } else {
            $text = [System.Text.Encoding]::UTF8.GetString($bytes)
        }
        $hash = [System.BitConverter]::ToString([System.Security.Cryptography.SHA256]::Create().ComputeHash($bytes))
        if ($text -and $hash -ne $lastHash) {
            Set-UnicodeClipboard $text
            $lastHash = $hash
            $preview = if ($text.Length -gt 40) { $text.Substring(0, 40) + "..." } else { $text }
            $preview = $preview -replace "[\r\n]+", " "
            Write-Host ("[" + (Get-Date -Format "HH:mm:ss") + "] clipboard OK (" + $text.Length + " chars) " + $preview)
        }
    } catch {
        # Port forward down or server restarting
        Start-Sleep -Seconds 2
    }
    Start-Sleep -Milliseconds 500
}
