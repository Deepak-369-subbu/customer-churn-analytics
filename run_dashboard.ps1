# Retail Banking Customer Churn Intelligence - PowerShell Launcher
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  Retail Banking Customer Churn Intelligence Platform     " -ForegroundColor Yellow
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host ""

# Check Python installation
$pythonPath = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $pythonPath) {
    Write-Host "[-] Python is not detected in your PATH. Please install Python 3.10+." -ForegroundColor Red
    Pause
    exit
}

Write-Host "[+] Python detected: $pythonPath" -ForegroundColor Green

# Check requirements
Write-Host "[*] Checking core dependencies..." -ForegroundColor Gray
python -c "import streamlit, plotly, sklearn, openpyxl, pandas" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[!] Installing missing dependencies from requirements.txt..." -ForegroundColor Yellow
    python -m pip install -r requirements.txt
}

Write-Host "[+] Launching Dashboard on http://localhost:8501..." -ForegroundColor Green
Write-Host "[*] Press Ctrl+C in this window anytime to stop the server." -ForegroundColor DarkGray
Write-Host ""

# Start Streamlit
streamlit run app.py
