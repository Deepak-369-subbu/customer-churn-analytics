import sys
import os
import subprocess
import webbrowser
import time

def main():
    print("=========================================================")
    print("  Retail Banking Customer Churn Intelligence Platform    ")
    print("=========================================================")
    print("\nStarting application...")
    
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        app_path,
        "--server.headless=true",
        "--browser.gatherUsageStats=false"
    ]
    
    # Launch Streamlit process
    process = subprocess.Popen(cmd)
    
    # Wait a moment for server to start, then open browser
    time.sleep(2.5)
    url = "http://localhost:8501"
    print(f"\nDashboard launched successfully! Access at: {url}")
    webbrowser.open(url)
    
    try:
        process.wait()
    except KeyboardInterrupt:
        print("\nStopping dashboard...")
        process.terminate()

if __name__ == "__main__":
    main()
