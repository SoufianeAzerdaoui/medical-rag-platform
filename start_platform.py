#!/usr/bin/env python3
import subprocess
import time
import os
import sys
from pathlib import Path

def kill_process_on_port(port):
    try:
        # Find PID on port
        output = subprocess.check_output(["lsof", "-t", f"-i:{port}"]).decode().strip()
        if output:
            pids = output.split('\n')
            for pid in pids:
                if pid:
                    print(f"⚠️ Port {port} is occupied by PID {pid}. Killing it...")
                    subprocess.run(["kill", "-9", pid])
            time.sleep(1)
    except Exception:
        pass

def start():
    root = Path(__file__).parent.absolute()
    
    # 0. Clean ports
    kill_process_on_port(8000)
    kill_process_on_port(3000)

    # 1. Start FastAPI Backend
    print("🚀 Starting FastAPI Backend...")
    api_cmd = [".venv/bin/python3", "scripts/retrieval/api_server.py"]
    api_proc = subprocess.Popen(api_cmd, cwd=root)
    
    # Give it a moment to start
    time.sleep(2)
    
    # 2. Start Next.js Frontend
    print("🎨 Starting Web UI...")
    # Add node to path
    node_path = "/usr/local/Cellar/node@20/20.20.2/bin"
    env = os.environ.copy()
    env["PATH"] = f"{node_path}:{env.get('PATH', '')}"
    
    # Force Next.js to use port 3000
    ui_cmd = ["npm", "run", "dev", "--", "-p", "3000"]
    ui_proc = subprocess.Popen(ui_cmd, cwd=root / "web-ui", env=env)
    
    print("\n" + "="*40)
    print("🟢 Platform Ready!")
    print("Backend: http://localhost:8000")
    print("Frontend: http://localhost:3000")
    print("="*40 + "\n")
    
    try:
        while True:
            time.sleep(1)
            if api_proc.poll() is not None:
                print("❌ Backend stopped unexpectedly")
                break
            if ui_proc.poll() is not None:
                print("❌ Frontend stopped unexpectedly")
                break
    except KeyboardInterrupt:
        print("\n👋 Stopping platform...")
    finally:
        api_proc.terminate()
        ui_proc.terminate()

if __name__ == "__main__":
    start()
