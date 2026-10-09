#!/usr/bin/env python3
"""
Remote Bridge Master Launcher
Orchestrates:
1. [CONFIRMED LINK 1] Upload UI Portal: Web upload for files & folders -> [repo]/uploaded_stuff (via Cloudflare)
2. [CONFIRMED LINK 2] Standard Repo HTTP Server: python -m http.server 8000 (via Cloudflare)
3. [OPTIONAL LINK 3 ] Repo Inbuilt Frontend: Gradio/ComfyUI detected and exposed (Gradio online link or Cloudflare)
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

# Import our local components
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from repo_analyzer import RepoAnalyzer
from cloudflare_manager import CloudflareTunnelManager


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def find_available_port(start_port: int, max_attempts: int = 50) -> int:
    port = start_port
    while is_port_in_use(port) and max_attempts > 0:
        port += 1
        max_attempts -= 1
    return port


class RemoteBridgeLauncher:
    def __init__(self, repo_dir: str = ".", upload_port: int = 7865, http_port: int = 8000, start_frontend: bool = False):
        self.repo_dir = Path(repo_dir).resolve()
        self.upload_port = upload_port
        self.http_port = http_port
        self.start_frontend = start_frontend
        
        self.bridge_dir = self.repo_dir / ".remote_bridge"
        self.bridge_dir.mkdir(parents=True, exist_ok=True)
        
        self.cf_manager = CloudflareTunnelManager(bin_dir=str(self.bridge_dir / "bin"))
        self.subprocesses = []
        self.links = {
            "upload_portal": None,
            "repo_http_server": None,
            "inbuilt_frontend": None,
            "frontend_type": None
        }

    def run(self):
        print("=" * 78)
        print("                   REMOTE BRIDGE LAUNCHER & PROXY ENGINE")
        print(f"[*] Target Repository: {self.repo_dir}")
        print("=" * 78)

        # -------------------------------------------------------------
        # 1. Start Confirmed Link 1: Upload Portal
        # -------------------------------------------------------------
        actual_upload_port = find_available_port(self.upload_port)
        print(f"\n[1/3] Launching Confirmed Link 1: Upload Portal (Port {actual_upload_port})...")
        upload_script = SCRIPT_DIR / "upload_server.py"
        upload_proc = subprocess.Popen(
            [sys.executable, str(upload_script), "--repo", str(self.repo_dir), "--port", str(actual_upload_port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        self.subprocesses.append(upload_proc)
        time.sleep(1.2)

        print(f"[*] Establishing Cloudflare Tunnel for Upload Portal...")
        upload_tunnel_url = self.cf_manager.start_tunnel(actual_upload_port, label="upload_portal")
        self.links["upload_portal"] = {
            "local_port": actual_upload_port,
            "public_url": upload_tunnel_url,
            "target": str(self.repo_dir / "uploaded_stuff")
        }
        if upload_tunnel_url:
            print(f"[+] Confirmed Link 1 Active: {upload_tunnel_url}")
        else:
            print("[!] Warning: Cloudflare tunnel for Upload Portal failed to acquire URL.")

        # -------------------------------------------------------------
        # 2. Start Confirmed Link 2: Standard python -m http.server
        # -------------------------------------------------------------
        actual_http_port = find_available_port(self.http_port)
        print(f"\n[2/3] Launching Confirmed Link 2: Standard Repo HTTP Server (Port {actual_http_port})...")
        http_proc = subprocess.Popen(
            [sys.executable, "-m", "http.server", str(actual_http_port), "--directory", str(self.repo_dir)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        self.subprocesses.append(http_proc)
        time.sleep(1.2)

        print(f"[*] Establishing Cloudflare Tunnel for Repo HTTP Server...")
        http_tunnel_url = self.cf_manager.start_tunnel(actual_http_port, label="repo_http_server")
        self.links["repo_http_server"] = {
            "local_port": actual_http_port,
            "public_url": http_tunnel_url,
            "target": str(self.repo_dir)
        }
        if http_tunnel_url:
            print(f"[+] Confirmed Link 2 Active: {http_tunnel_url}")
        else:
            print("[!] Warning: Cloudflare tunnel for Repo HTTP Server failed to acquire URL.")

        # -------------------------------------------------------------
        # 3. Handle Optional Link 3: Repo Inbuilt Frontend
        # -------------------------------------------------------------
        print(f"\n[3/3] Analyzing Repository for Inbuilt Frontends (Gradio / ComfyUI / etc.)...")
        analyzer = RepoAnalyzer(str(self.repo_dir))
        analysis = analyzer.analyze()

        if analysis.get("has_frontend"):
            f_type = analysis.get("frontend_type", "unknown")
            f_entry = analysis.get("frontend_entrypoint")
            f_port = analysis.get("detected_port", 7860)
            self.links["frontend_type"] = f_type
            print(f"[+] Found Inbuilt Frontend: {f_type.upper()} ({f_entry}) on port {f_port}")

            # Check if port is already active
            already_active = is_port_in_use(f_port)
            if not already_active and self.start_frontend and f_entry:
                print(f"[*] Auto-launching frontend process: {sys.executable} {f_entry}...")
                fe_proc = subprocess.Popen(
                    [sys.executable, f_entry],
                    cwd=str(self.repo_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                self.subprocesses.append(fe_proc)
                time.sleep(3)

            # Establish tunnel to frontend port
            print(f"[*] Establishing Cloudflare Reverse Proxy for Frontend (Port {f_port})...")
            fe_tunnel_url = self.cf_manager.start_tunnel(f_port, label="inbuilt_frontend")
            self.links["inbuilt_frontend"] = {
                "type": f_type,
                "local_port": f_port,
                "public_url": fe_tunnel_url,
                "entrypoint": f_entry
            }
            if fe_tunnel_url:
                print(f"[+] Optional Link 3 Active: {fe_tunnel_url}")
            else:
                print(f"[*] Frontend tunnel opened (awaiting traffic once server starts on port {f_port}).")
        else:
            print("[*] No inbuilt frontend detected in this repository. Skipping Link 3.")
            self.links["inbuilt_frontend"] = None

        # -------------------------------------------------------------
        # 4. Save and Display Summary
        # -------------------------------------------------------------
        self._save_links_to_disk()
        self._print_summary_banner()

        print("\n[*] Remote Bridge is running. Press Ctrl+C to terminate all services.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.shutdown()

    def _save_links_to_disk(self):
        json_file = self.bridge_dir / "bridge_links.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(self.links, f, indent=2)

        md_file = self.bridge_dir / "bridge_links.md"
        with open(md_file, "w", encoding="utf-8") as f:
            f.write("# Active Remote Bridge Links\n\n")
            if self.links["upload_portal"] and self.links["upload_portal"]["public_url"]:
                f.write(f"- **Link 1 (Upload UI)**: {self.links['upload_portal']['public_url']} (Saves to `uploaded_stuff`)\n")
            if self.links["repo_http_server"] and self.links["repo_http_server"]["public_url"]:
                f.write(f"- **Link 2 (Repo HTTP)**: {self.links['repo_http_server']['public_url']} (Repo file browser)\n")
            if self.links["inbuilt_frontend"] and self.links["inbuilt_frontend"].get("public_url"):
                f.write(f"- **Link 3 (Frontend)**: {self.links['inbuilt_frontend']['public_url']} ({self.links['frontend_type']})\n")

    def _print_summary_banner(self):
        u_url = self.links["upload_portal"]["public_url"] if self.links["upload_portal"] else "N/A"
        h_url = self.links["repo_http_server"]["public_url"] if self.links["repo_http_server"] else "N/A"
        fe_info = self.links.get("inbuilt_frontend")
        fe_url = fe_info.get("public_url") if fe_info else "None detected"

        print("\n" + "=" * 78)
        print("                     ACTIVE REMOTE BRIDGE ACCESS LINKS")
        print("=" * 78)
        print(f"[LINK 1 - CONFIRMED] File & Folder Upload Portal:")
        print(f"                     URL   : {u_url}")
        print(f"                     Target: {self.repo_dir}/uploaded_stuff/")
        print("-" * 78)
        print(f"[LINK 2 - CONFIRMED] Standard Repo File Viewer (python -m http.server):")
        print(f"                     URL   : {h_url}")
        print(f"                     Target: {self.repo_dir}")
        print("-" * 78)
        print(f"[LINK 3 - OPTIONAL ] Repo Inbuilt Frontend ({self.links.get('frontend_type') or 'N/A'}):")
        print(f"                     URL   : {fe_url}")
        print("=" * 78)

    def shutdown(self):
        print("\n[*] Shutting down Remote Bridge services...")
        self.cf_manager.stop_all()
        for p in self.subprocesses:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    p.kill()
        print("[+] All services and tunnels cleanly stopped.")


def main():
    parser = argparse.ArgumentParser(description="Remote Bridge Launcher & Proxy Engine")
    parser.add_argument("--repo", "-r", type=str, default=".", help="Target repository directory path")
    parser.add_argument("--upload-port", type=int, default=7865, help="Upload Portal local port (default 7865)")
    parser.add_argument("--http-port", type=int, default=8000, help="Repo HTTP Server local port (default 8000)")
    parser.add_argument("--start-frontend", action="store_true", help="Automatically start detected frontend entrypoint")
    args = parser.parse_args()

    launcher = RemoteBridgeLauncher(
        repo_dir=args.repo,
        upload_port=args.upload_port,
        http_port=args.http_port,
        start_frontend=args.start_frontend
    )
    launcher.run()


if __name__ == "__main__":
    main()
