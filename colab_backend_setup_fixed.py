"""
SIMPLE COLAB BACKEND SETUP - Paste this entire cell into Google Colab
This will show you the Cloudflare Tunnel URL clearly
"""

import subprocess
import time
import re

print("🔄 Setting up backend with Cloudflare Tunnel...")
print("=" * 70)

# Step 1: Kill any existing processes
print("\n1️⃣ Cleaning up old processes...")
subprocess.run(['pkill', '-f', 'cloudflared'], stderr=subprocess.DEVNULL)
subprocess.run(['pkill', '-f', 'flask'], stderr=subprocess.DEVNULL)
time.sleep(2)
print("   ✅ Cleanup done")

# Step 2: Install cloudflared
print("\n2️⃣ Installing cloudflared...")
subprocess.run([
    'wget', '-q', 
    'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64',
    '-O', 'cloudflared'
], check=False)
subprocess.run(['chmod', '+x', 'cloudflared'], check=False)
subprocess.run(['mv', 'cloudflared', '/usr/local/bin/'], check=False)
print("   ✅ Cloudflared ready")

# Step 3: Start Flask in background
print("\n3️⃣ Starting Flask backend...")
flask_log = open('flask.log', 'w')
flask_process = subprocess.Popen(
    ['python', 'backend/app_enhanced.py'],
    stdout=flask_log,
    stderr=flask_log
)
time.sleep(8)  # Give Flask time to start
print("   ✅ Flask running on localhost:5000")

# Step 4: Start tunnel and CAPTURE THE URL
print("\n4️⃣ Starting Cloudflare Tunnel...")
print("   ⏳ Please wait 20-30 seconds for the URL...\n")

tunnel_process = subprocess.Popen(
    ['cloudflared', 'tunnel', '--url', 'http://localhost:5000'],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    universal_newlines=True,
    bufsize=1
)

# Read stderr line by line and find the URL
backend_url = None
all_output = []

for i in range(60):  # Try for 60 seconds
    line = tunnel_process.stderr.readline()
    if line:
        all_output.append(line.strip())
        
        # Look for URL pattern
        match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
        if match:
            backend_url = match.group(0)
            print(f"   🎉 FOUND IT: {backend_url}")
            break
    time.sleep(0.5)

print("\n" + "=" * 70)
if backend_url:
    print("✅ SUCCESS! YOUR BACKEND URL IS:")
    print("=" * 70)
    print(f"\n   🌐 {backend_url}\n")
    print("=" * 70)
    print("\n📋 COPY THIS LINE TO YOUR .env.local FILE:")
    print(f"\n   NEXT_PUBLIC_API_URL={backend_url}\n")
    print("=" * 70)
    print("\n⚠️  IMPORTANT: Keep this cell running!")
    print("   If you stop it, the URL will stop working.\n")
    print("=" * 70)
else:
    print("❌ Could not find tunnel URL")
    print("=" * 70)
    print("\n📄 Here's all the output I saw:")
    for line in all_output:
        print(f"   {line}")
    print("\n💡 Try running this cell again!")
    print("=" * 70)
