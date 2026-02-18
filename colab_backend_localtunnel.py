"""
COLAB BACKEND SETUP WITH LOCALTUNNEL
Paste this entire cell into Google Colab
"""

import subprocess
import time
import threading
import re

print("🔄 Setting up backend with Localtunnel...")
print("=" * 70)

# Step 1: Kill any existing processes
print("\n1️⃣ Cleaning up old processes...")
subprocess.run(['pkill', '-f', 'lt'], stderr=subprocess.DEVNULL)
subprocess.run(['pkill', '-f', 'flask'], stderr=subprocess.DEVNULL)
time.sleep(2)
print("   ✅ Cleanup done")

# Step 2: Install Node.js and Localtunnel
print("\n2️⃣ Installing Localtunnel...")
subprocess.run(['apt-get', 'update', '-qq'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.run(['apt-get', 'install', '-y', '-qq', 'nodejs', 'npm'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.run(['npm', 'install', '-g', 'localtunnel'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("   ✅ Localtunnel ready")

# Step 3: Start Flask backend
print("\n3️⃣ Starting Flask backend...")

def run_flask():
    """Run Flask in background thread"""
    subprocess.run(['python', 'backend/app_enhanced.py'])

flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()
time.sleep(10)  # Give Flask time to start
print("   ✅ Flask running on localhost:5000")

# Step 4: Start Localtunnel with FIXED subdomain
print("\n4️⃣ Starting Localtunnel with fixed URL...")
print("   ⏳ Waiting for tunnel...\n")

# Use FIXED subdomain - URL will always be https://ayurvedic-qa.loca.lt
tunnel_process = subprocess.Popen(
    ['lt', '--port', '5000', '--subdomain', 'ayurvedic-qa'],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    universal_newlines=True,
    bufsize=1
)

# Capture the URL
backend_url = None
for i in range(30):
    line = tunnel_process.stdout.readline()
    if line:
        print(f"   {line.strip()}")
        # Look for URL pattern: https://something.loca.lt
        match = re.search(r'(https://[a-z0-9-]+\.loca\.lt)', line)
        if match:
            backend_url = match.group(1)
            print(f"\n   🎉 FOUND IT: {backend_url}")
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
    print("=" * 70)
    
    # Keep the cell running
    print("\n🔄 Backend is running... (This cell will stay active)")
    print("   Press the ⏹️ Stop button to shut down the backend.\n")
    
    try:
        # Wait indefinitely
        tunnel_process.wait()
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down backend...")
        tunnel_process.kill()
        flask_thread.join(timeout=2)
        print("✅ Backend stopped")
else:
    print("❌ Could not get Localtunnel URL")
    print("=" * 70)
    print("\n💡 Try running this cell again!")
    print("=" * 70)
