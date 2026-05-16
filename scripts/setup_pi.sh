#!/usr/bin/env bash
# One-time Pi 5 performance setup.
# Run once as root (or with sudo): sudo bash setup_pi.sh
set -euo pipefail

echo "=== ThreatDetect Pi 5 performance setup ==="

# ── 1. CPU performance governor ──────────────────────────────────────────────
echo "[1/4] Setting CPU governor to performance..."
apt-get install -y cpufrequtils >/dev/null 2>&1

for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance > "$cpu"
done

# Persist across reboots
cat > /etc/default/cpufrequtils <<'EOF'
GOVERNOR="performance"
EOF

# Also set via systemd in case cpufrequtils not installed
cat > /etc/systemd/system/cpu-performance.service <<'EOF'
[Unit]
Description=Set CPU governor to performance
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/bin/bash -c 'echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor'
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now cpu-performance.service

FREQ=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || echo "unknown")
echo "  governor=performance  freq=${FREQ}kHz"

# ── 2. zram ───────────────────────────────────────────────────────────────────
echo "[2/4] Configuring zram..."
apt-get install -y zram-tools >/dev/null 2>&1

cat > /etc/default/zramswap <<'EOF'
# zstd: best compression speed ratio for ARM
ALGO=zstd
# Use 50% of RAM as compressed swap (Pi 5 8GB → 4GB swap in ~1.3GB RAM)
PERCENT=50
PRIORITY=100
EOF

systemctl restart zramswap
sleep 1

# Disable SD card swap (slow, unnecessary with zram)
dphys-swapfile swapoff 2>/dev/null || true
systemctl disable dphys-swapfile 2>/dev/null || true

# Also kill any active file swap
swapoff -a 2>/dev/null || true
# Re-enable zram swap only
swapon --all 2>/dev/null || true

echo "  zram configured:"
swapon --show

# ── 3. Lock GPU memory split (Pi 5 has dedicated RAM, this is a no-op but safe) ─
echo "[3/4] GPU memory split check..."
GPU_MEM=$(vcgencmd get_mem gpu 2>/dev/null || echo "n/a")
echo "  gpu_mem=${GPU_MEM}"

# ── 4. Disable unneeded services to free RAM/CPU ─────────────────────────────
echo "[4/4] Disabling unneeded services..."
for svc in bluetooth hciuart avahi-daemon triggerhappy; do
    systemctl disable --now "$svc" 2>/dev/null && echo "  disabled $svc" || true
done

echo ""
echo "=== Setup complete. No reboot needed. ==="
echo ""
echo "Verify:"
echo "  cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor  # → performance"
echo "  swapon --show                                               # → zram0"
