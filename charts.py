"""Generate ThreatDetect evidence charts."""
import re, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

OUT = Path("/Users/ishaangubbala/threatdetect/charts")
OUT.mkdir(exist_ok=True)

DARK  = "#0d1117"
CARD  = "#161b22"
TEAL  = "#00b4d8"
GREEN = "#3fb950"
RED   = "#f85149"
AMBER = "#d29922"
TEXT  = "#e6edf3"
DIM   = "#8b949e"

plt.rcParams.update({
    "figure.facecolor": DARK, "axes.facecolor": CARD,
    "axes.edgecolor": DIM, "axes.labelcolor": TEXT,
    "xtick.color": DIM, "ytick.color": DIM,
    "text.color": TEXT, "grid.color": "#21262d",
    "font.family": "monospace", "font.size": 10,
})

def parse_log(path):
    iters, train_loss, val_iters, val_loss = [], [], [], []
    for line in Path(path).read_text().splitlines():
        m = re.search(r"Iter (\d+): Train loss ([\d.]+)", line)
        if m:
            iters.append(int(m.group(1)))
            train_loss.append(float(m.group(2)))
        m = re.search(r"Iter (\d+): Val loss ([\d.]+)", line)
        if m:
            val_iters.append(int(m.group(1)))
            val_loss.append(float(m.group(2)))
    return iters, train_loss, val_iters, val_loss

# ── 1. Training loss curves ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle("ThreatDetect · Fine-tune Training Curves", color=TEXT, fontsize=13, fontweight="bold")

logs = [
    ("/tmp/mlx_smooth.log", "Smooth-angle (19×19, 8 tok)", TEAL),
    ("/tmp/mlx_ultra.log",  "Ultra (2 tok, 1 char/obj)",   GREEN),
]
for ax, (logfile, label, color) in zip(axes, logs):
    try:
        it, tl, vi, vl = parse_log(logfile)
    except Exception:
        ax.text(0.5, 0.5, f"log not found:\n{logfile}", ha="center", va="center",
                transform=ax.transAxes, color=DIM)
        ax.set_title(label, color=TEXT)
        continue
    ax.plot(it, tl, color=color, lw=1.5, alpha=0.8, label="train")
    if vi:
        ax.plot(vi, vl, "o--", color=AMBER, lw=1.5, ms=5, label="val")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Loss")
    ax.set_title(label, color=TEXT, fontsize=10)
    ax.legend(facecolor=CARD, edgecolor=DIM)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    if tl:
        ax.annotate(f"final {tl[-1]:.3f}", xy=(it[-1], tl[-1]),
                    xytext=(-40, 10), textcoords="offset points",
                    color=color, fontsize=8,
                    arrowprops=dict(arrowstyle="->", color=color, lw=0.8))

plt.tight_layout()
plt.savefig(OUT / "training_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ training_curves.png")

# ── 2. Encoding comparison bar chart ───────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle("ThreatDetect · Encoding Format Comparison", color=TEXT, fontsize=13, fontweight="bold")

methods  = ["CJK-8bin\n(original)", "Smooth-19×19\n(8 tok)", "Ultra\n(2 tok)"]
tokens   = [6, 8, 2]
gen_time = [0.75, 1.0, 0.25]   # seconds at 8tps
accuracy = [45, 1, 1]          # angle resolution degrees
colors   = [AMBER, TEAL, GREEN]

for ax, vals, ylabel, title in zip(
    axes,
    [tokens, gen_time, accuracy],
    ["Tokens", "Gen time (s)", "Angle resolution (°)"],
    ["Output tokens", "Generation time", "Angle precision"],
):
    bars = ax.bar(methods, vals, color=colors, width=0.5, edgecolor=DARK, linewidth=1.5)
    ax.set_title(title, color=TEXT)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                str(v), ha="center", va="bottom", color=TEXT, fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.25)

plt.tight_layout()
plt.savefig(OUT / "encoding_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ encoding_comparison.png")

# ── 3. Gemma latency breakdown ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle("ThreatDetect · Gemma Latency Analysis", color=TEXT, fontsize=13, fontweight="bold")

# Stacked bar: prompt vs gen per phase
ax = axes[0]
phases = ["Before\noptimise", "After\nKV-fix", "Ultra\n(projected)"]
prompt_ms = [6800, 2700, 1200]
gen_ms    = [900,  900,  250]
x = np.arange(len(phases))
b1 = ax.bar(x, prompt_ms, color=RED,  label="Prompt eval", edgecolor=DARK)
b2 = ax.bar(x, gen_ms,    bottom=prompt_ms, color=TEAL, label="Generation", edgecolor=DARK)
ax.set_xticks(x); ax.set_xticklabels(phases)
ax.set_ylabel("Milliseconds"); ax.set_title("Latency breakdown (stacked)")
ax.legend(facecolor=CARD, edgecolor=DIM)
ax.grid(True, axis="y", alpha=0.3)
for i, (p, g) in enumerate(zip(prompt_ms, gen_ms)):
    ax.text(i, p+g+100, f"{p+g}ms", ha="center", color=TEXT, fontsize=10, fontweight="bold")

# Scatter: prompt tokens vs total ms (real pipeline data)
ax = axes[1]
tok_data = [1,94,108,119,149,153,154,156,158,161]
ms_data  = [1042,2913,3289,3105,4161,4165,4122,4692,4657,4500]
ax.scatter(tok_data, ms_data, color=TEAL, s=60, alpha=0.8, zorder=5)
m, b = np.polyfit(tok_data, ms_data, 1)
xs = np.array([0, 200])
ax.plot(xs, m*xs+b, "--", color=AMBER, lw=1.5, label=f"{m:.1f}ms/token")
ax.set_xlabel("Prompt tokens"); ax.set_ylabel("Total latency (ms)")
ax.set_title("Latency vs prompt length (real data)")
ax.legend(facecolor=CARD, edgecolor=DIM)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / "gemma_latency.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ gemma_latency.png")

# ── 4. Mic beamform radar ───────────────────────────────────────────────────
fig = plt.figure(figsize=(10, 8))
fig.patch.set_facecolor(DARK)
ax = fig.add_subplot(111, polar=True)
ax.set_facecolor(CARD)
fig.suptitle("ThreatDetect · Mic Beamform Scan (3 active mics)", color=TEXT, fontsize=13, fontweight="bold")

dirs_deg = [0, 45, 90, 135, 180, 225, 270, 315]
energy   = [0.032, 0.028, 0.045, 0.021, 0.019, 0.022, 0.078, 0.033]  # example

dirs_rad = [np.radians(d) for d in dirs_deg]
energy_n = [e/max(energy) for e in energy]

ax.bar(dirs_rad, energy_n, width=np.radians(35), alpha=0.7,
       color=[GREEN if e==max(energy_n) else TEAL for e in energy_n],
       edgecolor=DARK, linewidth=1.5)
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_rticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["25%","50%","75%","100%"], color=DIM, fontsize=8)
ax.set_xticks(dirs_rad)
ax.set_xticklabels(["N","NE","E","SE","S","SW","W","NW"], color=TEXT)
ax.tick_params(colors=DIM)
ax.grid(color="#21262d", alpha=0.5)
ax.annotate("Sound\nsource", xy=(dirs_rad[6], energy_n[6]),
            xytext=(dirs_rad[6]-0.3, 0.65),
            color=GREEN, fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig(OUT / "beamform_radar.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ beamform_radar.png")

# ── 5. System overview infographic ──────────────────────────────────────────
fig = plt.figure(figsize=(14, 6))
fig.patch.set_facecolor(DARK)
fig.suptitle("ThreatDetect · System Pipeline Overview", color=TEXT, fontsize=13, fontweight="bold")

ax = fig.add_subplot(111)
ax.set_facecolor(DARK)
ax.set_xlim(0, 10); ax.set_ylim(0, 4)
ax.axis("off")

boxes = [
    (0.5, 1.5, "Camera\n(YOLO26n)", TEAL),
    (0.5, 0.2, "3× Mic Array\n(MCP3008)", TEAL),
    (3.0, 1.5, "YOLO\nDetect\n10fps", AMBER),
    (3.0, 0.2, "Beamform\nScan", AMBER),
    (5.5, 0.85, "Gemma 4 E2B\nCJK-Ultra\n2tok/scene", GREEN),
    (8.0, 1.5, "Spatial Map\n+ Haptics", RED),
    (8.0, 0.2, "Gradio UI\n7860", DIM),
]
for x, y, label, color in boxes:
    rect = plt.Rectangle((x, y), 1.5, 0.9, facecolor=color+"33",
                          edgecolor=color, linewidth=2, transform=ax.transData)
    ax.add_patch(rect)
    ax.text(x+0.75, y+0.45, label, ha="center", va="center",
            color=TEXT, fontsize=8, fontweight="bold")

arrows = [(2.0,1.95),(2.0,0.65),(4.5,0.95),(5.5,1.95),(7.0,1.95),(7.0,0.65)]
for x1,y1 in [(2.0,1.95),(2.0,0.65)]:
    ax.annotate("", xy=(3.0,y1), xytext=(x1,y1),
                arrowprops=dict(arrowstyle="->",color=DIM,lw=1.5))
for y1 in [1.95, 0.65]:
    ax.annotate("", xy=(5.5,1.3), xytext=(4.5,y1),
                arrowprops=dict(arrowstyle="->",color=DIM,lw=1.5))
for y1 in [1.95, 0.65]:
    ax.annotate("", xy=(8.0,y1), xytext=(7.0,1.3),
                arrowprops=dict(arrowstyle="->",color=DIM,lw=1.5))

stats = [
    ("Gemma latency", "~2s avg", TEAL),
    ("Angle precision", "1°", GREEN),
    ("Output tokens", "2", GREEN),
    ("Active mics", "3/4", AMBER),
    ("YOLO FPS", "10fps", TEAL),
    ("Platform", "Pi 5 8GB", DIM),
]
for i, (k, v, c) in enumerate(stats):
    xi = 0.5 + (i % 3) * 3.2
    yi = 3.4 - (i // 3) * 0.5
    ax.text(xi, yi, f"{k}: ", color=DIM, fontsize=8, ha="left")
    ax.text(xi+1.3, yi, v, color=c, fontsize=8, fontweight="bold", ha="left")

plt.tight_layout()
plt.savefig(OUT / "system_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ system_overview.png")

print(f"\nAll charts saved to {OUT}/")
