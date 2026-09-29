"""Render the 'gsm8k: accuracy and latency (all budgets)' slide (budget x {vk, va}) from the grid JSONs.
usage: python make_gsm8k_slide.py <results_dir> <out_prefix>   -> <out_prefix>.pdf + <out_prefix>_<tag>.png
"""
import glob, json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

R, OUT = sys.argv[1], sys.argv[2]
rows = {os.path.basename(f)[7:-5]: json.load(open(f)) for f in glob.glob(f"{R}/gsm8k__*.json")}
base = rows["fullkv+cg"]; eager = rows["fullkv+eager"]
NAVY, GREEN, RED, GREY = "#1f3a5f", "#2e7d32", "#c62828", "#f2f0ea"
HDR = ["Budget", "Cross %", "vk acc (d)", "va acc (d)", "vk ttft (s)", "va ttft (s)", "vk e2e (s)", "va e2e (s)"]

def slide(pdf, tag, title_suffix, caption, box, budgets=None):
    have = {r["budget_tok"] for r in rows.values() if r.get("tag") == tag and r["config"][:2] in ("vk", "va")}
    budgets = sorted(have) if budgets is None else [b for b in budgets if b in have]
    cells, colors, bold = [], [], []
    for name, r in (("FullKV (eager)", eager), ("FullKV (cudagraph)", base)):
        cells.append([name, "0%", f"{r['acc']:.2f}", f"{r['acc']:.2f}", f"{r['ttft_s']:.0f}", f"{r['ttft_s']:.0f}", f"{r['e2e_s']:.0f}", f"{r['e2e_s']:.0f}"])
        colors.append(["black"] * 8); bold.append([False] * 8)
    for b in budgets:
        vk, va = rows.get(f"vk@{b}+{tag}"), rows.get(f"va@{b}+{tag}")
        acc = lambda r: f"{r['acc']:.1f} ({r['acc'] - base['acc']:+.1f})"
        pct = lambda r, k: f"{r[k]:.0f} ({100 * (r[k] - base[k]) / base[k]:+.0f}%)"
        cross = min(100, (va or vk)["cross_pct"])
        cells.append([f"{b:,}", f"{cross:.0f}%", acc(vk), acc(va),
                      pct(vk, "ttft_s"), pct(va, "ttft_s"), pct(vk, "e2e_s"), pct(va, "e2e_s")])
        win = va["acc"] >= vk["acc"]
        c = ["black", "black", RED if vk["acc"] < base["acc"] - 0.7 else GREEN, RED if va["acc"] < base["acc"] - 0.7 else GREEN,
             "black", "black", RED if vk["e2e_s"] > base["e2e_s"] * 1.08 else "black", RED if va["e2e_s"] > base["e2e_s"] * 1.08 else "black"]
        colors.append(c); bold.append([False, False, not win, win, False, False, vk["e2e_s"] > base["e2e_s"] * 1.08, va["e2e_s"] > base["e2e_s"] * 1.08])
    fig = plt.figure(figsize=(16, 9)); fig.patch.set_facecolor("white")
    fig.text(0.05, 0.93, f"gsm8k: accuracy and latency (all budgets){title_suffix}", fontsize=26, color="#333", va="top")
    fig.add_artist(plt.Line2D([0.05, 0.21], [0.865, 0.865], color="#8fb4d9", lw=3))
    fig.text(0.05, 0.84, caption, fontsize=12.5, color="#333", va="top", wrap=True)
    ax = fig.add_axes([0.05, 0.30, 0.90, 0.50]); ax.axis("off")
    t = ax.table(cellText=cells, colLabels=HDR, loc="upper center", cellLoc="center", colWidths=[0.13, 0.09, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13])
    t.auto_set_font_size(False); t.set_fontsize(11.5); t.scale(1, 2.2)
    for (i, j), cell in t.get_celld().items():
        cell.set_edgecolor("white")
        if i == 0:
            cell.set_facecolor(NAVY); cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor(GREY if i % 2 else "#faf9f6")
            cell.set_text_props(color=colors[i - 1][j], weight="bold" if bold[i - 1][j] else "normal")
    fig.add_artist(plt.Rectangle((0.05, 0.06), 0.90, 0.19, transform=fig.transFigure, color=NAVY))
    fig.text(0.50, 0.155, box, fontsize=12.5, color="white", ha="center", va="center", wrap=True, linespacing=1.5)
    pdf.savefig(fig); fig.savefig(f"{OUT}_{tag}{'_slidebudgets' if budgets and budgets[-1] > 1024 else ''}.png", dpi=110); plt.close(fig)

SLIDE_BUDGETS = [1024, 2048, 3072, 4096, 6144, 8192, 11264, 13664, 16384]  # the narrativeqa slide's rows
with PdfPages(f"{OUT}.pdf") as pdf:
    if all(f"va@{b}+pp" in rows for b in SLIDE_BUDGETS[1:]):
        slide(pdf, "pp", "",
              f"GSM8K exact match, FullKV {base['acc']:.2f} (cudagraph; deltas vs this row). Same budgets as the narrativeqa slide. 1,319 problems, 100 to 200-token\n"
              "prompts and about 450-token reasoning traces (2,048-token cap), so from 3,072 up no request can cross and every row is FullKV plus run-to-run noise.",
              "Mirror image of narrativeqa: there the 30k-token prompt made every request cross at every budget; here the whole request (prompt plus trace) is under\n"
              "2,300 tokens, so from 3,072 up cross % is 0 and nothing is evicted; those rows are pure run-to-run noise and calibrate it: 91.9 to 93.3 (batched greedy\n"
              "decoding is not bit-reproducible), so differences under about 1 point anywhere in this deck are noise. ttft/e2e are flat (+4 to 5% for value_attention is\n"
              "the scoring hook's cost, no eviction to pay it back). Only 1,024 and below evict; the budgets that separate the two metrics are on the next slide.",
              budgets=SLIDE_BUDGETS)
        rows_ok = True
    slide(pdf, "pp", " (256 to 1,024)",
          f"GSM8K exact match, FullKV {base['acc']:.2f} (cudagraph; deltas vs this row). 1,319 problems, 100 to 200-token prompts, about 450-token\n"
          "reasoning traces, so cross % falls with the budget. Both metrics prompt protected, evict every 16 steps, 128-token local window, piecewise CUDA graphs.",
          "Opposite regime to narrativeqa: prompts are tiny and the trace is the cache, so nothing evicts until the budget drops below the trace length.\n"
          "value_attention is lossless from 640 tokens up (93.3 at 640, 17% of KV reclaimed; 93.0 at 512, 26%) and loses only 1.7 at 384 with 39% reclaimed;\n"
          "vk_ratio needs 768 to be lossless and loses 5.3 at 384. ttft here is queueing (1,319 requests at once), so it tracks the batch's total work;\n"
          "e2e stays within 2 to 6% of FullKV down to 512 and only grows at 384 and 256, where traces lengthen. The load-bearing win is accuracy under eviction.",
          budgets=[256, 384, 512, 640, 768, 1024])
    slide(pdf, "pp_e64", " (256 to 1,024), evict every 64 steps",
          f"Same as the previous slide with SKIVE_EVICT_EVERY=64 SKIVE_EVICT_MARGIN=16: eviction in larger, less frequent rounds. FullKV {base['acc']:.2f}, deltas vs cudagraph.",
          "Slow cadence is the lever for budgets below the trace length: value_attention 91.9 at 384 (was 91.4) and 83.4 at 256 (was 77.2), vk_ratio 79.8 at 256 (was 74.8),\n"
          "with the same KV reclaimed. It also removes the latency overhead: value_attention e2e is within 1% of FullKV at every budget from 384 up\n"
          "(42 s vs 41 s), because requests run up to 16 blocks over budget between rounds and stop re-deriving the working context.",
          budgets=[256, 384, 512, 640, 768, 1024])
print("wrote", f"{OUT}.pdf")
