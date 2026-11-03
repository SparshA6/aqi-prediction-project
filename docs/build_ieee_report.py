"""Build the full project report in IEEE two-column format (Word, optionally PDF).

The section order follows the course's Report Structure and Guidelines. Tables and
figures are computed from the saved result files, so the numbers cannot drift from
them. Run src/analysis/gru_input_ablation.py and src/analysis/xgb_feature_importance.py
first (they write results/analysis/*.json). Then:

    python docs/build_ieee_report.py [--pdf]
"""

import argparse
import json
from pathlib import Path
import re
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = Path(__file__).with_name("AQI_Full_Project_Report_IEEE.docx")
FONT = "Times New Roman"
BLUE, GREY, RED = "#1f4e9c", "#7f7f7f", "#b03a2e"

plt.rcParams.update({"font.family": "serif", "font.serif": [FONT, "DejaVu Serif"], "font.size": 6.5,
                     "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "axes.spines.top": False, "axes.spines.right": False})


# ----------------------------------------------------------------- data
def analysis(name):
    path = RES / "analysis" / name
    if not path.exists():
        raise SystemExit(f"{path} is missing; run the scripts in src/analysis first")
    return json.loads(path.read_text())


def locked_runs():
    holdout = RES / "year_holdout/runs/2023-to-2024-v1"
    hm = pd.read_csv(holdout / "metrics.csv")
    hm = hm[(hm["mode"] == "same_day") & (hm.split == "test")].set_index("model")
    hman = json.loads((holdout / "manifest.json").read_text())["modes"]["same_day"]
    d25, d26 = RES / "forecast_2025/runs/locked-2025-v1", RES / "forecast_2026/runs/locked-2026-v1"
    return (hm, hman, pd.read_csv(d25 / "metrics.csv").set_index("model"), json.loads((d25 / "manifest.json").read_text()),
            pd.read_csv(d26 / "metrics.csv").set_index("model"), json.loads((d26 / "manifest.json").read_text()))


def gain_text(g):
    return f"{g['gain']:.2f} [{g['lower']:.2f}, {g['upper']:.2f}]"


def main_rows():
    """Rows of the main results table: model, task, rows, MAE, RMSE, R2, gain, coverage."""
    hm, hman, m25, man25, m26, man26 = locked_runs()
    xgb, gru26, ref26 = "XGBoost-change-forecast-weather-d5", "Frozen GRU-28days", "Persistence reference"
    def row(name, task, n, s, gain, cov):
        return [name, task, f"{int(n):,}", f"{s.MAE:.2f}", f"{s.RMSE:.3f}", f"{s.R2:.3f}", gain, cov]
    rows = [
        row("Baseline", "2024 same-day", hm.loc["Persistence", "n"], hm.loc["Persistence"], "n/a", "n/a"),
        row("Ridge-env", "2024 same-day", hm.loc["Ridge-env", "n"], hm.loc["Ridge-env"], "none", "n/a"),
        row("Ridge-history", "2024 same-day", hm.loc["Ridge-history", "n"], hm.loc["Ridge-history"], gain_text(hman["paired_rmse_gain"]), f"{hman['empirical_test_coverage'] * 100:.1f}%"),
        row("Baseline", "2025 next-day", m25.loc["Persistence reference", "n"], m25.loc["Persistence reference"], "n/a", "n/a"),
        row("GRU-28days", "2025 next-day", m25.iloc[0].n, m25.iloc[0], gain_text(man25["paired_rmse_gain"]), f"{man25['empirical_interval_coverage'] * 100:.1f}%"),
        row("Baseline", "2026 next-day", m26.loc[ref26, "n"], m26.loc[ref26], "n/a", "n/a"),
        row("Frozen GRU-28days", "2026 next-day", m26.loc[gru26, "n"], m26.loc[gru26], f"{m26.loc[ref26, 'RMSE'] - m26.loc[gru26, 'RMSE']:.2f}", "n/a"),
        row("Weather XGBoost", "2026 next-day", m26.loc[xgb, "n"], m26.loc[xgb], gain_text(man26["paired_rmse_gain"]), f"{man26['interval_coverage'] * 100:.1f}%"),
    ]
    assert rows[2][4] == "32.806" and rows[4][4] == "32.516" and rows[7][4] == "30.270" and rows[1][4] == "63.239", rows
    return rows


def selection_rows():
    d = pd.read_csv(RES / "forecast_development/runs/2025-ready-v1/development_metrics.csv")
    d = d[d.split == "selection"].sort_values("RMSE")
    inputs = {"short": "9", "long": "45", "delayed_weather": "57"}
    out = []
    for r in d.itertuples():
        family = r.model.split("-")[-1] if r.model != "Persistence" else None
        out.append([r.model.replace("delayed_weather", "delayed ERA5"), "yesterday only" if family is None else "28-day window" if family.startswith("28days") else inputs[family], f"{r.RMSE:.3f}"])
    n = int(d.n.iloc[0])
    return out[:10], n


def weather_dev_rows():
    d = pd.read_csv(RES / "forecast_development/runs/2026-ready-v1/development_metrics.csv")
    names = [("Persistence", "Baseline"), ("Frozen GRU-28days", "Frozen GRU"), ("XGBoost-change-history-d5", "XGBoost, history only"), ("XGBoost-change-forecast-weather-d5", "XGBoost, with weather forecast")]
    rows = []
    for model, label in names:
        s = d[d.model == model].set_index("split")
        rows.append([label, f"{s.loc['selection', 'RMSE']:.2f}", f"{s.loc['calibration', 'RMSE']:.2f}"])
    return rows


def aqi_bands():
    d = pd.read_csv(ROOT / "data/processed/multiyear_2023_2024_2025/aligned_dataset.csv", usecols=["AQI_eod", "aqi_flat_day"])
    d = d[d.AQI_eod.notna() & ~d.aqi_flat_day]
    labels = ["Good", "Satisfactory", "Moderate", "Poor", "Very poor", "Severe"]
    share = pd.cut(d.AQI_eod, bins=[-1, 50, 100, 200, 300, 400, 500], labels=labels).value_counts(normalize=True).reindex(labels) * 100
    return len(d), share, float(d.AQI_eod.mean()), float(d.AQI_eod.median()), float(d.AQI_eod.std())


def parse_results_report():
    text = (RES / "followthrough/oct12-v1/Results_Report.txt").read_text(encoding="utf-8")
    band = text.split("Error by observed AQI band:")[1].split("Error by actual daily change:")[0]
    change = text.split("Error by actual daily change:")[1].split("2. GRU-WEATHER ENSEMBLE")[0]
    pattern = r"^\s*(\S+|under 25|25-under 50|50 or more)\s+(Weather XGBoost|Frozen GRU|Persistence)\s+(\d+)\s+([\d.]+)\s+(-?[\d.]+)"
    def rows(block, groups):
        out = []
        for line in block.splitlines():
            m = re.match(pattern, line)
            if m and m.group(1) in groups:
                out.append(dict(group=m.group(1), model=m.group(2), n=int(m.group(3)), rmse=float(m.group(4)), bias=float(m.group(5))))
        assert len(out) == 3 * len(groups), (groups, len(out))
        return pd.DataFrame(out)
    return rows(band, ["0-50", "51-100", "101-200", "201-300", "301-400", "401-500"]), rows(change, ["under 25", "25-under 50", "50 or more"])


def faithfulness_rows():
    out = []
    for name, label in [("feature-20261009T192717Z-4f0ce5f2", "Feature (FA)"), ("spatial-20261009T194506Z-4b183585", "Spatial (S-SA)")]:
        c = pd.read_csv(RES / "faithfulness" / name / "correlations.csv").groupby(["baseline", "method"]).spearman.mean()
        for baseline, text in [("mean", "Training mean"), ("seasonal_donor", "Seasonal donor")]:
            out.append([label, text, f"{c[(baseline, 'attention')]:.3f}", f"{c[(baseline, 'SHAP')]:.3f}"])
    assert out[0][2:] == ["0.420", "0.664"] and out[2][2:] == ["0.242", "0.419"], out
    return out


# -------------------------------------------------------------- figures
def save(fig, path, tight=True):
    if tight:
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    else:
        fig.savefig(path, dpi=300)
    plt.close(fig)


def fig_pipeline(path):
    fig, ax = plt.subplots(figsize=(7.1, 1.55))
    ax.set_xlim(0, 100); ax.set_ylim(0, 22); ax.axis("off")
    steps = [("Raw data", "CPCB hourly AQI,\nsatellite, ERA5,\nGFS forecasts"), ("Preprocessing", "UTC alignment,\nend-of-day target,\nflags, z-scores"),
             ("Feature\nengineering", "28 AQI lags,\nsummaries, calendar,\nlocation, weather"), ("Chronological\nsplit", "fit, select,\ncalibrate, test\n(no overlap)"),
             ("Model\ntraining", "Ridge, GRU,\nXGBoost; attention\nvs uniform"), ("Evaluation\nand audit", "RMSE vs baseline,\npaired bootstrap,\nattention vs SHAP"),
             ("Dashboard\nand reports", "saved results,\nlocal forecast app,\nPDF reports")]
    w, gap = 12.6, 1.9
    for i, (title, body) in enumerate(steps):
        x = 0.4 + i * (w + gap)
        accent = i in (4, 5)
        ax.add_patch(FancyBboxPatch((x, 1.2), w, 19.6, boxstyle="round,pad=0.2,rounding_size=0.8", fc="#e8eef8" if accent else "white", ec=BLUE if accent else "#555555", lw=0.8 if accent else 0.6))
        ax.text(x + w / 2, 16.6, title, ha="center", va="center", fontsize=6.6, fontweight="bold", linespacing=1.1)
        ax.text(x + w / 2, 7.4, body, ha="center", va="center", fontsize=6.0, linespacing=1.2)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + w + gap - 0.15, 11), xytext=(x + w + 0.15, 11), arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#555555", shrinkA=0, shrinkB=0, mutation_scale=6))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    save(fig, path, tight=False)


def fig_importance(path, x):
    groups = list(x["test_2026"]["groups"])
    test = [x["test_2026"]["groups"][g]["shap_share"] * 100 for g in groups]
    dev = [x["development"]["groups"][g]["shap_share"] * 100 for g in groups]
    order = sorted(range(len(groups)), key=lambda i: test[i])
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for k, i in enumerate(order):
        ax.barh(k + 0.19, test[i], height=0.34, color=BLUE, label="Jan–Sep 2026 (test)" if k == 0 else None)
        ax.barh(k - 0.19, dev[i], height=0.34, color="#bdbdbd", label="Oct–Dec 2025 (development)" if k == 0 else None)
        ax.text(test[i] + 0.6, k + 0.19, f"{test[i]:.0f}%", va="center", fontsize=5.8)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([groups[i] for i in order])
    ax.set_xlabel("Share of mean absolute SHAP value (%)"); ax.set_xlim(0, 40)
    ax.legend(loc="lower right", frameon=False, fontsize=5.8, handlelength=1.1)
    save(fig, path)


def fig_city(path):
    g = pd.read_csv(RES / "forecast_2025/runs/locked-2025-v1/group_metrics.csv").query("group_type == 'city'")
    gru = g[g.model == "GRU-28days"].set_index("group").RMSE
    ref = g[g.model == "Persistence reference"].set_index("group").RMSE
    assert (gru < ref).all()
    order = gru.sort_values().index
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for i, city in enumerate(order):
        ax.plot([gru[city], ref[city]], [i, i], color="#999999", lw=1.0, zorder=1)
        ax.scatter(ref[city], i, s=16, facecolor="white", edgecolor=GREY, lw=0.9, zorder=2, label="Baseline" if i == 0 else None)
        ax.scatter(gru[city], i, s=16, color=BLUE, zorder=3, label="GRU-28days" if i == 0 else None)
        ax.text(ref[city] + 0.9, i, f"{ref[city] - gru[city]:.1f}", va="center", fontsize=6, color="#333333")
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order)
    ax.set_xlabel("RMSE (AQI points); number = gain"); ax.set_xlim(14, 50)
    ax.legend(loc="lower right", frameon=False, fontsize=6, handletextpad=0.2)
    save(fig, path)


def fig_bias(path, band):
    bands = ["0-50", "51-100", "101-200", "201-300", "301-400", "401-500"]
    style = {"Weather XGBoost": (BLUE, None), "Frozen GRU": ("#8c8c8c", None), "Persistence": ("#e3e3e3", "#8c8c8c")}
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    for k, (model, (fc, ec)) in enumerate(style.items()):
        d = band[band.model == model].set_index("group").loc[bands]
        ax.bar([i + (k - 1) * 0.27 for i in range(6)], d.bias, width=0.25, color=fc, edgecolor=ec or fc, lw=0.5, label="Baseline" if model == "Persistence" else model)
    ax.axhline(0, color="#333333", lw=0.6)
    ax.set_xticks(range(6)); ax.set_xticklabels(bands)
    ax.set_xlabel("Measured AQI band"); ax.set_ylabel("Mean error (AQI points)")
    ax.legend(loc="lower left", frameon=False, fontsize=6, handlelength=1.1)
    save(fig, path)


def fig_attention(path):
    g = pd.read_csv(RES / "attention_stability/gains.csv")
    protocols = [("chrono", "Chronological"), ("loco", "Held-out city"), ("season_cv", "Season block")]
    models = ["TA", "FA", "TA+FA", "S-TA", "S-SA", "S-TA+SA"]
    fig, axes = plt.subplots(3, 1, figsize=(3.4, 4.5), sharex=True)
    count = dict(helped=0, hurt=0, unclear=0)
    for ax, (protocol, name) in zip(axes, protocols):
        for r in g[g.protocol == protocol].itertuples():
            y = models.index(r.model) + (-0.17 if r.feature_set == "SW-w7" else 0.17)
            kind = "helped" if r.lower > 0 else "hurt" if r.upper < 0 else "unclear"
            count[kind] += 1
            color = {"helped": BLUE, "hurt": RED, "unclear": "#9a9a9a"}[kind]
            ax.plot([r.lower, r.upper], [y, y], color=color, lw=1.0)
            ax.scatter(r.gain, y, s=14, marker="D" if r.feature_set == "SW-w7" else "o", color=color, zorder=3)
        ax.axvline(0, color="#333333", lw=0.6)
        ax.set_yticks(range(6)); ax.set_yticklabels(models); ax.invert_yaxis()
        ax.set_title(name, loc="left", fontsize=7, fontweight="bold", pad=2)
    axes[-1].set_xlabel("RMSE gain (AQI points); right of 0 = attention better")
    handles = [plt.Line2D([], [], color=BLUE, lw=1, label=f"Helped ({count['helped']})"), plt.Line2D([], [], color=RED, lw=1, label=f"Hurt ({count['hurt']})"),
               plt.Line2D([], [], color="#9a9a9a", lw=1, label=f"Unclear ({count['unclear']})"),
               plt.Line2D([], [], color="#444444", marker="D", lw=0, ms=3.5, label="SW-w7"), plt.Line2D([], [], color="#444444", marker="o", lw=0, ms=3.5, label="SWT-w7")]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=5.8, handletextpad=0.25, columnspacing=0.8, handlelength=1.2)
    fig.subplots_adjust(top=0.9, hspace=0.35)
    assert (count["helped"], count["hurt"], count["unclear"]) == (11, 11, 14), count
    save(fig, path)


# ------------------------------------------------------------- document
class Report:
    def __init__(self):
        self.doc = Document()
        self.cites, self.h1, self.h2 = [], 0, 0
        self.figs = ["pipeline", "importance", "attention", "bias", "city"]
        self.tabs = ["literature", "gaps", "data", "features", "selection", "setup", "design", "results", "ablation", "ladder", "weatherdev", "faith", "sharp", "compare"]
        self.figs_done, self.tabs_done = [], []
        self.styles()
        self.page()

    # setup
    def font(self, style, size, bold=None, italic=None, small_caps=None):
        style.font.name = FONT
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor(0, 0, 0)
        if bold is not None: style.font.bold = bold
        if italic is not None: style.font.italic = italic
        if small_caps is not None: style.font.small_caps = small_caps
        fonts = style.element.get_or_add_rPr().get_or_add_rFonts()
        for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
            fonts.attrib.pop(qn(attr), None)
        for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
            fonts.set(qn(attr), FONT)

    def style(self, name, size, align=WD_ALIGN_PARAGRAPH.JUSTIFY, before=0, after=0, indent=0, base="Normal", **kw):
        s = self.doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
        s.base_style = self.doc.styles[base]
        self.font(s, size, **kw)
        f = s.paragraph_format
        f.alignment, f.space_before, f.space_after, f.first_line_indent = align, Pt(before), Pt(after), Inches(indent)
        f.line_spacing = 1.0
        return s

    def styles(self):
        normal = self.doc.styles["Normal"]
        self.font(normal, 10)
        normal.paragraph_format.space_after = Pt(0)
        self.style("IEEE Body", 10, indent=0.2)
        self.style("IEEE Abstract", 9, bold=True, indent=0.2)
        self.style("IEEE Table Text", 7.5, align=WD_ALIGN_PARAGRAPH.LEFT)
        self.style("IEEE Table Title", 8, align=WD_ALIGN_PARAGRAPH.CENTER, before=6, after=3, small_caps=True)
        self.style("IEEE Fig Caption", 8, before=2, after=6)
        self.style("IEEE Reference", 8, align=WD_ALIGN_PARAGRAPH.LEFT, indent=0, after=1)
        self.style("IEEE Title", 24, align=WD_ALIGN_PARAGRAPH.CENTER, after=8)
        self.style("IEEE Author", 9, align=WD_ALIGN_PARAGRAPH.CENTER)
        self.style("IEEE List", 10, after=1)
        self.style("IEEE Algorithm", 8, align=WD_ALIGN_PARAGRAPH.LEFT)
        h1, h2 = self.doc.styles["Heading 1"], self.doc.styles["Heading 2"]
        self.font(h1, 10, bold=False, italic=False, small_caps=True)
        self.font(h2, 10, bold=False, italic=True, small_caps=False)
        for h, before, after, align in ((h1, 10, 4, WD_ALIGN_PARAGRAPH.CENTER), (h2, 6, 2, WD_ALIGN_PARAGRAPH.LEFT)):
            f = h.paragraph_format
            f.space_before, f.space_after, f.alignment, f.keep_with_next, f.first_line_indent = Pt(before), Pt(after), align, True, Inches(0)

    def page(self):
        s = self.doc.sections[0]
        s.page_width, s.page_height = Inches(8.5), Inches(11)
        s.left_margin = s.right_margin = Inches(0.625)
        s.top_margin, s.bottom_margin = Inches(0.75), Inches(1.0)
        self.columns(s, 1)
        footer = s.footer.paragraphs[0]
        footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = footer.add_run()
        run.font.size, run.font.name = Pt(8), FONT
        for kind, text in (("begin", None), (None, "PAGE"), ("end", None)):
            if kind:
                el = OxmlElement("w:fldChar"); el.set(qn("w:fldCharType"), kind)
            else:
                el = OxmlElement("w:instrText"); el.set(qn("xml:space"), "preserve"); el.text = text
            run._r.append(el)

    @staticmethod
    def columns(section, n):
        cols = section._sectPr.find(qn("w:cols"))
        if cols is None:
            cols = OxmlElement("w:cols"); section._sectPr.append(cols)
        cols.set(qn("w:num"), str(n)); cols.set(qn("w:space"), "360")

    def section_break(self, columns):
        self.doc.add_section(WD_SECTION.CONTINUOUS)
        self.columns(self.doc.sections[-1], columns)
        carrier = self.doc.paragraphs[-1]          # the empty paragraph that holds the previous section's break
        carrier.paragraph_format.space_before = carrier.paragraph_format.space_after = Pt(0)
        carrier.paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY
        carrier.paragraph_format.line_spacing = Pt(1)

    # text
    @staticmethod
    def smart(text):
        out, open_quote = [], True
        for ch in text:
            if ch == '"':
                out.append("“" if open_quote else "”"); open_quote = not open_quote
            elif ch == "'":
                out.append("’")
            else:
                out.append(ch)
        return "".join(out)

    def tokens(self, text):
        def cite(m):
            numbers = []
            for key in re.findall(r"@(\w+)", m.group(1)):
                assert key in REFS, key
                if key not in self.cites: self.cites.append(key)
                numbers.append(self.cites.index(key) + 1)
            return ", ".join(f"[{n}]" for n in numbers)
        text = self.smart(text)
        text = re.sub(r"\[((?:@\w+(?:, ?)?)+)\]", cite, text)
        text = re.sub(r"\{fig:(\w+)\}", lambda m: f"Fig. {self.figs.index(m.group(1)) + 1}", text)
        return re.sub(r"\{tab:(\w+)\}", lambda m: f"Table {roman(self.tabs.index(m.group(1)) + 1)}", text)

    def runs(self, paragraph, text, size=None, bold=None):
        for part in re.split(r"(\*\*.+?\*\*|\*.+?\*)", self.tokens(text)):
            if not part: continue
            italic = part.startswith("*") and not part.startswith("**") and len(part) > 2
            run = paragraph.add_run(part[2:-2] if part.startswith("**") else part[1:-1] if italic else part)
            run.font.name = FONT
            if part.startswith("**") or bold: run.bold = True
            if italic: run.italic = True
            if size: run.font.size = Pt(size)
        return paragraph

    def para(self, text, style="IEEE Body"):
        return self.runs(self.doc.add_paragraph(style=style), text)

    def numbered(self, items):
        for i, text in enumerate(items, 1):
            p = self.doc.add_paragraph(style="IEEE List")
            p.paragraph_format.left_indent, p.paragraph_format.first_line_indent = Inches(0.25), Inches(-0.2)
            self.runs(p, f"{i}) {text}")

    def heading1(self, text):
        self.h1 += 1; self.h2 = 0
        self.doc.add_paragraph(f"{roman(self.h1)}. {text}", style="Heading 1")

    def heading2(self, text):
        self.h2 += 1
        self.doc.add_paragraph(f"{chr(64 + self.h2)}. {text}", style="Heading 2")

    def figure(self, key, path, caption, wide=False):
        assert self.figs[len(self.figs_done)] == key, key
        if wide: self.section_break(1)
        p = self.doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next, p.paragraph_format.space_before = True, Pt(4)
        p.add_run().add_picture(str(path), width=Inches(7.1 if wide else 3.4))
        c = self.doc.add_paragraph(style="IEEE Fig Caption")
        self.runs(c, f"Fig. {self.figs.index(key) + 1}. {caption}")
        c.alignment = WD_ALIGN_PARAGRAPH.CENTER if len(caption) < (140 if wide else 70) else WD_ALIGN_PARAGRAPH.JUSTIFY
        if wide: self.section_break(2)
        self.figs_done.append(key)

    def table(self, key, title, header, rows, widths, align=None, wide=False):
        assert self.tabs[len(self.tabs_done)] == key, key
        if wide: self.section_break(1)
        cap = self.doc.add_paragraph(style="IEEE Table Title")
        cap.paragraph_format.keep_with_next = True
        cap.add_run(f"TABLE {roman(self.tabs.index(key) + 1)}").font.name = FONT
        cap.add_run().add_break()
        r = cap.add_run(self.smart(title)); r.font.name = FONT
        t = self.doc.add_table(rows=1 + len(rows), cols=len(header))
        t.alignment, t.autofit = WD_TABLE_ALIGNMENT.CENTER, False
        align = align or ["l"] + ["c"] * (len(header) - 1)
        tbl_borders = OxmlElement("w:tblBorders")
        for side in ("top", "bottom"):
            e = OxmlElement(f"w:{side}"); e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "6"); e.set(qn("w:color"), "000000"); tbl_borders.append(e)
        t._tbl.tblPr.append(tbl_borders)
        for gc, w in zip(t._tbl.tblGrid.findall(qn("w:gridCol")), widths): gc.set(qn("w:w"), str(int(w * 1440)))
        for ri, row in enumerate([header] + rows):
            props = t.rows[ri]._tr.get_or_add_trPr(); props.append(OxmlElement("w:cantSplit"))
            for ci, text in enumerate(row):
                cell = t.cell(ri, ci); cell.width = Inches(widths[ci])
                para = cell.paragraphs[0]; para.style = self.doc.styles["IEEE Table Text"]
                para.alignment = {"l": WD_ALIGN_PARAGRAPH.LEFT, "c": WD_ALIGN_PARAGRAPH.CENTER, "r": WD_ALIGN_PARAGRAPH.RIGHT}[align[ci]]
                para.paragraph_format.keep_with_next = ri < (2 if wide else len(rows))
                self.runs(para, text, bold=ri == 0)
                if ri == 0: self.borders(cell, bottom=True)
                margins = OxmlElement("w:tcMar")
                for side in ("left", "right"):
                    m = OxmlElement(f"w:{side}"); m.set(qn("w:w"), "40"); m.set(qn("w:type"), "dxa"); margins.append(m)
                cell._tc.get_or_add_tcPr().append(margins)
        self.doc.add_paragraph().paragraph_format.space_after = Pt(2)
        if wide: self.section_break(2)
        self.tabs_done.append(key)

    @staticmethod
    def borders(cell, bottom):
        b = OxmlElement("w:tcBorders")
        if bottom:
            e = OxmlElement("w:bottom"); e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "6"); e.set(qn("w:color"), "000000"); b.append(e)
        cell._tc.get_or_add_tcPr().append(b)

    def algorithm(self, number, title, header_lines, lines):
        t = self.doc.add_table(rows=1, cols=1)
        t.alignment, t.autofit = WD_TABLE_ALIGNMENT.CENTER, False
        tbl_borders = OxmlElement("w:tblBorders")
        for side in ("top", "bottom"):
            e = OxmlElement(f"w:{side}"); e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "6"); e.set(qn("w:color"), "000000"); tbl_borders.append(e)
        t._tbl.tblPr.append(tbl_borders)
        t._tbl.tblGrid.findall(qn("w:gridCol"))[0].set(qn("w:w"), str(int(3.5 * 1440)))
        cell = t.cell(0, 0); cell.width = Inches(3.5)
        first = cell.paragraphs[0]; first.style = self.doc.styles["IEEE Algorithm"]
        self.runs(first, f"**Algorithm {number}.** {title}")
        first.paragraph_format.keep_with_next = True
        for line in header_lines:
            p = cell.add_paragraph(style="IEEE Algorithm"); p.paragraph_format.keep_with_next = True; self.runs(p, line)
        for i, (indent, text) in enumerate(lines, 1):
            p = cell.add_paragraph(style="IEEE Algorithm")
            p.paragraph_format.left_indent, p.paragraph_format.first_line_indent = Inches(0.22 + 0.14 * indent), Inches(-0.22)
            p.paragraph_format.keep_with_next = i < len(lines)
            self.runs(p, f"{i}:\t{text}")
            p.paragraph_format.tab_stops.add_tab_stop(Inches(0.22 + 0.14 * indent))
        self.doc.add_paragraph().paragraph_format.space_after = Pt(2)

    def title_block(self, title, authors):
        self.doc.add_paragraph(title, style="IEEE Title")
        t = self.doc.add_table(rows=2, cols=2)
        t.alignment, t.autofit = WD_TABLE_ALIGNMENT.CENTER, False
        for i, a in enumerate(authors):
            cell = t.cell(i // 2, i % 2); cell.width = Inches(3.625)
            first = cell.paragraphs[0]; first.style = self.doc.styles["IEEE Author"]
            r = first.add_run(a[0]); r.font.size, r.font.name = Pt(11), FONT
            first.paragraph_format.space_before = Pt(6)
            for line in a[1:]:
                p = cell.add_paragraph(style="IEEE Author"); r = p.add_run(line); r.italic = line.startswith("School") or line.startswith("Manip"); r.font.name = FONT
        self.doc.add_section(WD_SECTION.CONTINUOUS)
        self.columns(self.doc.sections[-1], 2)

    def references(self):
        self.doc.add_paragraph("References", style="Heading 1")
        for i, key in enumerate(self.cites, 1):
            p = self.doc.add_paragraph(style="IEEE Reference")
            p.paragraph_format.left_indent, p.paragraph_format.first_line_indent = Inches(0.3), Inches(-0.3)
            p.add_run(f"[{i}]\t").font.name = FONT
            self.runs(p, REFS[key])
            p.paragraph_format.tab_stops.add_tab_stop(Inches(0.3))
        unused = set(REFS) - set(self.cites)
        assert not unused, unused


def roman(n):
    return ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV", "XVI"][n]


REFS = {
    "who": 'World Health Organization, "Ambient (outdoor) air pollution," Fact sheet. [Online]. Available: https://www.who.int/news-room/fact-sheets/detail/ambient-(outdoor)-air-quality-and-health',
    "soga": 'State of Global Air, "State of Global Air Report 2024," Health Effects Institute, 2024. [Online]. Available: https://www.stateofglobalair.org/resources/report/state-global-air-report-2024',
    "pandey": 'India State-Level Disease Burden Initiative Air Pollution Collaborators, "Health and economic impact of air pollution in the states of India: The Global Burden of Disease Study 2019," *Lancet Planet. Health*, vol. 5, no. 1, pp. e25–e38, 2021, doi: 10.1016/S2542-5196(20)30298-9.',
    "cpcb": 'Central Pollution Control Board, "National Air Quality Index and AQI data repository." [Online]. Available: https://airquality.cpcb.gov.in/AQI_India/',
    "survey": 'AQI project, "Literature survey," docs/AQI_Literature_Survey_corrected.docx, Sep. 2026.',
    "ma": 'Z. Ma *et al.*, "Air pollutant prediction model based on transfer learning two-stage attention mechanism," *Sci. Rep.*, vol. 14, no. 1, Art. no. 7385, 2024, doi: 10.1038/s41598-024-57784-7.',
    "chen23": 'C. Chen *et al.*, "Prediction of pollutant concentration based on spatial–temporal attention, ResNet and ConvLSTM," *Sensors*, vol. 23, no. 21, Art. no. 8863, 2023, doi: 10.3390/s23218863.',
    "liu23": 'H. Liu, Q. Han, H. Sun, J. Sheng, and Z. Yang, "Spatiotemporal adaptive attention graph convolution network for city-level air quality prediction," *Sci. Rep.*, vol. 13, no. 1, Art. no. 13335, 2023, doi: 10.1038/s41598-023-39286-0.',
    "zhang": 'Z. Zhang, D. Schlesinger, C. Johansson, M. Engardt, and X. Ma, "X2-AQFormer: Unveiling dynamic drivers in multi-day hourly air pollution forecasting," *npj Clean Air*, vol. 2, no. 1, Art. no. 17, 2026, doi: 10.1038/s44407-026-00058-5.',
    "turbe": 'H. Turbé, M. Bjelogrlic, C. Lovis, and G. Mengaldo, "Evaluation of post-hoc interpretability methods in time-series classification," *Nature Mach. Intell.*, vol. 5, no. 3, pp. 250–260, 2023, doi: 10.1038/s42256-023-00620-w.',
    "simic": 'I. Šić, E. Veas, and V. Sabol, "A comprehensive analysis of perturbation methods in explainable AI feature attribution validation for neural time series classifiers," *Sci. Rep.*, vol. 15, no. 1, Art. no. 26607, 2025, doi: 10.1038/s41598-025-09538-2.',
    "li": 'L. Li *et al.*, "Improving air quality assessment using physics-inspired deep graph learning," *npj Clim. Atmos. Sci.*, vol. 6, no. 1, Art. no. 152, 2023, doi: 10.1038/s41612-023-00475-3.',
    "liu25": 'X. Liu, K. Su, S. Wang, and K. H. Ghazali, "Intelligent prediction of air quality index based on the Transformer-BiLSTM model," *Sci. Rep.*, vol. 15, no. 1, Art. no. 41838, 2025, doi: 10.1038/s41598-025-25865-w.',
    "poelzl": 'M. Poelzl, R. Kern, S. Kecorius, and M. Lovrić, "Exploration of transfer learning techniques for the prediction of PM10," *Sci. Rep.*, vol. 15, no. 1, Art. no. 2919, 2025, doi: 10.1038/s41598-025-86550-6.',
    "hertel": 'M. Hertel, S. Pütz, R. Mikut, V. Hagenmeyer, and B. Schäfer, "Explainable time-series forecasting with sampling-free SHAP for Transformers," *Nature Commun.*, vol. 17, no. 1, Art. no. 4723, 2026, doi: 10.1038/s41467-026-73243-5.',
    "jain": 'S. Jain and B. C. Wallace, "Attention is not explanation," in *Proc. NAACL-HLT*, 2019, doi: 10.18653/v1/N19-1357.',
    "wiegreffe": 'S. Wiegreffe and Y. Pinter, "Attention is not not explanation," in *Proc. EMNLP-IJCNLP*, 2019, doi: 10.18653/v1/D19-1002.',
    "bahdanau": 'D. Bahdanau, K. Cho, and Y. Bengio, "Neural machine translation by jointly learning to align and translate," in *Proc. ICLR*, 2015, arXiv:1409.0473.',
    "lundberg": 'S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Proc. NeurIPS*, 2017, arXiv:1705.07874.',
    "krishna": 'S. Krishna *et al.*, "The disagreement problem in explainable machine learning: A practitioner\'s perspective," arXiv:2202.01602, 2022.',
    "cho": 'K. Cho *et al.*, "Learning phrase representations using RNN encoder-decoder for statistical machine translation," in *Proc. EMNLP*, 2014, doi: 10.3115/v1/D14-1179.',
    "chen": 'T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining*, 2016, doi: 10.1145/2939672.2939785.',
    "veefkind": 'J. P. Veefkind *et al.*, "TROPOMI on the ESA Sentinel-5 Precursor: A GMES mission for global observations of the atmospheric composition for climate, air quality and ozone layer applications," *Remote Sens. Environ.*, vol. 120, pp. 70–83, 2012.',
    "modis": 'NASA LP DAAC, "MCD19A2.061 Terra and Aqua MAIAC land aerosol optical depth daily 1 km," Google Earth Engine Data Catalog. [Online]. Available: https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MCD19A2_GRANULES',
    "munoz": 'J. Muñoz-Sabater *et al.*, "ERA5-Land: A state-of-the-art global reanalysis dataset for land applications," *Earth Syst. Sci. Data*, vol. 13, no. 9, pp. 4349–4383, 2021.',
    "openmeteo": 'Open-Meteo, "Previous Runs API." [Online]. Available: https://open-meteo.com/en/docs/previous-runs-api',
    "paszke": 'A. Paszke *et al.*, "PyTorch: An imperative style, high-performance deep learning library," in *Proc. NeurIPS*, 2019.',
    "pedregosa": 'F. Pedregosa *et al.*, "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.',
    "locked": 'AQI project, "Locked evaluation records," results/year_holdout/runs/2023-to-2024-v1, results/forecast_2025/runs/locked-2025-v1 and results/forecast_2026/runs/locked-2026-v1 (metrics, manifests and verification records), 2026.',
    "stability": 'AQI project, "Attention versus uniform control across input sets and protocols," results/attention_stability/gains.csv, generated by src/models/report_attention_stability.py, Nov. 2026.',
    "status": 'AQI project, "Prospective tracking status," results/prospective/status.json, updated 2 Nov. 2026.',
}

ABSTRACT = ("Air quality forecasts let people and authorities act before pollution peaks, and attention-based models are often presented as showing which inputs drive those forecasts, "
            "yet this is rarely tested. This report studies daily station-level Air Quality Index (AQI) at 107 CPCB stations in eight Indian cities from 2023 to 2026. "
            "Station records are combined with satellite, reanalysis and archived forecast-weather data, ridge, GRU and XGBoost models are evaluated in locked year-level tests, and attention is audited against SHAP using input-replacement sensitivity. "
            "Every model is compared with a baseline that uses the previous day's AQI. The 28-day GRU lowered RMSE from 35.388 to 32.516 in 2025, and a weather-assisted XGBoost lowered it from 34.668 to 30.270 in 2026, with the weather forecast supplying much of that gain. "
            "Across 36 comparisons attention helped in 11, hurt in 11 and was inconclusive in 14, and attention agreed less than SHAP with measured sensitivity in all four audits. "
            "Attention maps therefore need checking before they are read as explanations. A final out-of-time evaluation opens in January 2027.")


def build(pdf=False):
    words = len(ABSTRACT.split())
    assert 150 <= words <= 200, words
    g, x = analysis("gru_input_ablation.json"), analysis("xgb_feature_importance.json")
    band, change = parse_results_report()
    n_aqi, share, mean_aqi, median_aqi, sd_aqi = aqi_bands()
    selection, n_sel = selection_rows()
    pct = lambda v, d=0: f"{v * 100:.{d}f}%"
    V, L = g["variants"], g["linear"]
    xt, xd = x["test_2026"], x["development"]
    yd = "Yesterday's AQI"
    r = Report()
    figs = Path(tempfile.mkdtemp())
    fig_pipeline(figs / "f1.png"); fig_importance(figs / "f2.png", x); fig_attention(figs / "f3.png"); fig_bias(figs / "f4.png", band); fig_city(figs / "f5.png")

    r.title_block("Air Quality Index Prediction and Attention Faithfulness Across Eight Indian Cities", [
        ["Saanvi Garg", "Section CSE-C", "Reg. No. 240905702, Roll No. 68", "School of Computer Engineering", "Manipal Institute of Technology", "Manipal Academy of Higher Education", "Manipal, Karnataka, India"],
        ["Sparsh Agrawal", "Section CSE-C", "Reg. No. 240905708, Roll No. 70", "School of Computer Engineering", "Manipal Institute of Technology", "Manipal Academy of Higher Education", "Manipal, Karnataka, India"],
        ["Dhruv Chamria", "Section CSE-C", "Reg. No. 240905706, Roll No. 69", "School of Computer Engineering", "Manipal Institute of Technology", "Manipal Academy of Higher Education", "Manipal, Karnataka, India"],
        ["Neelaksha Sisodiya", "Section CSE-C", "Reg. No. 240905642, Roll No. 65", "School of Computer Engineering", "Manipal Institute of Technology", "Manipal Academy of Higher Education", "Manipal, Karnataka, India"]])
    p = r.doc.add_paragraph(style="IEEE Abstract")
    p.add_run("Abstract—").italic = True
    r.runs(p, ABSTRACT, size=9, bold=True)
    p = r.doc.add_paragraph(style="IEEE Abstract")
    p.paragraph_format.space_after = Pt(4)
    p.add_run("Index Terms—").italic = True
    r.runs(p, "air quality index, attention faithfulness, explainable AI, time-series forecasting, GRU, XGBoost, SHAP, cross-city transfer", size=9, bold=True)

    # ------------------------------------------------------------------ I
    r.heading1("Introduction")
    r.heading2("Background and importance")
    r.para("Air pollution is one of the largest environmental risks to human health. The World Health Organization estimated that ambient air pollution caused 4.2 million premature deaths worldwide in 2019, that 99% of the world's population lived where its air quality guidelines were not met, and that 89% of those deaths occurred in low- and middle-income countries [@who]. The State of Global Air report attributed 8.1 million deaths worldwide to air pollution in 2021 [@soga]. India carries a large share of this burden: the Global Burden of Disease study for Indian states attributed 1.67 million deaths in 2019, 17.8% of all deaths in the country, to air pollution, with an economic loss of US$36.8 billion, or 1.36% of gross domestic product [@pandey].")
    r.heading2("Problem context and practical significance")
    r.para(f"India reports air quality through the Air Quality Index (AQI) computed from Central Pollution Control Board (CPCB) monitoring stations [@cpcb]. In this project's data, {share['Poor'] + share['Very poor'] + share['Severe']:.1f}% of station-days from 2023 to 2025 fell in the Poor, Very poor or Severe bands, so roughly one station-day in five was unhealthy. A forecast issued a day ahead lets residents and authorities act before a pollution peak, but it is useful only if it beats the simplest alternative, which is to assume that tomorrow equals today, and only if users can trust what the model is said to rely on.")
    r.para("Two pain points follow. First, tomorrow's AQI is strongly related to today's, so a model can look accurate while adding little; in this study the best AQI-history forecaster's output correlated 0.989 with yesterday's value. Second, attention models are routinely explained by showing their attention weights, but it has not been established that those weights reflect what drives the output of an AQI regression model.")
    r.heading2("Research gap")
    r.para("The critical gap is that attention is adopted as an explanation without a fixed-model test. The reviewed air-quality studies report predictive gains and component ablations for attention [@ma, @chen23, @liu23] and compare explanation methods on classification data or a single monitoring network [@turbe, @simic, @zhang], but within the bounds of our survey [@survey] none combines a fixed-model attention faithfulness test, grouped realistic perturbations and consistency across cities for station-level multimodal AQI regression. The gap matters because accuracy cannot reveal a misleading explanation: a user may credit wind or satellite signals to a model that ignores them. The debate over attention as explanation arose in text classification [@jain, @wiegreffe], and its answer for continuous, correlated environmental inputs is unknown. A second roadblock is evaluation practice, since the reviewed studies differ in splits, targets and horizons, and in one the independence of the split from preprocessing is unverified [@liu25].")
    r.heading2("Research questions")
    r.numbered(["How much do a history-based model and a weather-assisted model improve next-day station AQI forecasts over the previous-day baseline in locked chronological tests?",
                "Does attention improve accuracy over uniform-weight controls, and does the answer hold across input sets and test designs?",
                "Do attention rankings agree with measured input-replacement sensitivity as well as SHAP rankings do?",
                "Which inputs account for the forecast skill beyond the baseline?"])
    r.heading2("Research objectives")
    r.numbered(["To build, by September 2026, an aligned daily dataset for 107 CPCB stations in eight cities with a verified end-of-day AQI target, satellite and weather inputs and documented quality flags.",
                "To evaluate ridge, GRU and XGBoost forecasters in three locked tests (2024, 2025 and January–September 2026), each scored once against the baseline with 95% paired block-bootstrap intervals.",
                "To compare 36 attention variants with uniform controls on two input sets under chronological, leave-one-city-out and season-block designs.",
                "To audit attention and SHAP rankings against replacement sensitivity on 300 sampled rows per checkpoint.",
                "To quantify, with input ablations and importance scores, how much forecast skill comes from yesterday's AQI, older history, calendar and location, and weather forecasts.",
                "To publish the saved results in a dashboard that separates same-day estimates, backtests and the pending out-of-time evaluation."])
    r.heading2("Research contribution")
    r.para("This work makes four contributions. (1) It provides three locked, independently verified forecast evaluations against an explicit previous-day baseline across eight Indian cities. (2) It gives paired evidence on 36 attention variants showing that the accuracy effect of attention depends on the input set and test design. (3) It audits attention against SHAP with replacement sensitivity and finds lower agreement for attention in every sampled audit. (4) It shows how little of the forecast skill comes from AQI history beyond yesterday's value and how much comes from weather forecasts, with a reproducible code and dashboard package.")
    r.heading2("Paper roadmap")
    r.para("Section II reviews related work and states the research gap. Section III presents the methodology. Section IV reports results and discussion, Section V concludes, Section VI describes the study limitations and Section VII suggests future scope.")

    # ----------------------------------------------------------------- II
    r.heading1("Related Work")
    r.heading2("Classification of approaches")
    r.para("The ten reviewed studies [@survey] fall into three overlapping groups, and the attention debate and post-hoc explanation methods form their background. Attention was introduced as a soft alignment mechanism [@bahdanau], and it was later shown that attention weights can be altered without changing predictions [@jain], although the conclusion depends on how an explanation is defined and tested [@wiegreffe]. Post-hoc attribution such as SHAP assigns each input a contribution to a prediction [@lundberg], but explanation methods can disagree on the same model [@krishna].")
    r.para("*Explanation evaluation and attribution* studies tested explanations against prediction changes [@turbe, @simic] or made grouped attribution efficient [@hertel]. *Attention-based prediction and spatial assessment* studies used attention for pollutant or AQI prediction [@li, @zhang, @liu25, @chen23, @liu23]. *Transfer learning and spatial generalization* studies examined prediction at sites with little data [@ma, @poelzl]. Scores in these papers use different targets, horizons, data and splits, so they cannot be ranked against one another, and AQI differs from PM2.5 and PM10 concentrations.")
    r.table("literature", "Comparative summary of the reviewed studies", ["Study", "Year", "Method", "Key contribution", "Study limitation"], [
        ["Turbé et al. [@turbe]", "2023", "Compared six attribution methods on CNN, BiLSTM and Transformer classifiers with corrupted blocks", "Corruption-based audit that ranks attribution methods", "Transfer to regression remains unvalidated."],
        ["Šić et al. [@simic]", "2025", "Tested 23 perturbation methods and 12 attribution methods on five UCR datasets", "Showed that attribution rankings depend on the perturbation design", "The metrics are limited to classification probabilities."],
        ["Ma et al. [@ma]", "2024", "Pretrained an adaptive bidirectional GRU with temporal and external attention on nine Beijing stations, then fine-tuned on the target site", "Two-stage attention for sites with little data", "Adaptation requires target labels."],
        ["Li et al. [@li]", "2023", "Trained a physics-inspired graph network with attention on 1,604 stations in China", "Combined transport physics with learned spatial relations (site-based PM10 R² about 0.85)", "The physical meaning of the learned coefficients is limited."],
        ["Zhang et al. [@zhang]", "2026", "Trained an explainable Transformer (X2-AQFormer) for NOₓ and PM10 in Stockholm up to 72 hours ahead", "Intrinsic explanations compared with GradientSHAP", "The evaluation is confined to the Stockholm network."],
        ["Liu et al. [@liu25]", "2025", "Combined a Transformer and a BiLSTM to predict daily AQI in three northern Chinese cities", "Daily AQI forecasting with SHAP feature analysis", "The preprocessing description does not establish that the split is independent of fitting."],
        ["Poelzl et al. [@poelzl]", "2025", "Compared predictors, generalization and adaptation (TrAdaBoostR2) for PM10 between Graz and Zagreb", "Separated generalization from label-assisted adaptation", "The experiment covers two cities and excludes exceptional events."],
        ["Hertel et al. [@hertel]", "2026", "Introduced SHAPformer, a masked-attention Transformer, on electricity data", "Sampling-free SHAP for grouped inputs", "The cost grows exponentially with the number of feature groups."],
        ["Chen et al. [@chen23]", "2023", "Proposed STA-ResConvLSTM for PM2.5 around Beijing", "Spatial–temporal attention in residual blocks", "The model is limited by abrupt changes and by horizons beyond 48 hours."],
        ["Liu et al. [@liu23]", "2023", "Developed STAA-GCN for PM2.5 on Beijing, Tianjin and London datasets", "Adaptive graph attention over station relations", "Prediction in unmonitored areas is outside the evaluation."],
    ], [1.1, 0.4, 2.35, 1.7, 1.7], ["l", "c", "l", "l", "l"], wide=True)
    r.heading2("Gap synthesis")
    r.para("The reviewed literature shows that attention is widely used for air-quality prediction, but the evidence for it comes in three kinds that are rarely combined: predictive gains and component ablations [@ma, @chen23, @liu23]; comparisons of explanation methods on classification data [@turbe, @simic] or one monitoring network [@zhang]; and transfer experiments that mix zero-shot, station-holdout and label-assisted settings [@ma, @li, @liu25, @poelzl, @liu23]. This leaves four gaps. The methodological gap is that no study in the reviewed set tests attention against input-replacement sensitivity on a fixed model for station-level multimodal AQI regression. The knowledge gap is that the attention-as-explanation debate has not been applied to correlated environmental inputs. The population gap is that each study covers one or two cities, so consistency across regimes is unknown. The empirical gap is that conflicting views on attention faithfulness have no resolution for this task. This is a bounded finding of our survey and not proof that no such study exists. This study addresses these gaps by pairing locked forecast evaluations with fixed-model audits in eight cities ({tab:gaps}).")
    r.table("gaps", "Mapping of research gaps to objectives and contributions", ["Gap type", "Addressed by", "Objective", "Section"], [
        ["Methodological", "Fixed-model replacement audit of attention versus SHAP", "4", "III-H, IV-E"],
        ["Knowledge", "36 paired attention-versus-uniform comparisons on environmental inputs", "3", "IV-D"],
        ["Population", "Eight cities, leave-one-city-out and season-block designs", "1, 3", "III-A, IV-D"],
        ["Empirical", "Baseline comparison and input ablations on locked tests", "2, 5", "IV-A to IV-C"],
    ], [0.85, 1.5, 0.5, 0.65], ["l", "l", "c", "l"])

    # ---------------------------------------------------------------- III
    r.heading1("Methodology")
    r.heading2("Dataset description")
    r.para(f"Hourly station AQI came from CPCB exports [@cpcb]. Satellite inputs were Sentinel-5P NO₂ and HCHO [@veefkind] and MODIS aerosol optical depth [@modis], and weather inputs were ERA5-Land daily reanalysis [@munoz], all extracted through Google Earth Engine on a 0.03° grid. The 2026 weather model also used GFS forecasts issued 24 hours before each valid hour, taken from the Open-Meteo Previous Runs API [@openmeteo]. CPCB is the national agency for the official index, and the satellite and reanalysis products cover every city, which made one consistent pipeline possible. The 107 stations are in Delhi (39), Hyderabad (14), Mumbai (14), Bengaluru (13), Chennai (8), Kolkata (7), Jaipur (6) and Lucknow (6), covering plains, coastal, upland, semi-arid and delta climates ({{tab:data}}). A further 20,086 station-days from January to September 2026 at 91 stations form the weather test.")
    r.table("data", "Aligned data by year", ["Year", "Station-days", "With AQI", "Stations"],
            [["2023", "31,813", "30,081", "107"], ["2024", "33,860", "31,722", "106"], ["2025", "34,189", "32,842", "107"], ["All", "99,862", "94,645", "107"]], [0.6, 1.0, 1.0, 0.9])
    r.para(f"The target is skewed. Of {n_aqi:,} usable station-days, {share['Good']:.1f}% were Good (AQI 0–50), {share['Satisfactory']:.1f}% Satisfactory (51–100), {share['Moderate']:.1f}% Moderate (101–200), {share['Poor']:.1f}% Poor (201–300), {share['Very poor']:.1f}% Very poor (301–400) and {share['Severe']:.1f}% Severe (401–500). The mean was {mean_aqi:.1f}, the median {median_aqi:.0f} and the standard deviation {sd_aqi:.1f}.")
    r.heading2("Preprocessing")
    r.numbered(["*Time alignment.* CPCB's hourly AQI is a trailing 24-hour average, so a daily mean is blurred across two days: its correlation with the previous day's satellite NO₂ (0.217) nearly matched that with the same day's (0.230). The target is *AQI_eod*, the hourly AQI at 05:00 IST on the following day, whose window covers almost exactly the UTC day described by the inputs. All dates are UTC.",
                "*Cleaning.* Days on which AQI stayed constant below the 500 cap were flagged as stuck-sensor readings (906 rows) and excluded. Missing targets were never imputed. An independent audit reproduced all 31,813 aligned 2023 rows from the raw files with no mismatches.",
                "*Missing inputs.* Satellite values are kept as missing under cloud. Tree models read missing values natively; ridge used median imputation with missing indicators; the GRU received a validity channel and zeros after normalization.",
                "*Normalization.* Inputs were standardized with the training-window mean and standard deviation (z-score), which suits gradient-trained networks and keeps the scale of each input comparable. The GRU target was standardized the same way.",
                "*Encoding.* Day of year was encoded cyclically as sine and cosine so that 31 December and 1 January are close. City and station identity are not inputs; location enters only as latitude and longitude.",
                "*Balancing.* No resampling was used because AQI is continuous and resampling would distort the target. The effect of the skew is instead exposed by reporting errors per AQI band."])
    r.heading2("Feature engineering and selection")
    r.para("{tab:features} lists the input groups. For next-day forecasts every input for target day D uses only observations from days up to D−1, and weather forecasts were issued before D began. Three feature families were compared for the tree and linear models: a short family of 9 inputs, a long family of 45 and a delayed-weather family that adds ERA5 values lagged 7 and 14 days (57 inputs). There was no importance-based pruning; the family and model were chosen by lowest RMSE on October 2024, a month held out from fitting ({tab:selection}).")
    r.table("features", "Input groups", ["Group", "Inputs", "Count", "Used by"], [
        ["Past AQI", "Previous 28 daily AQI values", "28", "GRU, XGBoost, ridge"],
        ["AQI summaries", "Mean, spread and valid-day count over 3, 7, 14, 28 days; change from the day before", "13", "XGBoost, ridge"],
        ["Calendar", "Day of year (sine, cosine)", "2", "All models"],
        ["Location", "Station latitude, longitude", "2", "All models"],
        ["Forecast weather", "GFS temperature, humidity, wind speed, rain, pressure, issued 24 h ahead", "5", "Weather XGBoost"],
        ["Satellite and ERA5", "NO₂, HCHO, aerosol; temperature, dew point, wind, pressure, rain (same day)", "n/a", "Same-day ridge, attention study"],
    ], [0.8, 1.5, 0.4, 0.8], ["l", "l", "c", "l"])
    r.table("selection", f"Model selection on October 2024 ({n_sel:,} rows)", ["Candidate", "Inputs", "RMSE"], selection, [2.0, 0.9, 0.6], ["l", "c", "c"])
    r.heading2("Algorithm selection and justification")
    r.para("The baseline repeats the previous day's AQI, which every candidate must beat. Ridge regression is a transparent linear reference. XGBoost [@chen] fits non-linear trees, handles missing values and was trained on the change from yesterday so that the baseline is its default answer. The GRU [@cho] is a recurrent network that reads a 28-day window and can use temporal patterns that lag features capture only partly. A three-seed GRU ensemble lowers seed variance. In {tab:selection} the GRU had the lowest RMSE on the selection month, ahead of all tree and linear candidates, and it was frozen for the 2025 test. For 2026 a second candidate added archived weather forecasts to XGBoost. The attention study used GRU encoders with temporal, feature or spatial attention because these are the mechanisms whose weights are read as explanations.")
    r.heading2("System workflow")
    r.para("{fig:pipeline} shows the workflow from raw data to dashboard. The chronological split is the central safeguard: the model is fitted, selected and calibrated before the test period is scored once.")
    r.figure("pipeline", figs / "f1.png", "System workflow from raw data to dashboard. The shaded stages are the study's two main experiments.", wide=True)
    r.heading2("Experimental setup")
    r.table("setup", "Hardware, software and parameters", ["Item", "Value"], [
        ["Computer", "AMD Ryzen 7 7840HS (8 cores, 16 threads), 15.3 GB RAM, Windows 11 Home"],
        ["GPU", "NVIDIA GeForce RTX 4060 Laptop, 8 GB, driver 592.82, CUDA 12.8"],
        ["Software", "Python 3.13.3; PyTorch 2.10.0+cu128 [@paszke]; XGBoost 3.4.1; scikit-learn 1.6.0 [@pedregosa]; pandas 2.2.3; NumPy 2.1.2; SciPy 1.14.1; SHAP 0.53.0"],
        ["GRU", "1 layer, 64 units; Adam, learning rate 2×10⁻³, weight decay 10⁻⁴; batch 256; at most 60 epochs with early stopping (patience 8) on a 15% date-based validation split; gradient clip 1.0; MSE on the standardized target; 3 seeds (stopped at 23, 19 and 26 epochs)"],
        ["XGBoost", "600 trees, depth 5, learning rate 0.035, row sample 0.85, column sample 0.9, L2 5, minimum child weight 15, histogram method on GPU, seed 2"],
        ["Ridge", "alpha 1.0 after median imputation, missing indicators and standardization"],
        ["Run time", f"Development grid for 2025 (all candidates, GPU): 4 min 21 s. CPU inference: GRU ensemble {g['inference']['ms_per_1000']:.0f} ms and XGBoost {xt['inference']['ms_per_1000']:.0f} ms per 1,000 station-days"],
    ], [0.65, 2.85], ["l", "l"])
    r.heading2("Evaluation protocol")
    r.para("Each forecast test fixed its model, selection and interval radius before the test period was scored once ({tab:design}), and the dataset hashes were checked before scoring. The three tests use different periods and stations, so a model is compared only with the baseline on its own rows. The attention study used three designs on 2023 data: chronological (train January–August, validate September–October, test November–December), leave-one-city-out in eight folds, and season-block in four folds of interleaved months. The first attention runs stopped early on randomly held-out weeks; every attention result here comes from corrected runs with purged date-based validation.")
    r.table("design", "Design of the three locked forecast tests", ["Step", "2024 same-day", "2025 next-day", "2026 weather"],
            [["Fit", "Jan–Sep 2023", "Jan 2023–Sep 2024", "Jan 2024–Sep 2025"], ["Select", "Oct 2023", "Oct 2024", "Oct 2025"], ["Calibrate", "Nov–Dec 2023", "Nov–Dec 2024", "Nov–Dec 2025"],
             ["Scored", "2024", "2025", "Jan–Sep 2026"], ["Rows", "28,803", "30,263", "20,086"]], [0.65, 0.95, 0.95, 0.95])
    r.para("The metrics are root-mean-square error (RMSE), mean absolute error (MAE) and R². Gains are RMSE differences between paired models on identical rows, with 95% intervals from paired calendar-week block bootstraps (2,000 draws for the 2026 test). The intervals cover test-week sampling only and are unadjusted across multiple comparisons. A 90% prediction range was set as a fixed radius from calibration errors, and its coverage was checked on the test. A candidate could replace the default model only if the lower bound of its paired gain was positive against both the baseline and the GRU, with no loss of sharp-change precision or recall at a 50-point threshold. The rule was set before the 2026 test.")
    r.heading2("Faithfulness audit")
    r.para("Two audits were run on frozen models. In the fixed-model audits, 300 input rows were sampled for each corrected FA and S-SA checkpoint. Inputs were replaced group by group with either the training mean or a same-city, same-season donor value, and the change in model output defined a sensitivity ranking. The Spearman correlation between that ranking and each method's ranking, attention weights and SHAP values (128 samples), was averaged over seeds and rows. In an exploratory audit of temporal attention, 1,000 rows were used to compare replacing the days attention ranked highest against random and reverse rankings. High agreement shows that a ranking follows the model's behaviour; it does not show physical causation.")
    r.heading2("Pseudocode")
    r.algorithm(1, "Locked forecast evaluation", ["Input: aligned table D; periods T_fit, T_sel, T_cal, T_test", "Output: frozen model M_final, scores P, decision"], [
        (0, "D_c ← PREPROCESS(D)   // UTC day, AQI_eod, flags, lags"),
        (0, "**for** each candidate m and feature family f **do**"),
        (1, "fit m on D_c[T_fit]; s_m ← RMSE(m, D_c[T_sel])"),
        (0, "**end for**"),
        (0, "M_final ← argmin s_m;  r ← 90th percentile of |y − M_final(x)| on D_c[T_cal]"),
        (0, "FREEZE(M_final, r) and record file hashes"),
        (0, "ŷ ← M_final(x) on D_c[T_test];  b ← previous-day AQI on D_c[T_test]"),
        (0, "P ← {RMSE, MAE, R²} of ŷ and b on identical rows"),
        (0, "**for** k = 1 **to** 2000 **do**"),
        (1, "resample calendar weeks; g_k ← RMSE(b) − RMSE(ŷ)"),
        (0, "**end for**"),
        (0, "CI ← 2.5th and 97.5th percentiles of g"),
        (0, "promote M_final only if CI lower bound > 0 against b and the incumbent, with no sharp-change loss"),
        (0, "**return** M_final, P, CI")])
    r.algorithm(2, "Faithfulness audit of a frozen attention model", ["Input: model M, rows X, replacement rules R = {mean, seasonal donor}", "Output: mean Spearman agreement of attention and SHAP"], [
        (0, "S ← SAMPLE(X, 300)"),
        (0, "**for** each row x in S **do**"),
        (1, "a ← ATTENTION(M, x);  h ← SHAP(M, x)"),
        (1, "**for** each rule r in R **do**"),
        (2, "**for** each input group q **do** δ_q ← |M(REPLACE(x, q, r)) − M(x)|"),
        (2, "ρ_att ← SPEARMAN(a, δ);  ρ_shap ← SPEARMAN(|h|, δ)"),
        (1, "**end for**"),
        (0, "**end for**"),
        (0, "**return** mean ρ_att and ρ_shap over rows, averaged over seeds")])

    # ----------------------------------------------------------------- IV
    r.heading1("Results and Discussion")
    r.heading2("Performance metrics")
    r.para("Each AQI-history model beat the baseline on its own rows ({tab:results}) [@locked]. The gain was 3.39 AQI points (9.4%) for ridge with history in 2024, 2.87 (8.1%) for the GRU in 2025 and 4.40 (12.7%) for weather XGBoost in 2026, each with a 95% interval above zero. This indicates that the models forecast the next day more accurately than repeating today's value, but the typical error is still near 20 AQI points. Ridge-env, without AQI history, had RMSE 63.239, 75% above the baseline, so inputs other than AQI history are not enough on their own. The nominal 90% ranges covered 94.7% to 95.4% of observations, so they are conservative.")
    r.table("results", "Locked test results (RMSE, MAE in AQI points)", ["Model", "Test", "Rows", "MAE", "RMSE", "R²", "Gain over baseline [95% CI]", "Coverage"], main_rows(),
            [1.3, 1.0, 0.6, 0.55, 0.65, 0.55, 1.75, 0.8], ["l", "l", "r", "c", "c", "c", "c", "c"], wide=True)
    r.heading2("Comparison with baselines and ablations")
    r.para(f"A natural concern is that the GRU mostly repeats yesterday's AQI. To test this, the frozen GRU was re-run on all 2025 test rows with parts of its input removed ({{tab:ablation}}). Its predictions correlate {V['full']['corr_with_yesterday']:.3f} with yesterday's AQI and {pct(g['departure']['within10'])} lie within 10 points of it, so the concern is largely right: the GRU is anchored to yesterday. Yet yesterday's value alone is not enough. Given only yesterday's AQI, the GRU kept {pct(V['yesterday_only']['share_of_gain'])} of its gain; with yesterday repeated and calendar and location kept, {pct(V['yesterday_repeated']['share_of_gain'])}; and with the full history but no calendar or location, {pct(V['no_calendar_location']['share_of_gain'])}. A ridge model on all lags, calendar and location kept {pct(L['all_history']['share_of_gain'])} of the gain, so the GRU adds only about {V['full']['gain'] - L['all_history']['gain']:.1f} RMSE points beyond a linear model. The gain comes from older days, season and location, and from pulling extreme values toward the average (the one-variable fit is y = {g['departure']['one_variable_intercept']:.1f} + {g['departure']['one_variable_slope']:.3f} × yesterday). The inputs in the removal tests are unusual for the network, so they overstate its dependence on the removed data, and the linear comparisons are the cleaner evidence.")
    r.table("ablation", "Input ablations of the 2025 GRU and linear comparisons", ["Model", "RMSE", "Gain", "Share of GRU gain"], [
        ["Baseline (yesterday)", f"{g['baseline_rmse']:.3f}", "0.00", "0%"],
        ["Linear, yesterday only", f"{L['yesterday_only']['rmse']:.3f}", f"{L['yesterday_only']['gain']:.2f}", pct(L['yesterday_only']['share_of_gain'])],
        ["Linear, three recent inputs", f"{L['three_recent']['rmse']:.3f}", f"{L['three_recent']['gain']:.2f}", pct(L['three_recent']['share_of_gain'])],
        ["Linear, all history, calendar, location", f"{L['all_history']['rmse']:.3f}", f"{L['all_history']['gain']:.2f}", pct(L['all_history']['share_of_gain'])],
        ["GRU, only yesterday's AQI", f"{V['yesterday_only']['rmse']:.3f}", f"{V['yesterday_only']['gain']:.2f}", pct(V['yesterday_only']['share_of_gain'])],
        ["GRU, yesterday repeated", f"{V['yesterday_repeated']['rmse']:.3f}", f"{V['yesterday_repeated']['gain']:.2f}", pct(V['yesterday_repeated']['share_of_gain'])],
        ["GRU, no calendar or location", f"{V['no_calendar_location']['rmse']:.3f}", f"{V['no_calendar_location']['gain']:.2f}", pct(V['no_calendar_location']['share_of_gain'])],
        ["GRU, older days shuffled", f"{V['older_days_shuffled']['rmse']:.3f}", f"{V['older_days_shuffled']['gain']:.2f}", pct(V['older_days_shuffled']['share_of_gain'])],
        ["GRU, full model", f"{V['full']['rmse']:.3f}", f"{V['full']['gain']:.2f}", "100%"],
    ], [1.9, 0.5, 0.45, 0.65], ["l", "c", "c", "c"])
    r.para("Adding inputs step by step shows the same pattern for the tabular models fitted on 2023 data ({tab:ladder}): only the AQI-history step beat the baseline, and station location helped within known cities but not on an unseen one. For held-out cities, the spatial S-TA+SA model reached a city-averaged RMSE of 54.16 against 57.19 for its uniform control and improved five of eight cities, whereas sequence TA did not improve on its control (55.23 against 55.11). This is cross-city accuracy and is not evidence about faithfulness.")
    r.table("ladder", "Best test RMSE by input set, 2023 tabular models", ["Inputs", "Chrono.", "Held-out city", "Season"],
            [["Satellite only", "119.7", "72.4", "80.3"], ["Weather only", "97.8", "60.8", "60.7"], ["Satellite + weather", "93.5", "58.5", "57.2"],
             ["+ station location", "86.8", "57.1", "49.7"], ["+ day of year, past weather", "72.1", "57.2", "49.0"], ["+ AQI history", "42.4", "31.3", "34.9"],
             ["Baseline (yesterday)", "46.7", "32.8", "40.1"]], [1.6, 0.55, 0.8, 0.55])
    r.heading2("Feature importance of the weather model")
    r.para(f"The weather XGBoost uses 50 inputs ({{tab:features}}) and predicts the change from yesterday. On the 2026 test rows, the forecast weather carried {pct(xt['groups']['Weather forecast']['shap_share'])} of the model's mean absolute SHAP value, close to yesterday's AQI at {pct(xt['groups'][yd]['shap_share'])} and ahead of the older daily values at {pct(xt['groups']['Older daily AQI (days 2-28)']['shap_share'])} ({{fig:importance}}). Shuffling the weather inputs raised RMSE by {xt['groups']['Weather forecast']['rmse_rise']:.1f} points, and by {xt['weather_within_city_rise']:.1f} when shuffled within each city, which keeps city climate. Setting the forecast to missing raised RMSE from {xt['model_rmse']:.2f} to {xt['weather_missing_rmse']:.2f}, removing about {pct(1 - (xt['baseline_rmse'] - xt['weather_missing_rmse']) / (xt['baseline_rmse'] - xt['model_rmse']))} of the model's gain over the baseline. Forecast wind speed and humidity mattered most (RMSE rises of {xt['weather_variables']['wind_speed_10m']['rmse_rise']:.1f} and {xt['weather_variables']['relative_humidity_2m']['rmse_rise']:.1f}), which agrees with the physics of dispersion and moisture, but these scores describe the model's behaviour and do not prove cause. The development rows gave the same ranking. On those rows the history-only XGBoost scored close to the baseline, while adding weather lowered RMSE by 3 to 5 points ({{tab:weatherdev}}).")
    r.figure("importance", figs / "f2.png", "Share of the weather XGBoost's mean absolute SHAP value by input group.")
    r.table("weatherdev", "Weather forecast ablation on development rows (RMSE)", ["Model", "Oct 2025", "Nov–Dec 2025"], weather_dev_rows(), [2.0, 0.7, 0.8], ["l", "c", "c"])
    r.heading2("Attention and accuracy")
    r.para("Attention did not improve accuracy reliably ({fig:attention}). Across the 36 paired comparisons, attention helped in 11, hurt in 11 and was inconclusive in 14 [@stability]. The largest reversal was S-SA in the chronological test, which gained 9.49 on SWT-w7 and lost 3.58 on SW-w7. The most consistent result was S-TA+SA in held-out cities, which gained 3.03 on SWT-w7 and 3.00 on SW-w7 but showed no chronological gain on either set. Feature attention alone was a clear gain once and a clear loss in four of its six comparisons. With 36 unadjusted intervals, a few clear results would be expected by chance. These models omit AQI history, so their RMSE (about 53 to 95) is far above the forecast models'.")
    r.figure("attention", figs / "f3.png", "RMSE gain of each attention variant over its uniform control, with 95% intervals, on two input sets.")
    r.heading2("Faithfulness of attention")
    r.para("In all four fixed-model audits attention agreed less than SHAP with replacement sensitivity ({tab:faith}), which suggests that attention is a weaker guide to what the model uses. The result varied by city and replacement method. Under mean replacement, spatial attention nearly matched SHAP in Bengaluru (0.673 against 0.677) but was near zero or negative in Delhi (0.100), Jaipur (−0.017) and Lucknow (−0.104). City samples are uneven, from one row in Mumbai to 148 in Delhi, so small-city figures are weak.")
    r.table("faith", "Rank agreement with replacement sensitivity (Spearman)", ["Attention type", "Replacement", "Attention", "SHAP"], faithfulness_rows(), [1.0, 1.1, 0.6, 0.6], ["l", "l", "c", "c"])
    r.para("The exploratory temporal-attention audit shows the weights are not empty. Replacing the day attention ranked highest moved the output by 37.1 AQI points on average, against 17.3 for a random day and 7.5 for the lowest-ranked day. Rank agreement with leave-one-day-out sensitivity was 0.488 overall, and it did not fall on polluted days: 0.547 at AQI 300 or above (320 rows) and 0.461 below (680 rows). Chennai was weakest at 0.178.")
    r.heading2("Error analysis")
    r.para(f"All models read too high on clean days and too low on polluted days ({{fig:bias}}). The Severe band is only {share['Severe']:.1f}% of station-days, so the models see few examples of it and regress toward the mean. In the 401–500 band (94 station-days in 2026) the mean error was −44 for the weather model and −72 for the GRU, which means none of the models is suited to alerting on the worst days. Sharp changes are also missed ({{tab:sharp}}). The weather model detected 18.2% of moves of 50 points or more in the correct direction against 6.7% for the GRU, but its precision fell from 57.5% to 52.8%, so it failed the promotion rule's no-loss condition despite a lower RMSE (paired gain over the GRU 1.44, 95% interval 0.38 to 2.61). On days that changed by less than 25 points, the baseline had the lowest error, so the models earn their gain on the days that change.")
    r.figure("bias", figs / "f4.png", "Mean signed error by measured AQI band, January–September 2026. Below zero means too low.")
    sharp = []
    for grp in ["under 25", "25-under 50", "50 or more"]:
        d = change[change.group == grp].set_index("model")
        sharp.append([{"under 25": "Under 25", "25-under 50": "25 to under 50", "50 or more": "50 or more"}[grp], f"{d.loc['Weather XGBoost', 'n']:,}", f"{d.loc['Weather XGBoost', 'rmse']:.2f}", f"{d.loc['Frozen GRU', 'rmse']:.2f}", f"{d.loc['Persistence', 'rmse']:.2f}"])
    r.table("sharp", "RMSE by size of actual daily AQI change, 2026", ["Change (points)", "Days", "Weather", "GRU", "Baseline"], sharp, [0.95, 0.5, 0.6, 0.55, 0.9], ["l", "c", "c", "c", "c"])
    r.para("By city, the GRU improved on the baseline in all eight cities in 2025 ({fig:city}), with the largest gain in Delhi (3.74) and the smallest in Mumbai (1.29), although Delhi's error stays the largest at 41.5. Two alert classifiers fit on October 2024 and scored once on 2025 did not pass their promotion checks and were not deployed: for AQI of 300 or more the GRU threshold had precision 83.7% and recall 82.9% against 88.4% and 64.9% for the classifier.")
    r.figure("city", figs / "f5.png", "RMSE of the GRU and the baseline by city, 2025 test (30,263 station-days). Numbers show the gain.")
    r.heading2("Interpretation of results")
    r.para("Three conclusions follow. First, forecast skill beyond the baseline is real but modest: 8% to 13% in RMSE. Most of what the models predict is yesterday's value, the history-only GRU adds little over a linear model, and the clearest additional signal comes from weather forecasts. Second, attention is neither a reliable accuracy gain nor, in the sampled audits, as faithful a guide to model behaviour as SHAP, and its effects depend on the input set, the test design and the city. An attention map should therefore be checked for each model and setting before it is read as an explanation. Third, the forecasts regress toward the middle, which limits their use on the days that matter most for health. These findings describe model behaviour on past data and do not identify physical causes of pollution.")
    r.heading2("Comparison with past literature")
    r.para("The reviewed studies report scores on different targets, horizons and scales, so this work does not rank its numbers against theirs ({tab:compare}). The comparison below is therefore about what each study shows, and it connects the results to the gaps in Section II.")
    r.table("compare", "Reported results of reviewed studies and their relation to this work", ["Study", "Reported result", "Relation to this work"], [
        ["Chen et al. [@chen23]", "Hourly PM2.5, RMSE 9.82 one step and 13.49 averaged over 1–48 h", "Different pollutant, horizon and scale. Attention is a predictive precedent there; here its accuracy effect varied (11 helped, 11 hurt, 14 unclear)."],
        ["Liu et al. [@liu23]", "PM2.5 MAE 13.97, 16.13 and 4.07 on three datasets", "Component ablation there tests architecture; here a fixed-model audit tests whether attention reflects model behaviour."],
        ["Ma et al. [@ma]", "Lower error with four frozen layers than with one after target-site fine-tuning (scaled units)", "Adaptation uses target labels. The locked tests here use none."],
        ["Li et al. [@li]", "Site-based PM10 R² about 0.85 on 288 held-out stations", "Spatial assessment, not forecasting. Next-day R² here was 0.887 (2025) on a different target."],
        ["Liu et al. [@liu25]", "Beijing daily AQI RMSE 3.0012", "Different scale and preprocessing, and split independence is unverified there."],
        ["Zhang et al. [@zhang]", "Intrinsic explanations agreed closely with GradientSHAP", "Here attention agreed less than SHAP with replacement sensitivity in four of four audits. Models and cities differ, so the contrast is indicative."],
        ["Turbé et al. [@turbe]", "Shapley sampling ranked best among six attribution methods", "Consistent in direction: SHAP agreed more with measured sensitivity than attention did."],
        ["Šić et al. [@simic]", "Suitable perturbation depends on dataset and model", "Consistent: S-SA agreement moved from 0.242 to 0.483 when the replacement rule changed."],
    ], [1.05, 1.9, 4.2], ["l", "l", "l"], wide=True)

    # ------------------------------------------------------------------ V
    r.heading1("Conclusion")
    r.para("This study set out to measure how much AQI and weather models improve next-day station forecasts over a previous-day baseline, and whether attention explains what drives such models. It built an aligned dataset for 107 stations in eight cities and ran three locked forecast tests. The GRU lowered RMSE from 35.388 to 32.516 in 2025 and the weather XGBoost from 34.668 to 30.270 in 2026, while the history-only GRU added little beyond a linear model and the weather forecast supplied much of the extra skill. Across 36 comparisons attention did not reliably improve accuracy, and in every sampled audit it agreed less than SHAP with measured sensitivity. The contribution is a reproducible evaluation that separates forecast skill, input contribution and explanation faithfulness, with a dashboard of the saved results. The findings suggest that attention maps should be tested before they are used to explain environmental forecasts, and that a previous-day baseline should accompany any reported accuracy.")

    # ----------------------------------------------------------------- VI
    r.heading1("Study Limitations")
    r.para("*Dataset constraints.* The study used CPCB station exports and public satellite, reanalysis and archived forecast data for eight Indian cities. Station coverage is uneven, since Delhi supplies 39 of 107 stations, and 16 stations were not scored in 2026 because exports were unavailable or had no usable targets. Satellite values are limited by cloud cover, and the archived weather forecasts are city-level with an unverified delivery delay. The attention study uses 2023 data only.")
    r.para("*Technical and environmental restrictions.* All experiments ran on a single laptop with an 8 GB GPU. The attention study used three seeds and the fixed-model audits sampled 300 rows per checkpoint (1,000 for the temporal-attention audit), so the intervals cover test-week sampling and not training-seed variation. Replacing inputs with a training mean can create unrealistic inputs, and correlated inputs share credit in importance scores.")
    r.para("*Scope limitations.* The evaluation was offline on historical data and did not include live forecasts, because no same-day AQI source is configured. The out-of-time evaluation for 13 October to 31 December 2026 has no score as of 2 November 2026. The audits are model-behaviour interventions and not causal tests of atmospheric processes. LIME, remove-and-retrain, concept-activation testing and an audit of the held-out-city checkpoints were outside the scope of this study.")

    # ---------------------------------------------------------------- VII
    r.heading1("Future Scope")
    r.para("*Suggested improvements.* The frozen blend of 30% GRU and 70% weather XGBoost should be scored on the out-of-time evaluation after 1 January 2027 under the promotion rule stated in Section III. A history-only XGBoost can be retrained and scored on the 2026 test to isolate the weather effect cleanly. The audit can be extended to the held-out-city checkpoints and to LIME, remove-and-retrain and a concept-based stagnation test. Tail-aware training objectives could address the under-forecasting of the Severe band, and boundary-layer height could be added as a physically relevant input.")
    r.para("*Real-time implementation.* A same-day AQI source and a live weather forecast feed would allow forecasts to be recorded before their target day, so that live skill can be measured and not only backtested. The out-of-time status is tracked in [@status].")
    r.para("*Potential applications and extensions.* The method can be extended to PM2.5 and PM10 targets, to more cities, and to health advisories built on the dashboard. The audit procedure can also be reused to test attention explanations in other environmental forecasting tasks.")

    r.doc.add_paragraph("Acknowledgment", style="Heading 1")
    r.para("The authors thank Dr. Nisha P. Shetty for guidance throughout this project.")
    r.references()
    r.doc.save(OUT)
    print("saved", OUT, "| abstract words:", words, "| references:", len(r.cites))
    if pdf:
        import win32com.client as win32
        word = win32.Dispatch("Word.Application")
        word.Visible = False
        try:
            doc = word.Documents.Open(str(OUT), ReadOnly=True)
            doc.SaveAs(str(OUT.with_suffix(".pdf")), FileFormat=17)
            doc.Close(False)
        finally:
            word.Quit()
        print("saved", OUT.with_suffix(".pdf"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf", action="store_true", help="also export a PDF through Word")
    build(ap.parse_args().pdf)
