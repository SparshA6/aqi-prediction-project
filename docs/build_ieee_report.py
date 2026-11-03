"""Build the full project report in IEEE two-column format (Word, optionally PDF).

Tables and figures are computed from the saved result files, so the numbers
cannot drift from them. Run from anywhere:

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
from docx.enum.text import WD_ALIGN_PARAGRAPH
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
def result_tables():
    holdout = RES / "year_holdout/runs/2023-to-2024-v1"
    hm = pd.read_csv(holdout / "metrics.csv")
    hm = hm[(hm["mode"] == "same_day") & (hm.split == "test")].set_index("model")
    hman = json.loads((holdout / "manifest.json").read_text())["modes"]["same_day"]
    d25 = RES / "forecast_2025/runs/locked-2025-v1"
    m25 = pd.read_csv(d25 / "metrics.csv").set_index("model")
    man25 = json.loads((d25 / "manifest.json").read_text())
    d26 = RES / "forecast_2026/runs/locked-2026-v1"
    m26 = pd.read_csv(d26 / "metrics.csv").set_index("model")
    man26 = json.loads((d26 / "manifest.json").read_text())
    return hm, hman, m25, man25, m26, man26


def gain_text(g):
    return f"{g['gain']:.2f} [{g['lower']:.2f}, {g['upper']:.2f}]"


def accuracy_rows():
    hm, hman, m25, man25, m26, man26 = result_tables()
    ref24, ref25, ref26 = hm.loc["Persistence", "RMSE"], m25.loc["Persistence reference", "RMSE"], m26.loc["Persistence reference", "RMSE"]
    rows = [
        ["Ridge-history (2024)", f"{hm.loc['Ridge-history', 'RMSE']:.3f}", f"{ref24:.3f}", gain_text(hman["paired_rmse_gain"]), f"{hman['empirical_test_coverage'] * 100:.1f}%"],
        ["Ridge-env (2024)", f"{hm.loc['Ridge-env', 'RMSE']:.3f}", f"{ref24:.3f}", "none", "n/a"],
        ["GRU-28days (2025)", f"{m25.iloc[0].RMSE:.3f}", f"{ref25:.3f}", gain_text(man25["paired_rmse_gain"]), f"{man25['empirical_interval_coverage'] * 100:.1f}%"],
        ["Weather XGBoost (2026)", f"{m26.loc['XGBoost-change-forecast-weather-d5', 'RMSE']:.3f}", f"{ref26:.3f}", gain_text(man26["paired_rmse_gain"]), f"{man26['interval_coverage'] * 100:.1f}%"],
        ["Frozen GRU (2026)", f"{m26.loc['Frozen GRU-28days', 'RMSE']:.3f}", f"{ref26:.3f}", f"{ref26 - m26.loc['Frozen GRU-28days', 'RMSE']:.2f}", "n/a"],
    ]
    assert rows[0][1] == "32.806" and rows[2][1] == "32.516" and rows[3][1] == "30.270" and rows[1][1] == "63.239", rows
    return rows


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
    fig, ax = plt.subplots(figsize=(3.4, 2.85))
    ax.set_xlim(0, 100); ax.set_ylim(-6, 108); ax.axis("off")
    def box(x, y, w, h, text, accent=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5", fc="#e8eef8" if accent else "white", ec=BLUE if accent else "#555555", lw=0.8 if accent else 0.6))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.2, linespacing=1.15)
    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#555555", shrinkA=0, shrinkB=0, mutation_scale=6))
    box(1, 92, 30, 14, "CPCB hourly AQI\n107 stations, 8 cities")
    box(35, 92, 30, 14, "Earth Engine\nsatellite + ERA5-Land")
    box(69, 92, 30, 14, "Open-Meteo GFS\nforecasts (2026 only)")
    box(8, 66, 84, 14, "Aligned station-day table\ntarget: end-of-day AQI, stuck-sensor days flagged")
    box(1, 38, 46, 20, "Attention study\n(no AQI history)\nsequence and spatial models\nvs uniform controls", True)
    box(53, 38, 46, 20, "Forecast study\n(uses AQI history)\nRidge 2024, GRU 2025,\nXGBoost 2026")
    box(1, 14, 46, 16, "Faithfulness audits\nattention vs SHAP")
    box(53, 14, 46, 16, "Locked tests, then\nout-of-time test\n(opens 1 Jan 2027)")
    box(8, -4, 84, 11, "Reports and dashboard")
    for x in (16, 50, 84):
        arrow(x, 91, 50 if x == 50 else 16 + (x - 16) * 0.56 + 22, 81.5)
    arrow(30, 65, 24, 59.5); arrow(70, 65, 76, 59.5)
    arrow(24, 37, 24, 31.5); arrow(76, 37, 76, 31.5)
    arrow(24, 13, 30, 8.5); arrow(76, 13, 70, 8.5)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.995, bottom=0.005)
    save(fig, path, tight=False)


def fig_city(path):
    g = pd.read_csv(RES / "forecast_2025/runs/locked-2025-v1/group_metrics.csv").query("group_type == 'city'")
    gru = g[g.model == "GRU-28days"].set_index("group").RMSE
    ref = g[g.model == "Persistence reference"].set_index("group").RMSE
    assert (gru < ref).all()
    order = gru.sort_values().index
    fig, ax = plt.subplots(figsize=(3.4, 2.35))
    for i, city in enumerate(order):
        ax.plot([gru[city], ref[city]], [i, i], color="#999999", lw=1.0, zorder=1)
        ax.scatter(ref[city], i, s=16, facecolor="white", edgecolor=GREY, lw=0.9, zorder=2, label="Previous-day reading" if i == 0 else None)
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
        ax.bar([i + (k - 1) * 0.27 for i in range(6)], d.bias, width=0.25, color=fc, edgecolor=ec or fc, lw=0.5,
               label="Previous-day reading" if model == "Persistence" else model)
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
        self.figs = ["pipeline", "city", "bias", "attention"]
        self.tabs = ["data", "design", "accuracy", "sharp", "ladder", "faith"]
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
        text = re.sub(r"\[((?:@\w+(?:, ?)?)+)\]", cite, text)
        text = self.smart(text)
        text = re.sub(r"\{fig:(\w+)\}", lambda m: f"Fig. {self.figs.index(m.group(1)) + 1}", text)
        return re.sub(r"\{tab:(\w+)\}", lambda m: f"Table {roman(self.tabs.index(m.group(1)) + 1)}", text)

    def runs(self, paragraph, text, size=None, bold=None):
        for part in re.split(r"(\*\*.+?\*\*|\*.+?\*)", self.tokens(text)):
            if not part: continue
            run = paragraph.add_run(part[2:-2] if part.startswith("**") else part[1:-1] if part.startswith("*") and len(part) > 2 else part)
            run.font.name = FONT
            if part.startswith("**") or bold: run.bold = True
            if part.startswith("*") and not part.startswith("**") and len(part) > 2: run.italic = True
            if size: run.font.size = Pt(size)
        return paragraph

    def para(self, text, style="IEEE Body"):
        return self.runs(self.doc.add_paragraph(style=style), text)

    def heading1(self, text):
        self.h1 += 1; self.h2 = 0
        self.doc.add_paragraph(f"{roman(self.h1)}. {text}", style="Heading 1")

    def heading2(self, text):
        self.h2 += 1
        self.doc.add_paragraph(f"{chr(64 + self.h2)}. {text}", style="Heading 2")

    def figure(self, key, path, caption):
        assert self.figs[len(self.figs_done())] == key, key
        p = self.doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next, p.paragraph_format.space_before = True, Pt(4)
        p.add_run().add_picture(str(path), width=Inches(3.4))
        c = self.doc.add_paragraph(style="IEEE Fig Caption")
        self.runs(c, f"Fig. {self.figs.index(key) + 1}. {caption}")
        c.alignment = WD_ALIGN_PARAGRAPH.CENTER if len(caption) < 70 else WD_ALIGN_PARAGRAPH.JUSTIFY
        self._figs.append(key)

    _figs = None
    def figs_done(self):
        if self._figs is None: self._figs = []
        return self._figs

    def table(self, key, title, header, rows, widths, align=None):
        assert self.tabs[len(self.tabs_done())] == key, key
        cap = self.doc.add_paragraph(style="IEEE Table Title")
        cap.paragraph_format.keep_with_next = True
        cap.add_run(f"TABLE {roman(self.tabs.index(key) + 1)}").font.name = FONT
        cap.add_run().add_break()
        r = cap.add_run(title); r.font.name = FONT
        t = self.doc.add_table(rows=1 + len(rows), cols=len(header))
        t.alignment, t.autofit = WD_TABLE_ALIGNMENT.CENTER, False
        align = align or ["l"] + ["c"] * (len(header) - 1)
        tbl_borders = OxmlElement("w:tblBorders")
        for side in ("top", "bottom"):
            e = OxmlElement(f"w:{side}"); e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "6"); e.set(qn("w:color"), "000000"); tbl_borders.append(e)
        t._tbl.tblPr.append(tbl_borders)
        grid = t._tbl.tblGrid
        for gc, w in zip(grid.findall(qn("w:gridCol")), widths): gc.set(qn("w:w"), str(int(w * 1440)))
        for ri, row in enumerate([header] + rows):
            tr = t.rows[ri]._tr
            props = tr.get_or_add_trPr(); props.append(OxmlElement("w:cantSplit"))
            for ci, text in enumerate(row):
                cell = t.cell(ri, ci); cell.width = Inches(widths[ci])
                para = cell.paragraphs[0]; para.style = self.doc.styles["IEEE Table Text"]
                para.alignment = {"l": WD_ALIGN_PARAGRAPH.LEFT, "c": WD_ALIGN_PARAGRAPH.CENTER, "r": WD_ALIGN_PARAGRAPH.RIGHT}[align[ci]]
                para.paragraph_format.keep_with_next = ri < len(rows)
                self.runs(para, text, bold=ri == 0)
                self.borders(cell, top=False, bottom=ri == 0)
                margins = OxmlElement("w:tcMar")
                for side in ("left", "right"):
                    m = OxmlElement(f"w:{side}"); m.set(qn("w:w"), "40"); m.set(qn("w:type"), "dxa"); margins.append(m)
                cell._tc.get_or_add_tcPr().append(margins)
        self.doc.add_paragraph().paragraph_format.space_after = Pt(2)
        self._tabs.append(key)

    _tabs = None
    def tabs_done(self):
        if self._tabs is None: self._tabs = []
        return self._tabs

    @staticmethod
    def borders(cell, top, bottom):
        tcPr = cell._tc.get_or_add_tcPr()
        b = OxmlElement("w:tcBorders")
        for side, on in (("top", top), ("bottom", bottom)):
            if on:
                e = OxmlElement(f"w:{side}"); e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "6"); e.set(qn("w:color"), "000000"); b.append(e)
        tcPr.append(b)

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
    return ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"][n]


REFS = {
    "jain": 'S. Jain and B. C. Wallace, "Attention is not explanation," in *Proc. NAACL-HLT*, 2019, doi: 10.18653/v1/N19-1357.',
    "wiegreffe": 'S. Wiegreffe and Y. Pinter, "Attention is not not explanation," in *Proc. EMNLP-IJCNLP*, 2019, doi: 10.18653/v1/D19-1002.',
    "bahdanau": 'D. Bahdanau, K. Cho, and Y. Bengio, "Neural machine translation by jointly learning to align and translate," in *Proc. ICLR*, 2015, arXiv:1409.0473.',
    "cho": 'K. Cho *et al.*, "Learning phrase representations using RNN encoder-decoder for statistical machine translation," in *Proc. EMNLP*, 2014, doi: 10.3115/v1/D14-1179.',
    "chen": 'T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining*, 2016, doi: 10.1145/2939672.2939785.',
    "lundberg": 'S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Proc. NeurIPS*, 2017, arXiv:1705.07874.',
    "krishna": 'S. Krishna *et al.*, "The disagreement problem in explainable machine learning: A practitioner\'s perspective," arXiv:2202.01602, 2022.',
    "prior": 'AQI project, "Prior work review and novelty analysis," docs/Prior_Work_Review_and_Novelty_Analysis.pdf, Sep. 2026.',
    "cpcb": 'Central Pollution Control Board, "National Air Quality Index and AQI data repository." [Online]. Available: https://airquality.cpcb.gov.in/AQI_India/',
    "veefkind": 'J. P. Veefkind *et al.*, "TROPOMI on the ESA Sentinel-5 Precursor: A GMES mission for global observations of the atmospheric composition for climate, air quality and ozone layer applications," *Remote Sens. Environ.*, vol. 120, pp. 70–83, 2012.',
    "modis": 'NASA LP DAAC, "MCD19A2.061 Terra and Aqua MAIAC land aerosol optical depth daily 1 km," Google Earth Engine Data Catalog. [Online]. Available: https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MCD19A2_GRANULES',
    "munoz": 'J. Muñoz-Sabater *et al.*, "ERA5-Land: A state-of-the-art global reanalysis dataset for land applications," *Earth Syst. Sci. Data*, vol. 13, no. 9, pp. 4349–4383, 2021.',
    "openmeteo": 'Open-Meteo, "Previous Runs API." [Online]. Available: https://open-meteo.com/en/docs/previous-runs-api',
    "locked": 'AQI project, "Locked evaluation records," results/year_holdout/runs/2023-to-2024-v1, results/forecast_2025/runs/locked-2025-v1 and results/forecast_2026/runs/locked-2026-v1 (metrics, manifests and verification records), 2026.',
    "stability": 'AQI project, "Attention versus uniform control across input sets and protocols," results/attention_stability/gains.csv, generated by src/models/report_attention_stability.py, Nov. 2026.',
    "status": 'AQI project, "Prospective tracking status," results/prospective/status.json, updated 2 Nov. 2026.',
}


def build(pdf=False):
    hm_rows = accuracy_rows()
    band, change = parse_results_report()
    r = Report()
    figs = Path(tempfile.mkdtemp())
    fig_pipeline(figs / "f1.png"); fig_city(figs / "f2.png"); fig_bias(figs / "f3.png", band); fig_attention(figs / "f4.png")

    r.title_block("Air Quality Index Prediction and Attention Faithfulness Across Eight Indian Cities", [
        ["Saanvi Garg", "Section CSE-C", "Reg. No. 240905702, Roll No. 68", "School of Computer Engineering", "Manipal Institute of Technology", "Manipal Academy of Higher Education", "Manipal, Karnataka, India"],
        ["Sparsh Agrawal", "Section CSE-C", "Reg. No. 240905708, Roll No. 70", "School of Computer Engineering", "Manipal Institute of Technology", "Manipal Academy of Higher Education", "Manipal, Karnataka, India"],
        ["Dhruv Chamria", "Section CSE-C", "Reg. No. 240905706, Roll No. 69", "School of Computer Engineering", "Manipal Institute of Technology", "Manipal Academy of Higher Education", "Manipal, Karnataka, India"],
        ["Neelaksha Sisodiya", "Section CSE-C", "Reg. No. 240905642, Roll No. 65", "School of Computer Engineering", "Manipal Institute of Technology", "Manipal Academy of Higher Education", "Manipal, Karnataka, India"]])

    p = r.doc.add_paragraph(style="IEEE Abstract")
    p.add_run("Abstract—").italic = True
    r.runs(p, "Attention weights are often presented as explanations of air quality models without testing whether they reflect what drives the model's output. This report studies 107 CPCB stations in eight Indian cities from 2023 to 2026, combining station AQI with satellite, reanalysis and archived forecast-weather data. Two tasks are separated: same-day estimation and next-day forecasting. Against a previous-day reference, a history-aware ridge model reached RMSE 32.806 in 2024 same-day estimation, a 28-day GRU ensemble reached 32.516 against 35.388 in 2025 next-day forecasting, and a weather-assisted XGBoost model reached 30.270 against 34.668 on January–September 2026. The weather model lost sharp-change precision relative to the GRU and was not promoted. Across 36 paired comparisons, attention improved accuracy in 11, worsened it in 11 and was inconclusive in 14, and the sign of several effects reversed between input sets. In sampled fixed-model audits, attention rankings agreed less than SHAP with input-replacement sensitivity in all four comparisons. Every forecast model under-predicted the highest AQI band. A final out-of-time evaluation of the frozen models opens on 1 January 2027 and has no score yet. The results support treating attention as a hypothesis to be tested, not as an explanation.", size=9, bold=True)
    p = r.doc.add_paragraph(style="IEEE Abstract")
    p.paragraph_format.space_after = Pt(4)
    p.add_run("Index Terms—").italic = True
    r.runs(p, "air quality index, attention faithfulness, explainable AI, time-series forecasting, GRU, XGBoost, cross-city transfer, SHAP", size=9, bold=True)

    r.heading1("Introduction")
    r.para("Air quality varies steeply across stations, seasons and weather regimes, and tomorrow's Air Quality Index (AQI) at a station is already strongly related to today's. A forecast therefore has to be judged against the simplest alternative, repeating yesterday's reading. This report compares every model it evaluates with that previous-day reference on identical rows.")
    r.para("Attention layers are common in AQI models because they are accurate and are assumed to be interpretable: the weights are read as a ranking of the inputs the model relied on. Work on text classification has shown that attention weights can disagree with other importance measures and are not in general faithful explanations [@jain], although the evidence is contested [@wiegreffe]. Whether this transfers to a continuous AQI target driven by correlated physical inputs, across cities with different pollution regimes, is less well tested. In this project's prior-work review, air-quality studies typically showed attention beside post-hoc attributions for a single city [@prior].")
    r.para("This report therefore treats attention as a hypothesis to be tested. It covers 107 Central Pollution Control Board (CPCB) stations in eight Indian cities and makes five contributions: (1) an aligned multi-source dataset with a documented end-of-day target; (2) three locked year-level tests of same-day estimation and next-day forecasting, including a weather-assisted comparison; (3) a paired comparison of 36 attention variants against uniform controls across two input sets and three test designs; (4) sampled fixed-model audits comparing attention and SHAP rankings with input-replacement sensitivity; and (5) a public dashboard of the saved results. The predictor is a means to the audit, and forecast accuracy and explanation faithfulness are reported as separate questions. {fig:pipeline} shows how the pieces fit together.")
    r.figure("pipeline", figs / "f1.png", "Data and evaluation pipeline. Two studies share one aligned table.")

    r.heading1("Related Work")
    r.para("Attention was introduced as a soft alignment mechanism for sequence models [@bahdanau]. Jain and Wallace showed that attention weights can be altered without changing predictions and often correlate weakly with gradient-based importance [@jain]. Wiegreffe and Pinter argued that the conclusion depends on how an explanation is defined and tested [@wiegreffe]. This report follows that view in one respect: faithfulness is measured against an intervention, replacing inputs and observing the output, and is not assumed.")
    r.para("Post-hoc attribution methods such as SHAP assign each input a contribution to a prediction [@lundberg], but different explanation methods can disagree on the same model [@krishna]. SHAP is therefore used here as a comparison method and not as ground truth: attention and SHAP are both scored against the same replacement-based sensitivity. The forecasting models are standard, namely ridge regression, a gated recurrent unit (GRU) network [@cho] and gradient-boosted trees [@chen]. The emphasis of the study is evaluation design: locked year-level tests, persistence references, paired intervals and an explicit promotion rule.")

    r.heading1("Data and Target")
    r.heading2("Sources")
    r.para("Hourly station AQI came from CPCB exports [@cpcb]. Satellite inputs were Sentinel-5P NO₂ and HCHO [@veefkind] and MODIS aerosol optical depth [@modis], and weather inputs were ERA5-Land daily reanalysis [@munoz], all extracted through Google Earth Engine on a 0.03° grid. The 2026 weather model also used GFS forecasts issued 24 hours before each valid hour, taken from the Open-Meteo Previous Runs API [@openmeteo]. The 107 stations are in Delhi (39), Hyderabad (14), Mumbai (14), Bengaluru (13), Chennai (8), Kolkata (7), Jaipur (6) and Lucknow (6).")
    r.heading2("Target")
    r.para("CPCB's hourly AQI is a trailing 24-hour average, so a daily mean of it is blurred across two days: its correlation with the previous day's satellite NO₂ (0.217) nearly matched that with the same day's (0.230). The target is therefore *AQI_eod*, the hourly AQI at 05:00 IST on the following day, whose window covers almost exactly the UTC day described by the satellite and weather inputs. On the 2023 table this raised the same-day NO₂ correlation to 0.272 and lowered the previous-day value to 0.151. All dates are UTC.")
    r.heading2("Quality and coverage")
    r.para("{tab:data} lists the aligned data. Days on which AQI stayed constant below the 500 cap were flagged as stuck-sensor readings (906 rows), and missing targets were never imputed. An independent audit reproduced all 31,813 aligned rows of 2023 from the raw files with no mismatches.")
    r.table("data", "Aligned data by year", ["Year", "Station-days", "With AQI", "Stations"],
            [["2023", "31,813", "30,081", "107"], ["2024", "33,860", "31,722", "106"], ["2025", "34,189", "32,842", "107"], ["All", "99,862", "94,645", "107"]], [0.6, 1.0, 1.0, 0.9])
    r.para("Coverage is uneven. Delhi supplies 39 of the 107 stations. Satellite values are often missing under cloud: in 2023, 45% of rows had all three bands and 23% had none, with July to September weakest. Coastal stations match a farther grid cell, up to 9.0 km. The 2026 test scored 20,086 station-days at 91 stations. Thirteen stations had no January–September 2026 exports and three more had files but no usable targets; 12 of the 13 do have October 2026 exports, so some gaps may be download failures. Exclusions were not based on scores.")

    r.heading1("Methods")
    r.heading2("Forecasting models")
    r.para("Same-day estimates use satellite and weather data from the target day and can only be produced afterwards, so they are retrospective estimates and not forecasts. Next-day forecasts exclude every target-day observation. Three AQI-history models were evaluated. A ridge regression used previous-day AQI and lagged readings for same-day estimation (Ridge-history), alongside an environment-only variant without AQI history (Ridge-env). A three-seed ensemble of GRUs (GRU-28days) forecast the next day from 28 consecutive daily AQI values with calendar and location inputs. A gradient-boosted model (XGBoost, tree depth 5) predicted the change from the previous day's reading using 28 AQI lags and the weather forecasts issued 24 hours earlier. Each model's 90% range was a fixed radius set on calibration months, and hyperparameters were fixed before testing.")
    r.heading2("Attention models")
    r.para("The attention study used environment-only inputs (satellite and weather, no AQI history) over a 7-day window, so that attention had to work with physical drivers. Sequence models used a 64-unit GRU encoder with temporal attention over the days (TA), feature attention over each day's inputs (FA), or both (TA+FA). Spatial models read a 3×3 patch of grid cells around each station, about 10 km apart, with a shared GRU and attention over days (S-TA), over cells (S-SA), or both (S-TA+SA). Each variant was compared with a uniform control with identical inputs and fixed equal weights. Two input sets were used: SW-w7 (satellite and weather) and SWT-w7 (which adds day of year). Each result averages three seeds with untuned settings.")
    r.heading2("Evaluation protocols")
    r.para("Each forecast test fixed its model, selection and interval radius before the test period was scored once ({tab:design}). The three tests use different periods and stations, so a model is compared only with the previous-day reference on its own rows. The attention study used three designs on 2023 data: chronological (train January–August, validate September–October, test November–December), leave-one-city-out in eight folds, and season-block in four folds of interleaved months. The first attention runs stopped early on randomly held-out weeks; every attention result here comes from corrected runs with purged date-based validation.")
    r.table("design", "Design of the three locked forecast tests", ["Step", "2024 same-day", "2025 next-day", "2026 weather"],
            [["Fit", "Jan–Sep 2023", "Jan 2023–Sep 2024", "Jan 2024–Sep 2025"], ["Select", "Oct 2023", "Oct 2024", "Oct 2025"], ["Calibrate", "Nov–Dec 2023", "Nov–Dec 2024", "Nov–Dec 2025"],
             ["Scored", "2024", "2025", "Jan–Sep 2026"], ["Rows", "28,803", "30,263", "20,086"]], [0.65, 0.95, 0.95, 0.95])
    r.heading2("Statistics and promotion rule")
    r.para("Gains are RMSE differences between paired models on identical rows, with 95% intervals from paired calendar-week block bootstraps (2,000 draws for the 2026 test). The intervals cover test-week sampling only, not training seeds, and are unadjusted across multiple comparisons. A candidate could replace the default model only if the lower bound of its paired gain was positive against both the previous-day reference and the GRU, with no loss of sharp-change precision or recall at a 50-point threshold. The rule was set before the 2026 test.")
    r.heading2("Faithfulness audits")
    r.para("Two audits were run on frozen models. In the fixed-model audits, 300 input rows were sampled for each corrected FA and S-SA checkpoint. Inputs were replaced group by group with either the training mean or a same-city, same-season donor value, and the change in model output defined a sensitivity ranking. The Spearman correlation between that ranking and each method's ranking, attention weights and SHAP values (128 samples), was averaged over seeds and rows. In an exploratory audit of temporal attention, 1,000 rows were used to compare replacing the days attention ranked highest against random and reverse rankings. High agreement shows that a ranking follows the model's behaviour; it does not show physical causation.")

    r.heading1("Results")
    r.heading2("Forecast and estimation accuracy")
    r.para("Each AQI-history model beat the previous-day reference on its own rows ({tab:accuracy}) [@locked]. The gain was 3.39 AQI points (9.4%) for Ridge-history in 2024, 2.87 (8.1%) for the GRU in 2025 and 4.40 (12.7%) for weather XGBoost in 2026, each with a 95% interval above zero. Ridge-env, without AQI history, had RMSE 63.239, 75% above the reference. The nominal 90% ranges covered 94.7% to 95.4% of observations (coverage in {tab:accuracy}), so they are conservative. Even the best model's mean absolute error stays near 20 AQI points.")
    r.table("accuracy", "RMSE against the previous-day reference", ["Model (test year)", "RMSE", "Previous day", "Gain [95% CI]", "Coverage"], hm_rows, [1.22, 0.42, 0.48, 0.78, 0.6], ["l", "c", "c", "c", "c"])
    r.para("The GRU improved on the reference in all eight cities in 2025 ({fig:city}), with the largest gain in Delhi (3.74) and the smallest in Mumbai (1.29). Delhi's error is nevertheless the largest, at 41.5.")
    r.figure("city", figs / "f2.png", "RMSE of the GRU and the previous-day reading by city, 2025 test (30,263 station-days). Numbers show the gain.")

    r.heading2("Weather forecasts and sharp changes")
    sharp = []
    for g in ["under 25", "25-under 50", "50 or more"]:
        d = change[change.group == g].set_index("model")
        sharp.append([{"under 25": "Under 25", "25-under 50": "25 to under 50", "50 or more": "50 or more"}[g], f"{d.loc['Weather XGBoost', 'n']:,}", f"{d.loc['Weather XGBoost', 'rmse']:.2f}", f"{d.loc['Frozen GRU', 'rmse']:.2f}", f"{d.loc['Persistence', 'rmse']:.2f}"])
    r.para("The weather model had lower RMSE than the frozen GRU in all eight cities, with a paired gain of 1.44 (95% interval 0.38 to 2.61), but it was not promoted. At a 50-point threshold it detected 18.2% of sharp changes in the correct direction against 6.7% for the GRU, while precision fell from 57.5% to 52.8%, so the no-loss condition of the promotion rule failed. {tab:sharp} shows that the previous-day reading is the best of the three when the day changes by less than 25 points, so the models earn their gain on the days that change.")
    r.table("sharp", "RMSE by size of actual daily AQI change, 2026", ["Change (points)", "Days", "Weather", "GRU", "Previous day"], sharp, [0.95, 0.5, 0.6, 0.55, 0.9], ["l", "c", "c", "c", "c"])
    r.para("All models read too high on clean days and too low on polluted days ({fig:bias}). In the 401–500 band (94 station-days) the mean error was −44 for the weather model and −72 for the GRU, so none of the models is suited to alerting on the worst days.")
    r.figure("bias", figs / "f3.png", "Mean signed error by measured AQI band, January–September 2026. Below zero means too low.")

    r.heading2("Data sources and cross-city transfer")
    r.para("The 2023 baselines show what each input adds ({tab:ladder}). Only the AQI-history step beat the previous-day reference, and station location helped within known cities but not on an unseen one. For held-out cities, the spatial S-TA+SA model reached a city-averaged RMSE of 54.16 against 57.19 for its uniform control and improved five of eight cities, whereas sequence TA did not improve on its control (55.23 against 55.11). This is cross-city accuracy and is not evidence about faithfulness.")
    r.table("ladder", "Best test RMSE by input set, 2023 baselines", ["Inputs", "Chrono.", "Held-out city", "Season"],
            [["Satellite only", "119.7", "72.4", "80.3"], ["Weather only", "97.8", "60.8", "60.7"], ["Satellite + weather", "93.5", "58.5", "57.2"],
             ["+ station location", "86.8", "57.1", "49.7"], ["+ day of year, past weather", "72.1", "57.2", "49.0"], ["+ AQI history", "42.4", "31.3", "34.9"],
             ["Previous-day reading", "46.7", "32.8", "40.1"]], [1.6, 0.55, 0.8, 0.55])

    r.heading2("Does attention improve accuracy?")
    r.para("Not reliably ({fig:attention}). Across the 36 paired comparisons, attention helped in 11, hurt in 11 and was inconclusive in 14 [@stability]. The largest reversal was S-SA in the chronological test, which gained 9.49 on SWT-w7 and lost 3.58 on SW-w7. The most consistent result was S-TA+SA in held-out cities, which gained 3.03 on SWT-w7 and 3.00 on SW-w7 but showed no chronological gain on either set. Feature attention alone was a clear gain once and a clear loss in four of its six comparisons. With 36 unadjusted intervals, a few clear results would be expected by chance. All these models omit AQI history, so their RMSE (about 53 to 95) is far above the forecast models'.")
    r.figure("attention", figs / "f4.png", "RMSE gain of each attention variant over its uniform control, with 95% intervals, on two input sets.")

    r.heading2("Is attention a faithful explanation?")
    r.para("In all four fixed-model audits attention agreed less than SHAP with replacement sensitivity ({tab:faith}). The result varied by city and replacement method. Under mean replacement, spatial attention nearly matched SHAP in Bengaluru (0.673 against 0.677) but was near zero or negative in Delhi (0.100), Jaipur (−0.017) and Lucknow (−0.104). City samples are uneven, from one row in Mumbai to 148 in Delhi, so small-city figures are weak.")
    r.table("faith", "Rank agreement with replacement sensitivity (Spearman)", ["Attention type", "Replacement", "Attention", "SHAP"], faithfulness_rows(), [1.0, 1.1, 0.6, 0.6], ["l", "l", "c", "c"])
    r.para("The exploratory temporal-attention audit shows the weights are not empty. Replacing the day attention ranked highest moved the output by 37.1 AQI points on average, against 17.3 for a random day and 7.5 for the lowest-ranked day. Rank agreement with leave-one-day-out sensitivity was 0.488 overall, and it did not fall on polluted days: 0.547 at AQI 300 or above (320 rows) and 0.461 below (680 rows). Chennai was weakest at 0.178.")
    r.heading2("Alert classifiers and the frozen blend")
    r.para("Two alert classifiers were fit on October 2024 and scored once on 2025. Neither passed its promotion check and neither was deployed. For AQI of 300 or more the existing GRU threshold had precision 83.7% and recall 82.9%, against 88.4% and 64.9% for the classifier; for changes of 50 or more the existing rule had recall 4.3% against 0.2%. A frozen blend of 30% GRU and 70% weather XGBoost was chosen on October 2025 and has no test score.")

    r.heading1("Discussion")
    r.para("Three conclusions follow. First, AQI history carries most of the usable signal: models without it never beat the previous-day reading, and models with it improve on it by 8% to 13%. Second, attention is neither a reliable accuracy gain nor, in the sampled audits, as faithful a guide to model behaviour as SHAP. Its effects depend on the input set, the test design and the city, so an attention map should be checked for each model and setting before it is read as an explanation. Third, the forecast models regress toward the middle, which limits their use for the days that matter most for health. These findings describe model behaviour on past data and do not identify physical causes of pollution.")

    r.heading1("Dashboard and Reproducibility")
    r.para("A Flask dashboard and a static public demo present the saved results; they show past test periods and not live AQI. *Explore results* compares a chosen model, city, station and day with the measurement and the previous-day reference. *Try a forecast* runs the frozen GRU for a 2025 city date or from 28 supplied readings; earlier dates are not offered because they are training data. *How accurate?* presents the scores reported here, and *Guide* explains the terms. Scores on the page are recomputed from the saved rows and match the headline RMSEs, and the forecast form reproduces saved 2025 predictions to within 0.0001 AQI points on 19 Delhi stations. Saved metrics are checked by independent verification scripts, frozen model files by SHA-256 hashes, and 54 automated tests pass.")

    r.heading1("Out-of-Time Evaluation Status")
    r.para("A final evaluation of the frozen blend is reserved for targets from 13 October to 31 December 2026 and cannot open before 1 January 2027 [@status]. As of 2 November 2026, 104 of 107 stations had October exports, 1,559 usable station-days fell in the test window, the latest measured day was 30 October and no forecast had been recorded before its target day. A before-target record needs the previous day's AQI, which monthly exports cannot supply in time, so the evaluation will test frozen models on later data and not forecasts issued live. The weather snapshot of 2 November failed for all eight cities and is to be retried. The evaluation has no score yet, and the promotion rule above applies unchanged.")

    r.heading1("Limitations")
    r.para("Every score is a backtest on past data, and the archived weather forecasts have unverified publication delay and city-level resolution. The three forecast tests cannot be ranked against each other. Attention intervals cover test-week sampling only and are unadjusted across 36 comparisons, and the attention study covers 2023 with three seeds and untuned settings. The audits use sampled rows and correlated inputs, mean replacement can create unrealistic inputs, and the held-out-city checkpoints have not been audited. LIME, remove-and-retrain, concept-activation testing and a comparison with known atmospheric relationships were planned but not carried out, so no claim here rests on them. Delhi dominates the rows, satellite values are coarse once-a-day snapshots, and city values are plain averages of reporting stations and not official city AQI.")

    r.heading1("Conclusion and Future Work")
    r.para("AQI-history models beat the previous-day reference by 8% to 13% in three locked tests, and weather forecasts added accuracy but lost sharp-change precision. Attention did not reliably improve accuracy and ranked inputs less faithfully than SHAP in every sampled audit, so it should be tested before being used as an explanation. Future work is to score the out-of-time evaluation after 1 January 2027, audit the held-out-city checkpoints, run the planned explanation methods, obtain a same-day AQI source for live forecasts, and retry the 16 stations that were not scored in 2026.")

    r.doc.add_paragraph("Acknowledgment", style="Heading 1")
    r.para("The authors thank Dr. Nisha P. Shetty for guidance throughout this project.")
    r.references()
    r.doc.save(OUT)
    print("saved", OUT)
    if pdf:
        import win32com.client as win32
        word = win32.gencache.EnsureDispatch("Word.Application")
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
