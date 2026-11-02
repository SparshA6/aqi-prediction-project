"""Create the Results, Discussion and Inference project submission."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "AQI_Results_and_Discussion_Submission.pdf"
LOGO = ROOT / "docs" / "assets" / "manipal_logo_reference.png"
PAGE_W, PAGE_H = A4
LEFT = 75
RIGHT = 75
BOTTOM = 65
CONTENT_W = PAGE_W - LEFT - RIGHT

body = ParagraphStyle(
    "Body",
    fontName="Times-Roman",
    fontSize=10.5,
    leading=14.1,
    alignment=TA_JUSTIFY,
    spaceAfter=7.5,
    allowWidows=0,
    allowOrphans=0,
)
heading = ParagraphStyle(
    "Subsection",
    parent=body,
    fontName="Times-Bold",
    fontSize=11.5,
    leading=14,
    alignment=TA_LEFT,
    spaceBefore=8,
    spaceAfter=7,
    keepWithNext=True,
)
main_heading = ParagraphStyle(
    "MainHeading",
    fontName="Times-Bold",
    fontSize=16,
    leading=19,
    alignment=TA_LEFT,
    spaceAfter=16,
    keepWithNext=True,
)
section_heading = ParagraphStyle(
    "SectionHeading",
    fontName="Times-Bold",
    fontSize=13.5,
    leading=16,
    alignment=TA_LEFT,
    spaceBefore=13,
    spaceAfter=9,
    keepWithNext=True,
)
caption = ParagraphStyle(
    "TableCaption",
    fontName="Times-Roman",
    fontSize=9.2,
    leading=11,
    alignment=TA_LEFT,
    spaceBefore=6,
    spaceAfter=6,
    keepWithNext=True,
)
cell = ParagraphStyle(
    "TableCell",
    fontName="Times-Roman",
    fontSize=9.1,
    leading=11.2,
    alignment=TA_LEFT,
)
num = ParagraphStyle("TableNumber", parent=cell, alignment=TA_CENTER)
cellhead = ParagraphStyle("TableHeader", parent=cell, fontName="Times-Bold")
numhead = ParagraphStyle("TableNumberHeader", parent=cellhead, alignment=TA_CENTER)
group = ParagraphStyle("TableGroup", parent=cell, fontName="Times-Bold", fontSize=9.1)
note = ParagraphStyle(
    "TableNote",
    fontName="Times-Roman",
    fontSize=8.7,
    leading=10.4,
    alignment=TA_LEFT,
    spaceBefore=2,
    spaceAfter=9,
)
figure_caption = ParagraphStyle(
    "FigureCaption",
    fontName="Times-Roman",
    fontSize=9.2,
    leading=11,
    alignment=TA_LEFT,
    spaceBefore=4,
    spaceAfter=10,
)


def p(text):
    return Paragraph(text, body)


def h(text):
    return Paragraph(text, heading)


def main_h(text):
    return Paragraph(text, main_heading)


def section_h(text):
    return Paragraph(text, section_heading)


def table_block(label, title, headers, rows, widths, footnote, groups=None):
    data = [[Paragraph(v, cellhead if i == 0 else numhead) for i, v in enumerate(headers)]]
    for row in rows:
        if isinstance(row, str):
            data.append([Paragraph(row, group)] + [""] * (len(headers) - 1))
        else:
            data.append([
                Paragraph(str(value), cell if i == 0 else num)
                for i, value in enumerate(row)
            ])
    table = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=1)
    rules = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LINEABOVE", (0, 0), (-1, 0), 0.55, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.4, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.55, colors.black),
    ]
    if groups:
        for row_idx in groups:
            rules.append(("SPAN", (0, row_idx), (-1, row_idx)))
            rules.append(("LINEABOVE", (0, row_idx), (-1, row_idx), 0.25, colors.black))
    table.setStyle(TableStyle(rules))
    return KeepTogether([
        Paragraph(f"<b>Table {label}</b> {title}", caption),
        table,
        Paragraph(footnote, note),
    ])


def paired_rmse_figure():
    """Vector chart of a single paired test cohort (no cross-year bars)."""
    d = Drawing(CONTENT_W, 150)
    x0, scale = 160, 5.35
    for tick in (0, 10, 20, 30, 40):
        x = x0 + tick * scale
        d.add(Line(x, 25, x, 137, strokeColor=colors.HexColor("#d2d2d2"), strokeWidth=0.45))
        d.add(String(x, 10, str(tick), fontName="Times-Roman", fontSize=8.5, textAnchor="middle"))
    d.add(String(x0 + 20 * scale, 0, "RMSE (AQI points)", fontName="Times-Roman", fontSize=8.5, textAnchor="middle"))
    bars = [
        ("Weather XGBoost", 30.270, 110, colors.HexColor("#2f2f2f")),
        ("Frozen GRU", 31.707, 75, colors.HexColor("#777777")),
        ("Persistence", 34.668, 40, colors.HexColor("#b7b7b7")),
    ]
    for label, value, y, fill in bars:
        d.add(String(0, y + 5, label, fontName="Times-Roman", fontSize=9.5))
        d.add(Rect(x0, y, value * scale, 17, fillColor=fill, strokeColor=None))
        d.add(String(x0 + value * scale + 6, y + 4, f"{value:.3f}", fontName="Times-Roman", fontSize=9.2))
    return KeepTogether([
        d,
        Paragraph("<b>Fig. 1</b> RMSE on the same 20,086 January-September 2026 station-days. Lower is better; these bars do not compare different test years.", figure_caption),
    ])


def draw_cover(c, doc):
    c.saveState()
    cx = PAGE_W / 2
    logo_w = 180
    logo_h = logo_w * 429 / 1042
    c.drawImage(ImageReader(str(LOGO)), cx - logo_w / 2, 666, width=logo_w, height=logo_h, mask="auto")
    c.setFont("Times-Roman", 11)
    c.drawCentredString(cx, 626, "School of Computer Engineering")
    c.setFont("Times-Bold", 17)
    c.drawCentredString(cx, 602, "Manipal Institute of Technology")
    c.setFont("Times-Roman", 11)
    c.drawCentredString(cx, 587, "Manipal Academy of Higher Education")
    c.drawCentredString(cx, 572, "Manipal, Karnataka, India")

    title_style = ParagraphStyle("CoverTitle", fontName="Times-Bold", fontSize=18, leading=22, alignment=TA_CENTER)
    title = Paragraph("Auditing the Faithfulness and Cross-City Consistency<br/>of Attention in Multimodal Spatiotemporal Air<br/>Quality Index Prediction", title_style)
    tw, th = title.wrap(470, 100)
    title.drawOn(c, cx - tw / 2, 500 - th)
    c.setFont("Times-Roman", 15)
    c.drawCentredString(cx, 402, "Results, Discussion and Inference")
    c.setFont("Times-Roman", 10.5)
    c.drawCentredString(cx, 385, "ML Lab Project - 20 October 2026 | Revised 21 October 2026")
    c.setFont("Times-Italic", 10.5)
    c.drawCentredString(cx, 368, "Including same-day estimation and next-day forecasting")

    cover_data = [
        ["Team member", "Registration number", "Roll number"],
        ["Saanvi Garg", "240905702", "68"],
        ["Sparsh Agrawal", "240905708", "70"],
        ["Dhruv Chamria", "240905706", "69"],
        ["Neelaksha Sisodiya", "240905642", "65"],
    ]
    cover_table = Table(cover_data, colWidths=[150, 135, 75])
    cover_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("LEADING", (0, 0), (-1, -1), 12),
        ("LINEABOVE", (0, 0), (-1, 0), 0.6, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.45, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, colors.black),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    _, cover_h = cover_table.wrap(360, 150)
    cover_table.drawOn(c, cx - 180, 335 - cover_h)
    c.setFont("Times-Roman", 11)
    c.drawCentredString(cx, 174, "Section CSE-C")
    c.drawCentredString(cx, 145, "Faculty guide: Dr Nisha P Shetty")
    c.drawCentredString(cx, 130, "Academic year: 2026-2027")
    c.restoreState()


def draw_body(c, doc):
    c.saveState()
    if doc.page > 2:
        c.setFont("Times-Bold", 11.5)
        c.drawString(LEFT, PAGE_H - 39, "IV. Results, Discussion and Inference")
    c.setFont("Times-Roman", 9)
    c.drawCentredString(PAGE_W / 2, 35, str(doc.page - 1))
    c.restoreState()


def make_story():
    return [
        NextPageTemplate("Body"),
        PageBreak(),
        main_h("IV. Results, Discussion and Inference"),
        section_h("A. Experimental Results"),
        p("The completed experiments use distinct historical test periods and protocols. Within each comparison, models use the same eligible station-days. The 2024 same-day, 2025 next-day, 2026 paired forecast, 2023 chronological attention, and eight-city leave-one-city-out (LOCO) results should not be ranked across cohorts as though they shared one test distribution. The reserved October-December 2026 prospective period has no valid score yet."),
        h("A.1 Metric-wise performance and historical baselines"),
        p("Table 1 shows that the 2024 same-day Ridge model with AQI history reduced RMSE from 36.197 for persistence to 32.806 AQI points on 28,803 station-days, a 9.4% reduction. Its MAE was 23.001, compared with 23.716 for persistence. Environment-only Ridge had RMSE 63.239. In the separate 2025 next-day test, the three-seed 28-day GRU reduced RMSE from 35.388 for persistence to 32.516 on 30,263 station-days, an 8.1% reduction. Its MAE was 21.857, compared with 22.981 for persistence."),
        table_block(
            "1", "Historical performance against persistence. Each task has its own test cohort.",
            ["Model", "Station-<br/>days", "MAE", "RMSE", "R<super>2</super>"],
            [
                "Same-day estimation, 2024",
                ["Ridge with AQI history", "28,803", "23.001", "32.806", "0.892"],
                ["Persistence", "28,803", "23.716", "36.197", "0.869"],
                ["Environment-only Ridge", "28,803", "45.804", "63.239", "0.599"],
                "Next-day forecast, 2025",
                ["Three-seed GRU ensemble", "30,263", "21.857", "32.516", "0.887"],
                ["Persistence", "30,263", "22.981", "35.388", "0.867"],
            ], [173, 72, 68, 68, 64],
            "MAE and RMSE are in AQI points. Same-day estimation uses target-day environmental observations; next-day forecasting does not.",
            groups=[1, 5],
        ),
        p("The paired weekly block-bootstrap RMSE gain was 3.391 AQI points [95% interval: 2.03, 4.67] for 2024 same-day Ridge and 2.872 [2.168, 3.605] for the 2025 GRU. GRU RMSE was lower than persistence in all eight cities in 2025. The nominal 90% station prediction interval covered 95.4% of the observed 2025 targets; this is historical coverage, not a future guarantee."),
        h("A.2 Paired 2026 forecast comparison"),
        p("Weather-assisted XGBoost, the frozen GRU, and persistence were evaluated on exactly the same 20,086 station-days from January-September 2026 (91 of 107 configured stations). Table 2 and Fig. 1 show the metric-wise comparison. XGBoost had the lowest MAE and RMSE, and its RMSE was lower than GRU in each of the eight represented cities. Its paired RMSE gain was 1.437 points over GRU [95% weekly-bootstrap interval: 0.376, 2.614] and 4.398 points over persistence [3.190, 5.715]."),
        table_block(
            "2", "Paired January-September 2026 forecast comparison.",
            ["Model", "Station-<br/>days", "MAE", "RMSE", "R<super>2</super>"],
            [
                ["Weather-assisted XGBoost", "20,086", "19.345", "30.270", "0.848"],
                ["Frozen GRU ensemble", "20,086", "20.394", "31.707", "0.833"],
                ["Persistence", "20,086", "21.763", "34.668", "0.800"],
            ], [173, 72, 68, 68, 64],
            "All three scores use identical rows. MAE and RMSE are in AQI points.",
        ),
        paired_rmse_figure(),
        h("A.3 Abrupt-change and severe-AQI outcomes"),
        p("Average error did not capture the full operational tradeoff. Of 2,315 observed station-days with an absolute daily AQI change of at least 50, XGBoost detected 422 changes in the correct direction, versus 154 for GRU. It flagged 799 changes overall, versus 268 for GRU (Table 3). XGBoost therefore had higher recall but lower precision."),
        table_block(
            "3", "Sharp-change detection on the paired 2026 cohort; threshold = 50 AQI points.",
            ["Measure", "Weather XGBoost", "Frozen GRU"],
            [
                ["Observed sharp changes", "2,315", "2,315"],
                ["Forecast-flagged changes", "799", "268"],
                ["Correct directional detections", "422", "154"],
                ["Recall", "18.23%", "6.65%"],
                ["Precision", "52.82%", "57.46%"],
            ], [205, 120, 120],
            "A correct detection must exceed the threshold in the observed direction. Precision is computed over forecast-flagged changes.",
        ),
        p("For observed AQI of 301-400, the weather model's mean prediction error was -38.51 points; for 401-500 it was -44.36 points. Only 94 observations were in the latter band, so that estimate is less stable. These severe-day errors matter even though the model led the average-error metrics."),
        h("A.4 Attention accuracy and faithfulness"),
        p("The corrected attention study used a separate November-December 2023 chronological test of 5,434 station-days. Within the seven-day sequence family, temporal attention (TA), feature attention (FA), and their combination all had higher RMSE than uniform weighting. Within the spatial-patch family, spatial attention (S-SA) reduced RMSE from 75.328 to 65.836 (Table 4)."),
        table_block(
            "4", "Corrected attention experiment, November-December 2023.",
            ["Model family", "Variant", "Station-<br/>days", "RMSE"],
            [
                ["Seven-day sequence", "Uniform weights", "5,434", "78.361"],
                ["Seven-day sequence", "Temporal attention (TA)", "5,434", "84.394"],
                ["Seven-day sequence", "Feature attention (FA)", "5,434", "82.284"],
                ["Seven-day sequence", "TA + FA", "5,434", "94.525"],
                ["Spatial patch", "Uniform weights", "5,434", "75.328"],
                ["Spatial patch", "Temporal attention (S-TA)", "5,434", "83.641"],
                ["Spatial patch", "Spatial attention (S-SA)", "5,434", "65.836"],
                ["Spatial patch", "S-TA + S-SA", "5,434", "75.966"],
            ], [120, 184, 74, 67],
            "RMSE is in AQI points. Compare variants only within the same model family.",
        ),
        p("Faithfulness was audited on 300 sampled rows for each corrected FA and S-SA checkpoint. Table 5 reports row-weighted mean Spearman agreement between each ranking and fixed-model input-replacement sensitivity. SHAP exceeded attention under both replacement methods for both models; the size of the gap depended on the replacement method."),
        table_block(
            "5", "Ranking agreement with input-replacement sensitivity.",
            ["Model", "Replacement method", "Attention r", "SHAP r", "Rows"],
            [
                ["FA", "Mean replacement", "0.420", "0.664", "300"],
                ["FA", "Same-city seasonal donor", "0.392", "0.546", "300"],
                ["S-SA", "Mean replacement", "0.242", "0.419", "300"],
                ["S-SA", "Same-city seasonal donor", "0.483", "0.586", "300"],
            ], [60, 170, 75, 75, 65],
            "Values are row-weighted mean Spearman correlations, not changes in forecast accuracy.",
        ),
        h("A.5 Corrected cross-city transfer"),
        p("The completed SWT-w7 LOCO experiment trained each fold without one city and evaluated the held-out city on all eligible test rows. Across eight cities and 29,923 station-days, Table 6 reports both equal-city-weighted and pooled scores. Within the sequence family, TA had city-macro RMSE 55.23 versus 55.11 for Uniform, despite a slightly lower pooled RMSE (71.61 versus 71.67). Within the spatial-patch family, S-TA+SA reduced city-macro RMSE from 57.19 to 54.16 and pooled RMSE from 71.73 to 70.76. It improved five of eight held-out cities; sequence TA improved four of eight. These are separate historical transfer experiments, not prospective forecast scores."),
        table_block(
            "6", "Corrected SWT-w7 LOCO evaluation: eight held-out cities and 29,923 station-days.",
            ["Family", "Variant", "City-macro<br/>MAE", "City-macro<br/>RMSE", "Pooled<br/>RMSE"],
            [
                ["Sequence", "Uniform", "40.42", "55.11", "71.67"],
                ["Sequence", "TA", "40.45", "55.23", "71.61"],
                ["Sequence", "TA + FA", "42.75", "57.96", "73.80"],
                ["Sequence", "FA", "44.67", "60.27", "75.95"],
                ["Spatial patch", "S-Uniform", "42.28", "57.19", "71.73"],
                ["Spatial patch", "S-TA", "42.04", "56.34", "71.91"],
                ["Spatial patch", "S-SA", "41.78", "56.79", "73.12"],
                ["Spatial patch", "S-TA + S-SA", "40.20", "54.16", "70.76"],
            ], [88, 122, 78, 82, 75],
            "MAE and RMSE are AQI points. City-macro averages give each held-out city equal weight; pooled RMSE weights station-days. Compare attention only with its within-family Uniform control.",
        ),
        h("A.6 Development-only event alerts"),
        p("A separate 2025 historical test examined severe AQI (at least 300) and abrupt absolute daily change (at least 50 AQI points). Alert classifiers were fitted on October 2024 GRU predictions and thresholds chosen on November-December 2024, before the 2025 evaluation. Table 7 compares them with direct thresholds on the frozen forecast. Severe-event precision rose from 83.7% to 88.4%, but recall fell from 82.9% to 64.9%. For abrupt changes, the candidate produced only nine alerts and recall fell from 4.25% to 0.20%; its 88.9% precision is therefore not evidence of better detection. Neither classifier met the prespecified promotion check, and neither was deployed."),
        table_block(
            "7", "Historical 2025 alert test; development candidates are not deployed.",
            ["Event", "Method", "Events", "Alerts", "Precision", "Recall"],
            [
                ["AQI >= 300", "Frozen forecast threshold", "2,975", "2,948", "83.68%", "82.92%"],
                ["AQI >= 300", "Candidate classifier", "2,975", "2,183", "88.41%", "64.87%"],
                ["Change >= 50", "Frozen forecast threshold", "3,975", "304", "55.59%", "4.25%"],
                ["Change >= 50", "Candidate classifier", "3,975", "9", "88.89%", "0.20%"],
            ], [82, 129, 52, 48, 67, 67],
            "All rows use the same 30,263 historical 2025 station-days. Precision and recall measure event alerts, not point-forecast accuracy.",
        ),
        section_h("B. Discussion"),
        h("B.1 Why the average-error models differed"),
        p("Persistence was a strong baseline in every forecast comparison, showing that recent AQI carries substantial predictive information. The much poorer 2024 environment-only Ridge score, alongside the better history-enhanced Ridge score, supports the importance of an AQI-history anchor in same-day estimation. The 28-day GRU improved on repeating yesterday's value in the 2025 cohort, consistent with useful information in a longer AQI sequence and known calendar/location inputs. That comparison does not isolate which individual GRU input caused the gain."),
        p("The 2026 weather-assisted XGBoost model predicts a correction to the previous-day AQI and uses archived forecasts available before the target day. Its lower paired error is consistent with a useful combination of that anchor, nonlinear residual modeling, and forecast-weather context. Because the comparator is a different architecture, this test alone cannot measure a weather-only causal contribution. The observed gain is a model-level result, not proof that any single weather variable explains it."),
        h("B.2 Where the gain weakened"),
        p("The weather candidate flagged roughly three times as many sharp changes as GRU (799 versus 268), which helps explain its higher recall and lower precision. Its underprediction in high-AQI bands shows why a lower overall RMSE can coexist with weak performance on severe episodes. Rare extremes, incomplete station coverage, and different city weights make the pooled result insufficient as an operational safety criterion. Delhi contributed 12,387 of 30,263 rows in the 2025 GRU test (40.9%), so that pooled metric gives Delhi substantial influence."),
        p("The promotion rule required both sharp-change recall and precision to be at least as high as GRU's, in addition to positive paired RMSE gains. XGBoost missed the precision condition (52.82% versus 57.46%), so GRU remained the default next-day model despite XGBoost's lower average error. This decision follows the predeclared rule rather than choosing a model after viewing only its best metric."),
        h("B.3 Why attention accuracy is not explanation faithfulness"),
        p("Spatial attention improved the spatial-patch model on its own 2023 test, but FA and TA did not improve the sequence model. Learned weighting can therefore help one architecture without reliably helping another. The faithfulness audit further separates accuracy from interpretability: attention rankings agreed less with replacement sensitivity than SHAP in all four comparisons. For S-SA under mean replacement, the highest-attention group's deletion changed output by 72.53 AQI points, close to 70.46 for a random ranking; SHAP's top group changed it by 73.76. This weak separation limits an explanation claim for that deletion test."),
        p("Replacement is an intervention on a fixed model, not a controlled experiment on atmospheric causes. Correlated inputs and unrealistic mean replacements can change rankings; the different same-city seasonal-donor correlations in Table 5 demonstrate that sensitivity. Thus, neither an attention heat map nor one perturbation score establishes the physical source of pollution."),
        h("B.4 Why cross-city conclusions differ from pooled conclusions"),
        p("The corrected LOCO study directly tests transfer to an unseen city. It changes the previous inference that cross-city validation was still pending. The spatial S-TA+SA variant outperformed its within-family Uniform control on the equal-city average and in five cities, suggesting that joint temporal and spatial weighting can help this architecture transfer across locations. The gain is not uniform: three cities did not improve. Sequence TA illustrates why aggregation matters: a small pooled RMSE gain occurred alongside a small city-macro RMSE loss. Delhi supplied 11,658 of the 29,923 LOCO station-days, so the pooled score is influenced strongly by that city's error. Equal-city averaging answers a different question from pooled station-day error."),
        p("The LOCO checkpoints were evaluated for predictive accuracy, not with the 300-row replacement-based faithfulness audit used for the separate chronological checkpoints. Therefore the transfer gain cannot be used to claim that their attention maps are faithful explanations. Likewise, the LOCO architecture uses different inputs and training folds from the operational GRU and weather XGBoost; its scores do not justify replacing the frozen next-day default."),
        h("B.5 Why higher alert precision did not justify promotion"),
        p("The severe-event candidate issued 765 fewer 2025 alerts than the frozen forecast threshold. That more selective behavior increased precision but missed many additional true severe days, reducing recall by 18.05 percentage points. The abrupt-change candidate was still more selective: nine alerts for 3,975 observed events, with eight correct, so high precision describes a very small subset and almost no coverage. This is consistent with the difficulty of predicting sudden AQI changes from the current forecast and prior-day AQI alone. Both candidates failed the calibration promotion rule; their 2025 results are research findings only."),
        section_h("C. Inference"),
        p("The research gap has three distinct answers. First, recent AQI is a strong baseline, while history-aware and weather-assisted models improved average error within their own paired historical cohorts. Second, the corrected LOCO experiment supplies evidence of partial cross-city transfer: S-TA+SA improved the eight-city average relative to S-Uniform, but not every city; sequence TA did not improve the equal-city average. Third, attention-based accuracy and explanation quality remain separate. On sampled chronological checkpoints, attention rankings agreed less with fixed-model replacement sensitivity than SHAP; the LOCO checkpoints have not yet received that audit. Neither analysis supports a causal claim about pollution sources."),
        p("Lower RMSE alone is insufficient for deployment. The weather XGBoost candidate failed the predeclared sharp-change precision condition, and the event classifiers lost too much recall and failed their own promotion checks. The frozen GRU therefore remains the next-day default. The 30% GRU / 70% XGBoost blend was selected on earlier development data but has no independent prospective score and is excluded from the validated comparisons."),
        p("The October-December 2026 prospective protocol remains reserved. As of 21 October 2026, the CPCB October collection had no usable new station-days or immutable before-target forecast records, so there is no prospective metric to report. A valid future comparison requires forecasts recorded before target-day UTC midnight from eligible prior-day AQI and earlier-issued weather, followed by quality-checked observations. Historical and cross-city results should be interpreted as evidence for the specified evaluation settings, not as verified future performance."),
        p("<b>Reproducibility note:</b> Numerical results are traceable to the project's saved forecast metrics (results/forecast_2025 and results/forecast_2026), corrected LOCO summary (results/city_transfer/model_summary.csv), alert metrics (results/event_alerts/metrics.csv), attention audits (results/followthrough/oct12-v1), and prospective status (results/prospective/status.json)."),
    ]


def build():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(
        str(OUTPUT), pagesize=A4, leftMargin=LEFT, rightMargin=RIGHT,
        topMargin=65, bottomMargin=BOTTOM,
        title="AQI Prediction - Results, Discussion and Inference",
        author="Saanvi Garg; Sparsh Agrawal; Dhruv Chamria; Neelaksha Sisodiya",
        subject="Results and discussion using verified project results",
    )
    cover_frame = Frame(LEFT, BOTTOM, CONTENT_W, PAGE_H - 2 * BOTTOM, id="cover", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    body_frame = Frame(LEFT, BOTTOM, CONTENT_W, PAGE_H - BOTTOM - 67, id="body", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates([
        PageTemplate(id="Cover", frames=[cover_frame], onPage=draw_cover),
        PageTemplate(id="Body", frames=[body_frame], onPage=draw_body),
    ])

    doc.build(make_story())
    print(OUTPUT)


if __name__ == "__main__":
    build()
