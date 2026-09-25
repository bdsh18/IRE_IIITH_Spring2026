"""Create the Assignment 2 dual-dataset design note PDF.

The report intentionally distinguishes local temporal validation from each
Codabench result and status. It is a submission artifact, not a model-training
script.
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs" / "Design_Note_A2.pdf"
PAGE_SIZE = letter
MIND_SUBMISSION_SCREENSHOT = next(
    Path("/Users/bidishashaw/Desktop").glob("Screenshot 2026-09-20 at 1.38.27*PM.png"),
    None,
)
EB_SUBMISSION_SCREENSHOT = next(
    Path("/Users/bidishashaw/Desktop").glob("Screenshot 2026-09-20 at 2.09.51*PM.png"),
    None,
)
MIND_LEADERBOARD_SCREENSHOT = next(
    Path("/Users/bidishashaw/Desktop").glob("Screenshot 2026-09-20 at 5.00.02*PM.png"),
    None,
)


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def table(data, widths, *, header=True, font_size=8.4):
    rendered = [[cell if isinstance(cell, Paragraph) else p(str(cell), STYLES["Table"]) for cell in row] for row in data]
    value = Table(rendered, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#B7C7D6")),
    ]
    if header:
        commands += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EDF2")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
    value.setStyle(TableStyle(commands))
    return value


def heading(text: str):
    return [Spacer(1, 5), p(text, STYLES["H1"]), Spacer(1, 4)]


def subheading(text: str):
    return [Spacer(1, 4), p(text, STYLES["H2"]), Spacer(1, 2)]


def bullet(text: str):
    return p(f'<bullet>&bull;</bullet>{text}', STYLES["BulletCustom"])


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#6A6A6A"))
    canvas.drawString(doc.leftMargin, PAGE_SIZE[1] - 0.42 * inch, "CS4.406 - Information Retrieval & Extraction | Assignment 2")
    canvas.drawRightString(PAGE_SIZE[0] - doc.rightMargin, 0.42 * inch, f"News Retrieval Design Note | {doc.page}")
    canvas.restoreState()


STYLES = getSampleStyleSheet()
STYLES.add(ParagraphStyle(
    name="TitleCustom", parent=STYLES["Title"], fontName="Helvetica-Bold",
    fontSize=19, leading=23, textColor=colors.HexColor("#123154"), alignment=TA_CENTER,
    spaceAfter=8,
))
STYLES.add(ParagraphStyle(
    name="Subtitle", parent=STYLES["Normal"], fontName="Helvetica", fontSize=10.5,
    leading=13.2, alignment=TA_CENTER, textColor=colors.black, spaceAfter=8,
))
STYLES.add(ParagraphStyle(
    name="H1", parent=STYLES["Heading1"], fontName="Helvetica-Bold", fontSize=14,
    leading=17, textColor=colors.HexColor("#123B5D"), spaceBefore=0, spaceAfter=0,
))
STYLES.add(ParagraphStyle(
    name="H2", parent=STYLES["Heading2"], fontName="Helvetica-Bold", fontSize=11.5,
    leading=14, textColor=colors.HexColor("#1E5A88"), spaceBefore=0, spaceAfter=0,
))
STYLES.add(ParagraphStyle(
    name="Body", parent=STYLES["BodyText"], fontName="Helvetica", fontSize=11,
    leading=14.2, textColor=colors.HexColor("#1A2730"), spaceAfter=7,
))
STYLES.add(ParagraphStyle(
    name="BulletCustom", parent=STYLES["BodyText"], fontName="Helvetica", fontSize=10.6,
    leading=13.4, leftIndent=15, firstLineIndent=-10, bulletIndent=0, spaceAfter=4,
))
STYLES.add(ParagraphStyle(
    name="Table", parent=STYLES["BodyText"], fontName="Helvetica", fontSize=8.4,
    leading=10.2, textColor=colors.HexColor("#17232C"), alignment=TA_LEFT,
))
STYLES.add(ParagraphStyle(
    name="Caption", parent=STYLES["BodyText"], fontName="Helvetica-Oblique", fontSize=8.7,
    leading=10.5, textColor=colors.HexColor("#526777"), spaceBefore=3, spaceAfter=5,
))
STYLES.add(ParagraphStyle(
    name="Callout", parent=STYLES["BodyText"], fontName="Helvetica", fontSize=10.4,
    leading=13.2, textColor=colors.HexColor("#123B5D"), spaceAfter=0,
))


def callout(title: str, text: str):
    body = p(f"<b>{title}</b><br/>{text}", STYLES["Callout"])
    block = Table([[body]], colWidths=[6.25 * inch])
    block.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EAF3F8")),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#7BA8C6")),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return block


def build_story():
    story = []

    # Page 1: scope and headline result.
    story += [
        Spacer(1, 0.18 * inch),
        p("Assignment 2 Design Note", STYLES["TitleCustom"]),
        p("Behaviour-aware retrieve-then-rank news recommendation on MIND and EB-NeRD", STYLES["Subtitle"]),
        p("Assignment 2 Design Note", STYLES["Subtitle"]),
        p("CS4.406: Information Retrieval & Extraction", STYLES["Subtitle"]),
        p("Roll Nos.: 2022900023, 2026701030 | Spring 2026", STYLES["Subtitle"]),
    ]
    story += heading("1. Problem and contribution")
    story += [
        p(
            "The task is to rank the candidate articles in a news impression by click likelihood, using article text, "
            "a user's earlier clicks, session context where available, and time-sensitive article signals. I built a "
            "reproducible pipeline for English MIND and Danish EB-NeRD data, then developed a practical two-stage "
            "system: lexical and semantic retrieval signals feed a behavioural LambdaRank re-ranker.",
            STYLES["Body"],
        ),
        p(
            "The report presents local temporal experiments for both datasets and a separate large-test Codabench evidence section. "
            "The shared implementation uses a normalized schema while preserving each publisher's language, metadata, and released "
            "behavioural fields. Offline development values are explicitly distinguished from hidden-test leaderboard values.",
            STYLES["Body"],
        ),
    ]
    story += subheading("Key design decisions")
    story += [
        bullet("Use chronological rather than random interaction splits, so future clicks and popularity never leak into training features."),
        bullet("Combine BM25, recency-weighted semantic similarity, causal popularity, freshness, category/topic affinity, and position/history signals in a LightGBM LambdaRank model."),
        bullet("Keep a strict distinction between catalog retrieval experiments and in-view re-ranking, because Codabench requires a rank for every supplied candidate."),
        bullet("Stream large hidden-test archives, checkpoint output, validate each rank permutation, and zip only the required root-level prediction file."),
    ]
    story.append(PageBreak())

    # Page 2: temporal protocol and feature store.
    story += heading("2. Data protocol and leakage boundary")
    story += [
        p(
            "Raw files are parsed into shared articles and impressions tables. An impression contains a timestamp, "
            "candidate article IDs, clicked IDs, user history, and optional session fields. Data are partitioned "
            "chronologically: old interactions train the model, the subsequent period validates choices, and the latest "
            "local period is held out. There is no random split of interaction rows.",
            STYLES["Body"],
        ),
    ]
    # Continue page 2 with the reproducible feature store.
    story += heading("3. Reproducible feature pipeline")
    story += [
        table([
            ["Local development store", "Catalog", "Train impressions", "Validation", "Held-out test"],
            ["MIND-small", "65,238", "126,690", "30,269", "73,158"],
            ["EB-NeRD demo/small", "11,777", "42,202", "3,962", "3,916"],
        ], [1.63 * inch, 0.86 * inch, 1.31 * inch, 1.16 * inch, 1.29 * inch], font_size=7.7),
        p("Table 1. Processed temporal development stores used for fast, reproducible local experiments.", STYLES["Caption"]),
        p(
            "Feature construction is causal. For an event at time t, popularity counts only clicks before t; the item "
            "clicked at t and all later clicks are excluded. Session aggregates are scanned by timestamp and use only "
            "prior impressions in the same session. MIND lacks native session and dwell-time fields, so those values are "
            "zero rather than treating a user's full history as one artificial session. For EB-NeRD, released histories are "
            "treated as upstream-causal and are never augmented from later logs.",
            STYLES["Body"],
        ),
    ]
    story += subheading("Candidate-level feature store")
    story += [
        table([
            ["Family", "Signal and reason"],
            ["Lexical", "BM25 score over title + abstract. The query concatenates titles from the five most recent clicks."],
            ["Semantic", "Recency-weighted user/article similarity. MIND uses 100-dimensional entity-derived vectors; EB-NeRD uses supplied 300-dimensional document vectors."],
            ["History preference", "Category affinity, topic affinity, and history length. Recent clicks get larger weight (0.8 per step from old to new)."],
            ["Article state", "Causal click popularity and freshness. These model topical demand and rapid news decay."],
            ["Context", "Input-list position bias and, where safely available, prior session aggregates. MIND session/dwell fields are absent; test-time unavailable fields are excluded from serving."],
        ], [1.45 * inch, 4.8 * inch]),
        p("Table 2. Re-ranker inputs. The final submission schema is constrained to features available in released test inputs.", STYLES["Caption"]),
    ]
    story.append(PageBreak())

    # Page 3: candidate generation and model results.
    story += heading("4. Candidate generation and ranking protocol")
    story += [
        p(
            "Stage 1 is BM25 over the complete catalog, using k1 = 1.5 and b = 0.75. For global-catalog diagnostics it "
            "retrieves top-200 articles and excludes articles already clicked by the user. Recall is modest because news is "
            "time-sensitive and a user's next click often differs from old history.",
            STYLES["Body"],
        ),
        table([
            ["Dataset", "BM25 Recall@50 / @100 / @200", "Semantic Recall@50 / @100 / @200"],
            ["MIND", "0.01348 / 0.02152 / 0.03219", "0.00351 / 0.00621 / 0.01045"],
            ["EB-NeRD", "0.00959 / 0.01615 / 0.03054", "0.00833 / 0.01388 / 0.02764"],
        ], [1.15 * inch, 2.55 * inch, 2.55 * inch], font_size=7.9),
        p("Table 3. Local open-catalog retrieval recall. Semantic search is a brute-force cosine baseline, not a production ANN service.", STYLES["Caption"]),
        p(
            "Stage 2 is an in-view LightGBM LambdaRank re-ranker: candidates are grouped by impression and binary click labels "
            "are optimized with a listwise objective. For evaluation, the code reports both normal in-view ranking and a separate "
            "retrieval-gated diagnostic. The former is the right protocol for Codabench, which supplies the candidate list and requires "
            "a complete rank permutation; it is not mislabeled as full open-catalog evaluation.",
            STYLES["Body"],
        ),
    ]
    # Continue page 3 with model and local performance.
    story += heading("5. Re-ranker architecture and local results")
    story += [
        p(
            "The primary ranker is LightGBM LGBMRanker with the LambdaRank objective, 200 trees, 31 leaves, and a learning "
            "rate of 0.05. This choice is lightweight enough for tabular click features while allowing nonlinear interactions, such as "
            "a category match becoming more useful when an article is fresh. A histogram gradient-boosting fallback keeps the project "
            "runnable on macOS if LightGBM/OpenMP is unavailable, but all reported values below use LambdaRank.",
            STYLES["Body"],
        ),
        table([
            ["Dataset / validation", "Method", "AUC", "MRR", "nDCG@5", "nDCG@10"],
            ["MIND (30,269)", "BM25", "0.54512", "0.28294", "0.25438", "0.31106"],
            ["MIND (30,269)", "LambdaRank", "0.59020", "0.30192", "0.27650", "0.33589"],
            ["EB-NeRD (3,962)", "BM25", "0.49933", "0.31375", "0.34496", "0.42899"],
            ["EB-NeRD (3,962)", "LambdaRank", "0.65124", "0.40594", "0.45562", "0.52147"],
        ], [1.65 * inch, 1.00 * inch, 0.85 * inch, 0.85 * inch, 0.95 * inch, 0.95 * inch], font_size=7.65),
        p("Table 4. Local chronological experiments. These are local validation metrics, not hidden-test leaderboard scores.", STYLES["Caption"]),
        p(
            "All four BM25-to-re-ranker improvements are statistically supported by a paired 2,000-resample bootstrap. "
            "The 95% confidence intervals for the differences are AUC [0.04125, 0.04881], MRR [0.01623, 0.02183], "
            "nDCG@5 [0.01907, 0.02508], and nDCG@10 [0.02217, 0.02750] on MIND. EB-NeRD likewise has positive "
            "reranker-versus-BM25 intervals; their exact values appear in Table 8. Every reported interval excludes zero.",
            STYLES["Body"],
        ),
    ]
    story += [KeepTogether([
        *subheading("Offline MINDlarge development validation"),
        p(
            "This is an offline development experiment, not a Codabench leaderboard result. MINDlarge_train trained the model and the temporally later official MINDlarge_dev "
            "split selected the configuration. Across 376,471 development impressions, the model achieved AUC 0.61820, MRR 0.34163, "
            "nDCG@5 0.32400, and nDCG@10 0.38310. The causal popularity-only reference achieved 0.53857, 0.26179, 0.24527, and "
            "0.30787, respectively. The model metric bootstrap intervals are narrow, for example AUC [0.61734, 0.61914]. "
            "EB-NeRD local validation remains the engineering comparison reported in this note.",
            STYLES["Body"],
        ),
        callout(
            "Interpretation",
            "The large-dev results were used for model selection. They should not be directly compared with the smaller local "
            "MIND experiment because the data scale, split, and candidate distributions differ.",
        ),
    ])]
    story += heading("6. Why this improves the baseline")
    story += [
        p(
            "BM25 captures direct topical overlap, but it cannot model which category, entity profile, freshness level, and popularity pattern "
            "best match a particular user. The re-ranker learns these interactions from prior behavior. Semantic similarity helps when lexical "
            "terms do not overlap, while causal popularity supplies a safe demand prior for sparse histories.",
            STYLES["Body"],
        ),
    ]
    story += subheading("Cross-dataset observations")
    story += [
        table([
            ["Dataset", "Lexical versus semantic finding", "Effect of behavioural re-ranking"],
            ["MIND", "Semantic AUC is higher (0.555 vs. 0.544), but BM25 has higher nDCG@10 (0.311 vs. 0.301).", "LambdaRank has the strongest local relevance results; history features improve top-rank ordering."],
            ["EB-NeRD", "Semantic slightly exceeds BM25 in AUC (0.507 vs. 0.499) and nDCG@10 (0.435 vs. 0.429).", "LambdaRank produces a much larger local lift, especially for fresh and preference-matched items."],
        ], [0.85 * inch, 2.65 * inch, 2.75 * inch], font_size=7.45),
        p("Table 5. Retrieval observations from the extended local evaluation. Values are not comparable across publishers as absolute quality scores.", STYLES["Caption"]),
        p(
            "The lexical-semantic relationship is dataset-dependent: text overlap is relatively more useful at the top of MIND rankings, while the supplied EB-NeRD vectors offer a small standalone gain. "
            "In both datasets, the behavioural re-ranker is the main source of improvement because it can combine user preference, freshness, and popularity safely over time.",
            STYLES["Body"],
        ),
    ]
    story.append(PageBreak())

    # Page 5: large-test leaderboard evidence only.
    story += heading("7. Large-test Codabench evidence")
    story += [
        p(
            "This section contains only submissions made with the required large hidden-test archives: MINDlarge_test and ebnerd_testset. "
            "The local small/demo and development experiments elsewhere in this report are offline diagnostics and are not leaderboard submissions.",
            STYLES["Body"],
        ),
        table([
            ["Competition", "Large test input", "Submission", "Status", "Visible result"],
            ["MIND", "MINDlarge_test", "mind_v2_submission.zip (ID 934568)", "Finished", "AUC 0.6290; MRR 0.3071; nDCG@5 0.3295; nDCG@10 0.3860"],
            ["EB-NeRD", "ebnerd_testset", "ebnerd_personalized_submission_A2.zip (ID 934830)", "Submitted", "No numerical score at capture"],
        ], [0.80 * inch, 1.05 * inch, 1.65 * inch, 0.75 * inch, 2.00 * inch], font_size=7.2),
        p("Table 6. Large-test Codabench evidence. EB-NeRD is correctly reported as submitted, not scored.", STYLES["Caption"]),
    ]
    if MIND_SUBMISSION_SCREENSHOT is not None and MIND_SUBMISSION_SCREENSHOT.exists():
        mind_submission = Image(str(MIND_SUBMISSION_SCREENSHOT), width=6.25 * inch, height=0.67 * inch)
        story += [mind_submission, p("Figure 1. MIND Codabench large-test submission: mind_v2_submission.zip (ID 934568), finished on 20 September 2026; displayed score 0.6290.", STYLES["Caption"])]
    if MIND_LEADERBOARD_SCREENSHOT is not None and MIND_LEADERBOARD_SCREENSHOT.exists():
        mind_leaderboard = Image(str(MIND_LEADERBOARD_SCREENSHOT), width=6.25 * inch, height=3.20 * inch)
        story += [mind_leaderboard, p("Figure 2. MIND Codabench leaderboard row for submission ID 934568. In the official metric order, the visible values are AUC 0.6290, MRR 0.3071, nDCG@5 0.3295, and nDCG@10 0.3860.", STYLES["Caption"])]
    if EB_SUBMISSION_SCREENSHOT is not None and EB_SUBMISSION_SCREENSHOT.exists():
        eb_submission = Image(str(EB_SUBMISSION_SCREENSHOT), width=6.25 * inch, height=0.85 * inch)
        story += [eb_submission, p("Figure 3. EB-NeRD Codabench large-test submission status on 20 September 2026. The A2 archive is Submitted; the earlier archive is Scoring. No EB-NeRD score was available at capture time.", STYLES["Caption"])]
    story.append(PageBreak())

    # Page 6: ablations and broader local evaluation.
    story += heading("8. Baseline, improvement, and statistical significance")
    story += subheading("MIND ablation: category affinity")
    story += [
        p(
            "The principal controlled MIND improvement is category affinity: the fraction of a user's recency-weighted history assigned to the candidate's "
            "category. I remove this feature, retrain on identical data, and pair full-model and ablated metrics within each impression. This isolates a "
            "behavioural preference signal rather than merely changing model capacity.",
            STYLES["Body"],
        ),
        table([
            ["Full model minus ablated model", "Delta", "95% paired bootstrap CI", "Zero excluded?"],
            ["AUC", "+0.02542", "[0.02287, 0.02806]", "Yes"],
            ["MRR", "+0.01409", "[0.01192, 0.01636]", "Yes"],
            ["nDCG@5", "+0.01572", "[0.01353, 0.01798]", "Yes"],
            ["nDCG@10", "+0.01655", "[0.01455, 0.01861]", "Yes"],
        ], [1.95 * inch, 1.0 * inch, 2.2 * inch, 1.1 * inch]),
        p("Table 7. MIND local feature ablation, 2,000 paired bootstrap resamples.", STYLES["Caption"]),
        p(
            "The positive confidence intervals show that category-aware matching is not a random fluctuation in this local temporal split. The internal causal "
            "temporal-popularity baseline is deliberately named as an internal reference rather than an official NRMS reproduction. The code supports strict comparison "
            "against an externally supplied starter prediction file when it is available on the identical validation split.",
            STYLES["Body"],
        ),
    ]
    story += subheading("EB-NeRD local re-ranker improvement")
    story += [
        table([
            ["Metric", "BM25", "LambdaRank", "Delta and 95% paired CI"],
            ["AUC", "0.49933", "0.65124", "+0.15190 [0.13998, 0.16499]"],
            ["MRR", "0.31375", "0.40594", "+0.09219 [0.08079, 0.10314]"],
            ["nDCG@5", "0.34496", "0.45562", "+0.11067 [0.09829, 0.12295]"],
            ["nDCG@10", "0.42899", "0.52147", "+0.09248 [0.08246, 0.10215]"],
        ], [0.90 * inch, 1.05 * inch, 1.15 * inch, 3.15 * inch], font_size=7.7),
        p("Table 8. EB-NeRD local reranker-versus-BM25 comparison, 3,962 impressions and paired bootstrap confidence intervals.", STYLES["Caption"]),
    ]
    story += heading("9. Extended evaluation")
    story += [
        table([
            ["Dataset / local re-ranker", "ILD@10", "Novelty@10", "Coverage@10"],
            ["MIND", "0.6393", "15.515", "0.04500"],
            ["EB-NeRD", "0.1775", "14.262", "0.06160"],
        ], [2.60 * inch, 1.15 * inch, 1.35 * inch, 1.15 * inch]),
        p("Table 9. Beyond-accuracy local snapshots. Raw diversity values are dataset-specific because the two embedding spaces differ.", STYLES["Caption"]),
        p(
            "The re-ranker improves relevance but lowers novelty relative to BM25, an expected accuracy-novelty trade-off: behavioural and popularity signals concentrate "
            "recommendations on more familiar content. MIND reports cold-versus-warm and head-versus-tail slices. EB-NeRD reports head-versus-tail slices; its local validation subset "
            "contains no cold-history impressions, so a cold slice is explicitly unavailable rather than imputed.",
            STYLES["Body"],
        ),
    ]
    story.append(PageBreak())

    # Page 7: serving / scale.
    story += heading("10. Serving and scale analysis")
    story += [
        p(
            "A local single-process benchmark measures complete request work: build the BM25 history query, retrieve around 200 catalog candidates, construct features, and score "
            "the ranker. The machine is not a production server, so these values are measured local evidence and not a cloud capacity guarantee.",
            STYLES["Body"],
        ),
        table([
            ["Measured local quantity", "MIND", "EB-NeRD"],
            ["BM25 postings estimate", "103.2 MB", "14.1 MB"],
            ["Embedding index", "22.5 MB (100-d)", "14.1 MB (300-d)"],
            ["Feature-store footprint", "56.5 MB", "44.5 MB"],
            ["End-to-end p50 / p95 / p99", "278 / 378 / 464 ms", "166 / 414 / 571 ms"],
            ["Single-process throughput", "3.64 requests/s", "5.06 requests/s"],
            ["Semantic-only p99", "95.2 ms", "93.6 ms"],
            ["1000 QPS rough estimate", "275 workers; USD 137.50/h", "198 workers; USD 99.00/h"],
        ], [2.10 * inch, 2.075 * inch, 2.075 * inch], font_size=7.65),
        p("Table 10. Measured local serving snapshots over 300 validation requests. The corpora and hardware profile differ, so this is a scale diagnostic, not a fair speed race.", STYLES["Caption"]),
        p(
            "Both end-to-end p99 values miss the illustrative 100 ms SLA, while semantic-only scoring is close to the target. This identifies Python-side lexical retrieval, per-request "
            "feature construction, and ranker invocation as the dominant bottlenecks. The cost extrapolations are intentionally conservative and should not be treated as production quotes.",
            STYLES["Body"],
        ),
    ]
    story += subheading("What breaks first at 10x")
    story += [
        bullet("Dense vector search: brute-force dot products become expensive as the catalog grows. Use a measured FAISS/HNSW ANN index with recall monitoring."),
        bullet("Feature computation: Pandas reconstruction of histories and popularity is too slow online. Materialize time-safe, versioned features in an online store such as Redis or Feast."),
        bullet("Lexical retrieval: in-memory BM25 postings need sharding, caching, or a dedicated search service as catalog and query traffic grow."),
        bullet("Ranking throughput: batch candidate matrices across requests and use concurrent workers or a model-serving endpoint instead of sequential Python calls."),
        bullet("Feature freshness: popularity, embeddings, and article metadata need explicit update SLAs, or newly published articles receive stale/default signals."),
    ]
    story += heading("11. Dataset comparison")
    story += [
        p(
            "MIND is English and supplies entity annotations that make entity-derived vectors convenient. EB-NeRD is Danish and provides document vectors, so its semantic signal comes from "
            "the supplied Word2Vec representation. Both sources share rapid news decay, but their schemas differ in language, article metadata, and session availability. The common feature API and "
            "submission validators make the experiments comparable without pretending that identical feature values have identical meaning across publishers.",
            STYLES["Body"],
        ),
    ]
    story.append(PageBreak())

    # Page 8: reproducibility / caveats / conclusion.
    story += heading("12. Reproducibility and submission procedure")
    story += [
        p(
            "The repository is organized around one-command reproducibility: build the normalized store, compute retrieval features, train/re-rank, run ablations and extended evaluation, then generate "
            "validated prediction archives. Large raw data, intermediate caches, and final prediction files are excluded from Git. The final generators use bounded batches, durable checkpoints, and strict "
            "rank-list validation before archiving.",
            STYLES["Body"],
        ),
        table([
            ["Stage", "Representative command"],
            ["Rebuild development store", "python build_pipeline.py --source-root <data-root> --output data/processed"],
            ["Strict re-ranker reports", "python reranker.py --dataset mind --serving-only --strict-two-stage; repeat with --dataset ebnerd"],
            ["Ablation and confidence intervals", "python ablation.py --dataset mind --serving-only; repeat with --dataset ebnerd"],
            ["Extended evaluation", "python offline_evaluation.py --serving-only --strict-two-stage"],
            ["Large-test prediction archives", "python make_mind_submission.py ...; python make_ebnerd_personalized_submission.py ..."],
        ], [2.05 * inch, 4.2 * inch], font_size=8.0),
        p("Table 11. Reproduction entry points. Exact data archive paths are passed as command-line arguments or environment settings.", STYLES["Caption"]),
    ]
    story += heading("13. Limitations and responsible interpretation")
    story += [
        bullet("Logged in-view metrics evaluate the supplied candidate set. The strict top-200 retrieval-gated diagnostic is reported separately rather than claiming full open-catalog performance."),
        bullet("Candidate input position can encode editorial or prior-system exposure bias. It is useful predictively but should be removed or corrected when causal exposure data are available."),
        bullet("MIND does not supply rich within-session dwell context. Those unavailable fields are excluded from the final test-time serving schema."),
        bullet("The internal temporal-popularity baseline is a transparent reference, not a claim of reproducing the official NRMS starter implementation."),
        bullet("Local temporal validation is not proof that every metric or slice will generalize identically to the hidden leaderboard data."),
    ]
    story += heading("14. Conclusion")
    story += [
        p(
            "The project delivers a measured and reproducible news-ranking pipeline: time-safe data preparation, lexical and semantic retrieval signals, a behavioural LambdaRank re-ranker, bootstrap-tested "
            "ablations, extended ranking diagnostics, serving measurements, and validated Codabench submission generation. The large-test evidence records the finished MIND "
            "result (AUC 0.6290, MRR 0.3071, nDCG@5 0.3295, nDCG@10 0.3860) separately from offline validation, while the EB-NeRD large-test submission remains pending a score. The next engineering priority is not a larger local model alone; it is productionizing fast ANN retrieval, cached causal features, and concurrent serving while preserving the same temporal "
            "leakage boundary.",
            STYLES["Body"],
        ),
    ]
    story.append(PageBreak())

    # References do not count toward the six-page target.
    story += heading("References")
    references = [
        "F. Wu, A. Q. Zhang, K. Zhang, et al. MIND: A Large-Scale Dataset for News Recommendation. ACL, 2020.",
        "J. Kruse, K. Lindskow, S. Kalloori, et al. EB-NeRD: A Large-Scale Dataset for News Recommendation. RecSys Challenge, 2024.",
        "G. Ke, Q. Meng, T. Finley, et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS, 2017.",
        "S. Robertson and H. Zaragoza. The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in Information Retrieval, 2009.",
        "J. Johnson, M. Douze, and H. Jegou. Billion-Scale Similarity Search with GPUs. IEEE Transactions on Big Data, 2019.",
    ]
    for number, reference in enumerate(references, start=1):
        story.append(p(f"[{number}] {reference}", STYLES["Body"]))
    return story


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(OUTPUT), pagesize=PAGE_SIZE,
        leftMargin=inch, rightMargin=inch, topMargin=inch, bottomMargin=0.82 * inch,
        title="Assignment 2 Design Note - MIND and EB-NeRD", author="CS4.406 Assignment 2",
    )
    document.build(build_story(), onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)


if __name__ == "__main__":
    main()
