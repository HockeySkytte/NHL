from __future__ import annotations

from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "docs" / "puckpedia_api_service_agreement_draft.md"
OUTPUT_PATH = ROOT / "docs" / "puckpedia_api_service_agreement_draft.pdf"


def escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def apply_inline_formatting(text: str) -> str:
    escaped = escape_html(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r'"([^"]+)"', r'&ldquo;\1&rdquo;', escaped)
    escaped = escaped.replace("\n", "<br/>")
    return escaped


def build_styles():
    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="ContractTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#14213d"),
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ContractSubTitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#666666"),
            spaceAfter=20,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ChapterHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=18,
            textColor=colors.HexColor("#0f172a"),
            spaceBefore=12,
            spaceAfter=8,
            borderPadding=0,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyContract",
            parent=styles["BodyText"],
            fontName="Times-Roman",
            fontSize=10.5,
            leading=15,
            alignment=TA_JUSTIFY,
            textColor=colors.black,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyIndented",
            parent=styles["BodyText"],
            fontName="Times-Roman",
            fontSize=10.5,
            leading=15,
            alignment=TA_JUSTIFY,
            textColor=colors.black,
            leftIndent=18,
            firstLineIndent=-12,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SignatureHeading",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#111827"),
            spaceBefore=8,
            spaceAfter=5,
        )
    )
    return styles


def classify_paragraph(text: str) -> str:
    stripped = text.strip()
    if re.match(r"^(\d+\.|[a-z]\.|[A-Z]\)|\([0-9]+\)|\([a-z]\))\s+", stripped):
        return "BodyIndented"
    return "BodyContract"


def parse_markdown(text: str, styles):
    story = []
    blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
    title_rendered = False

    for block in blocks:
        if block.startswith("# "):
            title = block[2:].strip()
            story.append(Spacer(1, 0.75 * inch))
            story.append(Paragraph(apply_inline_formatting(title), styles["ContractTitle"]))
            title_rendered = True
            continue

        if block.startswith("## "):
            if title_rendered and not story[-1].__class__.__name__ == "PageBreak":
                if block[3:].strip().startswith("Chapter 1."):
                    story.append(Spacer(1, 0.2 * inch))
                else:
                    story.append(Spacer(1, 0.08 * inch))
            story.append(Paragraph(apply_inline_formatting(block[3:].strip()), styles["ChapterHeading"]))
            continue

        lines = [line.rstrip() for line in block.splitlines()]
        paragraph_text = "<br/>".join(apply_inline_formatting(line) for line in lines if line.strip())

        if block == "**Draft for Discussion**":
            story.append(Paragraph(apply_inline_formatting(block), styles["ContractSubTitle"]))
            story.append(Spacer(1, 0.1 * inch))
            continue

        if block in {"**SUBSCRIBER**", "**PUCKPEDIA INC.**"}:
            story.append(Spacer(1, 0.12 * inch))
            story.append(Paragraph(apply_inline_formatting(block), styles["SignatureHeading"]))
            continue

        story.append(Paragraph(paragraph_text, styles[classify_paragraph(block)]))

    return story


def draw_page(canvas, doc):
    canvas.saveState()
    width, height = LETTER
    canvas.setStrokeColor(colors.HexColor("#d1d5db"))
    canvas.setLineWidth(0.5)
    canvas.line(doc.leftMargin, height - 0.65 * inch, width - doc.rightMargin, height - 0.65 * inch)

    canvas.setFont("Helvetica-Bold", 9)
    canvas.setFillColor(colors.HexColor("#374151"))
    canvas.drawString(doc.leftMargin, height - 0.52 * inch, "PuckPedia Inc. API Service Agreement")

    canvas.setFont("Helvetica", 8.5)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    footer_y = 0.55 * inch
    canvas.line(doc.leftMargin, footer_y + 0.18 * inch, width - doc.rightMargin, footer_y + 0.18 * inch)
    canvas.drawString(doc.leftMargin, footer_y, "Draft for Discussion")
    canvas.drawRightString(width - doc.rightMargin, footer_y, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()


def main() -> None:
    styles = build_styles()
    markdown = SOURCE_PATH.read_text(encoding="utf-8")
    story = parse_markdown(markdown, styles)

    doc = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=LETTER,
        leftMargin=0.9 * inch,
        rightMargin=0.9 * inch,
        topMargin=0.9 * inch,
        bottomMargin=0.8 * inch,
        title="PuckPedia Inc. API Service Agreement",
        author="GitHub Copilot",
        subject="Draft contract for discussion",
    )
    doc.build(story, onFirstPage=draw_page, onLaterPages=draw_page)
    print(f"Created {OUTPUT_PATH}")


if __name__ == "__main__":
    main()