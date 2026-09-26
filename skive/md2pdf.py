"""Render a Markdown file to a clean PDF (no syntax highlighting -> no black bars).

    python scripts/md2pdf.py INPUT.md OUTPUT.pdf ["Title"] ["Author"]

Put a line `<!-- pagebreak -->` in the Markdown before a table-heavy section so
the table never breaks across pages (the PDF engine otherwise leaves orphaned /
ghost header rows). Requires: pip install markdown pymupdf
"""
import os
import re
import sys

import fitz  # pymupdf
import markdown

src, out = sys.argv[1], sys.argv[2]
title = sys.argv[3] if len(sys.argv) > 3 else "Document"
author = sys.argv[4] if len(sys.argv) > 4 else "Amit Agarwal"
# PDF_LANDSCAPE=1 -> landscape letter + smaller table font (for 12+ column tables)
landscape = os.environ.get("PDF_LANDSCAPE") == "1"

md = open(src, encoding="utf-8").read()
md = re.sub(r"(?m)^\s*---\s*$", "", md)  # thematic breaks render as black bars
repl = {"→": "->", "←": "<-", "≥": ">=", "≤": "<=", "×": "x", "—": "-", "–": "-",
        "’": "'", "‘": "'", "“": '"', "”": '"', "‖": "||", "Σ": "sum", "≠": "!=",
        "≈": "~=", "·": "*", "✓": "[x]", "✅": "[YES]", "❌": "[NO]"}
for k, v in repl.items():
    md = md.replace(k, v)
# Each `<!-- pagebreak -->` chunk becomes its OWN Story: PyMuPDF's Story leaks the
# header-row background of a table it *tried* to place on the previous page (ghost
# bands), so sections are laid out independently and simply appended.
chunks = [c for c in md.split("<!-- pagebreak -->") if c.strip()]
css = """
body{font-family:Helvetica,Arial,sans-serif;font-size:10pt;line-height:1.45;color:#1a1a1a;}
h1{font-size:19pt;color:#0b3d91;} h2{font-size:14pt;color:#0b3d91;margin-top:18px;}
h3{font-size:11.5pt;color:#222;margin-top:12px;} p{margin:6px 0;}
code{font-family:Courier,monospace;font-size:8.5pt;color:#8a1a1a;}
pre{background:#f5f5f7;border:1px solid #dddddd;padding:8px;}
pre code{font-family:Courier,monospace;font-size:7.5pt;color:#111111;line-height:1.25;}
table{border-collapse:collapse;width:100%;font-size:TBLpt;margin:8px 0;}
th,td{border:1px solid #bbbbbb;padding:3px 5px;text-align:left;vertical-align:top;}
th{background:#e8eef7;}
""".replace("TBL", "7.6" if landscape else "8.5")

writer = fitz.DocumentWriter(out)
mb = fitz.paper_rect("letter-l" if landscape else "letter")
area = mb + (48, 48, -48, -54) if landscape else mb + (54, 54, -54, -60)
for chunk in chunks:
    body = markdown.markdown(chunk, extensions=["tables", "fenced_code", "sane_lists"])
    story = fitz.Story(html=f"<html><head><style>{css}</style></head><body>{body}</body></html>")
    more = 1
    while more:
        dev = writer.begin_page(mb)
        more, _ = story.place(area)
        story.draw(dev)
        writer.end_page()
writer.close()
d = fitz.open(out)
d.set_metadata({"title": title, "author": author})
d.saveIncr()
print(f"WROTE {out} pages={d.page_count}")
