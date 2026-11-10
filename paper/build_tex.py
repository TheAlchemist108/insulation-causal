"""Convert the paper's master text (Claude Doc exported as markdown, saved to build/doc_export.md)
into LaTeX for arXiv. Charts in the doc are replaced by the matplotlib figures in figures/,
which are drawn from the same output files. Usage: python build_tex.py"""
import re, subprocess
from pathlib import Path

P = Path(__file__).resolve().parent
md = (P / "build" / "doc_export.md").read_text()

# --- title, byline, abstract, keywords
title = re.search(r"^# (.+)$", md, re.M).group(1).strip()
md = md[md.index("\n", md.index("# " + title)) + 1:]
md = re.sub(r"^@Kaustubh.*$", "", md, count=1, flags=re.M)
abs_m = re.search(r"^## Abstract\n(.*?)(?=^## )", md, re.S | re.M)
abstract = abs_m.group(1).strip() if abs_m else "ABSTRACT MISSING"
if abs_m:
    md = md.replace(abs_m.group(0), "")
kw = re.search(r"^\*\*Keywords:\*\*(.+)$", abstract, re.M)
keywords = kw.group(1).strip() if kw else ""
abstract = re.sub(r"^\*\*Keywords:\*\*.+$", "", abstract, flags=re.M).strip()

# --- figures: doc chart placeholders -> LaTeX figures
FIGS = {"Figure 1.": ("fig1_event_study.png", "fig:event"), "Figure 2.": ("fig2_subgroups.png", "fig:subgroups")}
def fig_repl(m):
    cap = m.group(1).strip().rstrip("\\").strip()
    for key, (fn, lab) in FIGS.items():
        if cap.startswith(key):
            body = cap[len(key):].strip()
            return f"\n\n```{{=latex}}\n\\begin{{figure}}[H]\\centering\\includegraphics[width=\\linewidth]{{figures/{fn}}}\\caption{{{latex_escape(body)}}}\\label{{{lab}}}\\end{{figure}}\n```\n\n"
    raise ValueError(cap)
def latex_escape(s):
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_")]:
        s = s.replace(a, b)
    return s
md = re.sub(r"(?:&#91;|\[)embedded content: (.+?)\]", fig_repl, md)

# --- symbols pdflatex cannot take directly
md = md.replace("M\u0304", "$\\bar{M}$")

# --- table captions in bold; pipe-table column widths proportional to content
md = re.sub(r"^(Table [A-Z]?\d+\.)", r"**\1**", md, flags=re.M)
def widths(block):
    lines = block.split("\n")
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines if l.strip().startswith("|")]
    hdr, body = rows[0], rows[2:]
    n = len(hdr)
    mx = [max(len(r[i]) if i < len(r) else 0 for r in [hdr] + body) for i in range(n)]
    mx = [max(12, min(m, 60)) for m in mx]
    sep = "| " + " | ".join("-" * m for m in mx) + " |"
    out = [lines[0], sep] + lines[2:]
    return "\n".join(out)
md = re.sub(r"(?:^\|.*\|[ \t]*\n)+", lambda m: widths(m.group(0).rstrip("\n")) + "\n", md, flags=re.M)

# --- display maths: ```latex blocks -> $$ $$
md = re.sub(r"```latex\n(.*?)\n```", lambda m: "$$\n" + m.group(1).strip() + "\n$$", md, flags=re.S)

# --- section numbering is in the text already; drop pandoc's own numbers
(P / "build" / "body.md").write_text(md)
subprocess.run(["pandoc", str(P / "build" / "body.md"), "-f", "markdown+pipe_tables+tex_math_dollars+raw_attribute-auto_identifiers",
                "-t", "latex", "--wrap=preserve", "--shift-heading-level-by=-1", "-o", str(P / "build" / "body.tex")], check=True)
body = (P / "build" / "body.tex").read_text()
body = body.replace("\\section{", "\\section*{").replace("\\subsection{", "\\subsection*{").replace("\\subsubsection{", "\\subsubsection*{")
i = body.find("\\section*{References}")
if i >= 0:
    j = body.find("\\begin{itemize}", i); k = body.find("\\end{itemize}", j)
    body = (body[:j] + "\\begin{list}{}{\\setlength{\\leftmargin}{1.5em}\\setlength{\\itemindent}{-1.5em}\\setlength{\\itemsep}{2pt}}" +
            body[j + len("\\begin{itemize}"):k] + "\\end{list}" + body[k + len("\\end{itemize}"):])
# headings already carry their numbers; add them to the TOC-less layout and keep anchors
tpl = (P / "template.tex").read_text()
out = (tpl.replace("@@TITLE@@", latex_escape(title)).replace("@@ABSTRACT@@", latex_escape(abstract))
          .replace("@@KEYWORDS@@", latex_escape(keywords)).replace("@@BODY@@", body))
(P / "main.tex").write_text(out)
print("main.tex written:", len(out), "chars")
