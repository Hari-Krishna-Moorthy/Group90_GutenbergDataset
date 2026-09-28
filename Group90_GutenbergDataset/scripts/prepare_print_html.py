"""Add print-only CSS so notebook code and wide tables fit cleanly in PDF."""

from pathlib import Path
import sys


PRINT_STYLE = """
<!-- group90-print-style -->
<style>
@media print {
  @page {
    size: A4 landscape;
    margin: 10mm;
  }

  body {
    font-size: 9pt !important;
  }

  /* Wrap long source-code lines instead of clipping them at the page edge. */
  pre, code, .highlight pre, .jp-CodeCell pre {
    white-space: pre-wrap !important;
    overflow-wrap: anywhere !important;
    word-break: break-word !important;
    font-size: 7.2pt !important;
    line-height: 1.25 !important;
    overflow: visible !important;
  }

  /* Fit every DataFrame column on the printed page. */
  table.dataframe {
    width: 100% !important;
    max-width: 100% !important;
    table-layout: fixed !important;
    font-size: 6.8pt !important;
  }

  table.dataframe th,
  table.dataframe td {
    white-space: normal !important;
    overflow-wrap: anywhere !important;
    word-break: break-word !important;
    padding: 2px 3px !important;
  }

  /* Keep figures inside the printable area. */
  img, svg, .jp-OutputArea-output {
    max-width: 100% !important;
  }
}
</style>
"""


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: prepare_print_html.py PATH_TO_HTML")

    html_path = Path(sys.argv[1])
    html = html_path.read_text(encoding="utf-8")
    if "group90-print-style" not in html:
        html = html.replace("</head>", PRINT_STYLE + "\n</head>", 1)
        html_path.write_text(html, encoding="utf-8")
    print(f"Prepared print layout: {html_path}")


if __name__ == "__main__":
    main()
