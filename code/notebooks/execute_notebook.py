"""Execute Phishing_Research_End_to_End.ipynb top to bottom in a new kernel and export the executed copy to HTML."""
from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

HERE = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export-only", action="store_true", help="export the executed notebook to HTML without running it")
    a = p.parse_args()
    out = HERE / "Phishing_Research_End_to_End_EXECUTED.ipynb"
    if a.export_only:
        nb = nbformat.read(out, as_version=4)
    else:
        if sys.platform == "win32":  # the kernel messaging library (zmq) expects the selector event loop on Windows
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        nb = nbformat.read(HERE / "Phishing_Research_End_to_End.ipynb", as_version=4)
        client = NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": str(HERE)}})
        client.execute()
        nbformat.write(nb, out)
    body, _ = HTMLExporter(template_name="lab").from_notebook_node(nb)
    # comments of the export template's style sheets are developer notes of that template, not part of the report
    body = re.sub(r"<style[^>]*>.*?</style>", lambda m: re.sub(r"/\*.*?\*/", "", m.group(0), flags=re.S), body,
                  flags=re.S)
    (HERE / "Phishing_Research_End_to_End_EXECUTED.html").write_text(body, encoding="utf-8")
    executed = sum(1 for c in nb.cells if c.cell_type == "code" and c.get("execution_count"))
    errors = sum(1 for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "error")
    print(f"executed code cells: {executed}; error outputs: {errors}")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
