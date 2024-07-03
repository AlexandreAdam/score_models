import sys
import nbformat
import re


def strip_output(nb):
    for cell in nb.cells:
        if cell.cell_type == "code":
            # Preserve HTML comments in code cells
            preserved_html = re.findall(r"<!--.*?-->", cell.source, re.DOTALL)
            cell.outputs = []
            cell.execution_count = None
            # Reinsert preserved HTML comments
            for html in preserved_html:
                cell.source += f"\n{html}\n"
        elif cell.cell_type == "markdown":
            pass  # Preserving HTML in markdown cells


if __name__ == "__main__":
    for filename in sys.argv[1:]:
        with open(filename, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=nbformat.NO_CONVERT)

        strip_output(nb)

        with open(filename, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
