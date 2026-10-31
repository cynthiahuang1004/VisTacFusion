"""Replace only the eight Real input panels in the existing vector Fig. 7 PDF.

Requires PyMuPDF. Run from any directory after updating assets/fig7_real/.
The other 40 images, page dimensions, labels, and panel positions are verified
against the original PDF before the new PDF and its PNG preview are saved.
This updates display thumbnails; it does not recompute model predictions.
"""
from collections import Counter
from pathlib import Path
import hashlib
import pymupdf


ROOT = Path(__file__).resolve().parents[1]
ROWS = ["1_V-shape", "2_Corner", "3_Y-junction", "4_Rod"]


def panels(page):
    images = page.get_image_info(xrefs=True)
    assert len(images) == 48, "Expected four rows of twelve image panels"
    groups = {}
    for image in images:
        groups.setdefault(round(image["bbox"][1], 1), []).append(image)
    assert len(groups) == 4
    ordered = []
    for y in sorted(groups):
        row = sorted(groups[y], key=lambda image: image["bbox"][0])
        assert len(row) == 12
        ordered.extend(row)
    assert all(n == 1 for n in Counter(i["xref"] for i in ordered).values())
    return ordered


def digest(doc, xref):
    pixmap = pymupdf.Pixmap(doc, xref)
    return (pixmap.width, pixmap.height, pixmap.n,
            hashlib.sha256(pixmap.samples).hexdigest())


def main():
    pdf = ROOT / "figures/qualitative_results.pdf"
    temporary = pdf.with_name("qualitative_results.updated.pdf")
    original = pymupdf.open(pdf)
    page = original[0]
    before = panels(page)
    original_text, original_rect = page.get_text(), tuple(page.rect)
    original_digests = [digest(original, image["xref"]) for image in before]
    replacements = {}
    for row, name in enumerate(ROWS):
        for column, modality in [(2, "tactile"), (3, "visual")]:
            index = row * 12 + column
            replacement = ROOT / "assets/fig7_real" / f"{name}_{modality}.jpg"
            assert replacement.is_file(), replacement
            page.replace_image(before[index]["xref"], filename=str(replacement))
            replacements[index] = replacement
    original.save(temporary, garbage=4, deflate=True)
    original.close()

    updated = pymupdf.open(temporary)
    after = panels(updated[0])
    assert tuple(updated[0].rect) == original_rect
    assert updated[0].get_text() == original_text
    for index, (old, new) in enumerate(zip(before, after)):
        assert all(abs(a - b) < .001 for a, b in zip(old["bbox"], new["bbox"]))
        if index in replacements:
            # Exact JPEG stream preservation: no display-image resampling or edits.
            image = updated.extract_image(new["xref"])
            assert image["image"] == replacements[index].read_bytes()
        else:
            assert digest(updated, new["xref"]) == original_digests[index]
    updated[0].get_pixmap(dpi=300, alpha=False).save(pdf.with_suffix(".png"))
    updated.close()
    temporary.replace(pdf)
    print("PASS: eight supplied JPEGs embedded exactly; other 40 panels, text, "
          "page size, and panel positions preserved.")


if __name__ == "__main__":
    main()
