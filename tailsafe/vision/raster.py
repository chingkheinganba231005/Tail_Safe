"""Load floor-plan images (PNG, JPEG, or the first page of a PDF) as grey arrays."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

PDF_DPI = 150


def _pdf_first_page(data: bytes) -> NDArray[np.float32]:
    try:
        import pypdfium2 as pdfium  # optional dependency: pip install "tailsafe[vision]"
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ValueError(
            "reading PDF plans needs the optional pypdfium2 package "
            '(pip install "tailsafe[vision]"); export the page as PNG instead'
        ) from exc
    pdf = pdfium.PdfDocument(data)
    page = pdf[0]
    img = page.render(scale=PDF_DPI / 72).to_pil().convert("L")
    return np.asarray(img, dtype=np.float32) / 255.0


def load_image(source: bytes | str | Path) -> NDArray[np.float32]:
    """Grey image in [0, 1] (1 = paper) from bytes, a data URL or a file path."""
    from PIL import Image

    if isinstance(source, Path) or (isinstance(source, str) and not source.startswith("data:")):
        data = Path(source).read_bytes()
    elif isinstance(source, str):
        data = base64.b64decode(source.split(",", 1)[1])
    else:
        data = source
    if data[:5] == b"%PDF-":
        return _pdf_first_page(data)
    try:
        opened = Image.open(io.BytesIO(data))
        opened.load()
    except Exception as exc:
        raise ValueError(f"not a readable image: {exc}") from exc
    img: Image.Image = opened
    if img.mode in ("RGBA", "LA", "P"):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, rgba)
    return np.asarray(img.convert("L"), dtype=np.float32) / 255.0


def to_data_url(png: bytes) -> str:
    """``data:image/png;base64,...`` for the browser."""
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")
