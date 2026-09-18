"""Preparing images, and knowing what they cost before sending them.

An image is billed as ceil(width/28) * ceil(height/28) vision tokens. That
formula is the reason this module exists rather than passing bytes straight
through: a phone photo at full resolution is several times the cost of the same
photo downscaled, and above a certain size it buys nothing, because the model's
own high-resolution tier caps out anyway.

Downscaling the long edge to ~1500 px lands around 2,200 vision tokens per
image and keeps the detail an inspection needs. It is a default, not a law —
domains that hinge on fine texture should raise it and pay for it knowingly.

EXIF transposition is not a nicety. Without it a portrait photo from a phone
arrives lying on its side, and the model dutifully evaluates a rotated scene and
reports nonsense with high confidence.
"""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from PIL import Image, ImageOps

DEFAULT_LONG_EDGE = 1500
JPEG_QUALITY = 85


@dataclass(frozen=True)
class PreparedImage:
    b64: str
    media_type: str
    width: int
    height: int
    vision_tokens: int
    origin: str


def vision_tokens(width: int, height: int) -> int:
    """ceil(width/28) * ceil(height/28)."""
    return math.ceil(width / 28) * math.ceil(height / 28)


def prepare(source: str | Path | BinaryIO, long_edge: int = DEFAULT_LONG_EDGE) -> PreparedImage:
    """Accepts a path or a file-like object.

    The file-like path is what lets an image living in object storage be
    evaluated without landing it in a temp file first.
    """
    name = str(source) if isinstance(source, (str, Path)) else getattr(source, "name", "<memory>")
    with Image.open(source) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        im.thumbnail((long_edge, long_edge), Image.LANCZOS)
        buffer = io.BytesIO()
        im.save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        width, height = im.size

    return PreparedImage(
        b64=base64.standard_b64encode(buffer.getvalue()).decode("ascii"),
        media_type="image/jpeg",
        width=width,
        height=height,
        vision_tokens=vision_tokens(width, height),
        origin=name,
    )


def image_block(image: PreparedImage) -> dict:
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": image.media_type, "data": image.b64},
    }
