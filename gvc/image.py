"""Image preparation: EXIF orientation, downscaling, JPEG encoding, and a
vision-token estimate of ceil(width/28) * ceil(height/28).

The long edge is downscaled to DEFAULT_LONG_EDGE unless `long_edge` says
otherwise. EXIF orientation is applied first, so a portrait phone photo is not
sent rotated.
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
    """Accepts a path or a binary file-like object, such as a stream read from
    object storage."""
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
