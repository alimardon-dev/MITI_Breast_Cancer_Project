import os
import sys
import logging
import argparse
from pathlib import Path

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError
from PIL import Image

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Windowing helpers
# ---------------------------------------------------------------------------

def apply_windowing(pixel_array: np.ndarray, center: float, width: float) -> np.ndarray:
    """
    Apply DICOM windowing (contrast/brightness adjustment).

    Maps the input Hounsfield / raw pixel values to the [0, 255] range
    using the supplied Window Center and Window Width.

    Args:
        pixel_array: 2-D float array of raw pixel values.
        center:      Window Center (level) from DICOM tag (0028,1050).
        width:       Window Width from DICOM tag (0028,1051).

    Returns:
        uint8 array in [0, 255].
    """
    low  = center - width / 2.0
    high = center + width / 2.0

    windowed = np.clip(pixel_array, low, high)
    windowed = ((windowed - low) / (high - low) * 255.0)
    return windowed.astype(np.uint8)


def normalize(pixel_array: np.ndarray) -> np.ndarray:
    """
    Linear min–max normalisation to [0, 255] uint8.

    Used as a fallback when no windowing metadata is present.

    Args:
        pixel_array: Any numeric numpy array (1-D, 2-D, or 3-D).

    Returns:
        uint8 array with the same shape, values in [0, 255].
    """
    pmin, pmax = pixel_array.min(), pixel_array.max()
    if pmax == pmin:
        # Flat image — return zeros to avoid division by zero
        return np.zeros_like(pixel_array, dtype=np.uint8)
    normalized = (pixel_array - pmin) / (pmax - pmin) * 255.0
    return normalized.astype(np.uint8)


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def extract_pixel_array(ds: pydicom.Dataset) -> np.ndarray:
    """
    Extract and decompress the pixel array from a DICOM dataset.

    Applies the DICOM Rescale Slope / Intercept transformation when present
    so that the returned values are in the physical unit (e.g. HU for CT).

    Args:
        ds: A loaded pydicom Dataset.

    Returns:
        Float64 numpy array of shape (rows, cols) for single-frame images
        or (frames, rows, cols) for multi-frame images.

    Raises:
        AttributeError: If no pixel data exists in the dataset.
    """
    pixel_array = ds.pixel_array.astype(np.float64)

    # Apply Rescale Slope / Intercept (common in CT modality)
    slope     = float(getattr(ds, "RescaleSlope",     1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    pixel_array = pixel_array * slope + intercept

    return pixel_array


def frame_to_png(frame: np.ndarray, ds: pydicom.Dataset, use_windowing: bool) -> Image.Image:
    """
    Convert a single 2-D pixel frame to a Pillow Image.

    Windowing is applied when the DICOM metadata contains valid
    Window Center / Window Width tags and *use_windowing* is True.
    Otherwise, linear min–max normalisation is used.

    Args:
        frame:         2-D float64 numpy array (one frame).
        ds:            Source DICOM dataset (for windowing metadata).
        use_windowing: Whether to attempt windowing from metadata.

    Returns:
        A Pillow Image in mode 'L' (8-bit grayscale).
    """
    if use_windowing:
        wc_tag = getattr(ds, "WindowCenter", None)
        ww_tag = getattr(ds, "WindowWidth",  None)

        if wc_tag is not None and ww_tag is not None:
            # Tags may be multi-valued sequences — use the first value
            center = float(wc_tag[0] if hasattr(wc_tag, "__iter__") and not isinstance(wc_tag, str) else wc_tag)
            width  = float(ww_tag[0] if hasattr(ww_tag, "__iter__") and not isinstance(ww_tag, str) else ww_tag)
            log.debug("  Applying windowing: center=%.1f  width=%.1f", center, width)
            pixels_u8 = apply_windowing(frame, center, width)
        else:
            log.debug("  No windowing metadata — using min-max normalisation.")
            pixels_u8 = normalize(frame)
    else:
        pixels_u8 = normalize(frame)

    return Image.fromarray(pixels_u8, mode="L")


def convert_dicom_to_png(
    dicom_path: Path,
    output_dir: Path,
    use_windowing: bool = True,
) -> list[Path]:
    """
    Convert a single DICOM file to one or more PNG images.

    Single-frame DICOMs produce one PNG (``<stem>.png``).
    Multi-frame DICOMs produce one PNG per frame
    (``<stem>_frame_0001.png``, ``<stem>_frame_0002.png``, …).

    Args:
        dicom_path:   Path to the source ``.dcm`` file.
        output_dir:   Directory where PNG files will be saved.
        use_windowing: Apply DICOM windowing when metadata is available.

    Returns:
        List of Path objects for every PNG file that was written.

    Raises:
        InvalidDicomError: If the file is not a valid DICOM file.
        Exception:         For any other unexpected errors.
    """
    log.info("Processing: %s", dicom_path.name)

    try:
        ds = pydicom.dcmread(str(dicom_path))
    except InvalidDicomError as exc:
        raise InvalidDicomError(f"Not a valid DICOM file: {dicom_path}") from exc

    if not hasattr(ds, "PixelData"):
        raise ValueError(f"DICOM file has no pixel data: {dicom_path}")

    pixel_array = extract_pixel_array(ds)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # ── Multi-frame DICOM (shape: frames × rows × cols) ──────────────────
    if pixel_array.ndim == 3:
        n_frames = pixel_array.shape[0]
        log.info("  Multi-frame DICOM detected (%d frames).", n_frames)
        for i, frame in enumerate(pixel_array):
            img      = frame_to_png(frame, ds, use_windowing)
            out_path = output_dir / f"{dicom_path.stem}_frame_{i + 1:04d}.png"
            img.save(str(out_path))
            saved.append(out_path)
            log.info("  Saved frame %d/%d → %s", i + 1, n_frames, out_path.name)

    # ── Single-frame / RGB DICOM (shape: rows × cols  or  rows × cols × 3)
    elif pixel_array.ndim == 2:
        img      = frame_to_png(pixel_array, ds, use_windowing)
        out_path = output_dir / f"{dicom_path.stem}.png"
        img.save(str(out_path))
        saved.append(out_path)
        log.info("  Saved → %s", out_path.name)

    elif pixel_array.ndim == 3 and pixel_array.shape[2] == 3:
        # True-colour (RGB) DICOM — normalise each channel independently
        rgb_u8   = normalize(pixel_array)
        img      = Image.fromarray(rgb_u8, mode="RGB")
        out_path = output_dir / f"{dicom_path.stem}.png"
        img.save(str(out_path))
        saved.append(out_path)
        log.info("  Saved RGB image → %s", out_path.name)

    else:
        raise ValueError(
            f"Unexpected pixel array shape {pixel_array.shape} in {dicom_path}"
        )

    return saved


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def batch_convert(
    input_path: Path,
    output_dir: Path,
    use_windowing: bool = True,
    recursive: bool = False,
) -> None:
    """
    Convert all DICOM files found under *input_path* to PNG.

    If *input_path* is a file, only that file is processed.
    If it is a directory, every ``.dcm`` file inside is processed.

    Args:
        input_path:    File or directory to process.
        output_dir:    Root output directory for PNG files.
        use_windowing: Forward to :func:`convert_dicom_to_png`.
        recursive:     If True, search subdirectories recursively.
    """
    if input_path.is_file():
        dicom_files = [input_path]
    elif input_path.is_dir():
        pattern     = "**/*.dcm" if recursive else "*.dcm"
        dicom_files = sorted(input_path.glob(pattern))
        if not dicom_files:
            log.warning("No .dcm files found in %s", input_path)
            return
    else:
        log.error("Input path does not exist: %s", input_path)
        sys.exit(1)

    total   = len(dicom_files)
    success = 0
    failed  = 0

    log.info("=" * 60)
    log.info("Found %d DICOM file(s) to convert.", total)
    log.info("Output directory: %s", output_dir)
    log.info("=" * 60)

    for idx, dcm_path in enumerate(dicom_files, start=1):
        log.info("[%d/%d] %s", idx, total, dcm_path.name)
        try:
            # Mirror the source directory structure inside output_dir
            relative = dcm_path.parent.relative_to(input_path) if input_path.is_dir() else Path(".")
            dest_dir = output_dir / relative
            convert_dicom_to_png(dcm_path, dest_dir, use_windowing)
            success += 1
        except (InvalidDicomError, ValueError) as exc:
            log.error("  SKIPPED — %s", exc)
            failed += 1
        except Exception as exc:  # noqa: BLE001
            log.error("  ERROR — Unexpected error for %s: %s", dcm_path.name, exc)
            failed += 1

    log.info("=" * 60)
    log.info("Done.  Success: %d  |  Failed / Skipped: %d", success, failed)
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert DICOM (.dcm) files to PNG format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Convert a single file
  python dicom_to_png.py scan.dcm -o ./png_output

  # Convert all DICOMs in a folder
  python dicom_to_png.py ./dicoms -o ./png_output

  # Recursive search, disable windowing
  python dicom_to_png.py ./dicoms -o ./png_output -r --no-windowing
""",
    )
    parser.add_argument(
        "input",
        help="Path to a DICOM file or a directory containing DICOM files.",
    )
    parser.add_argument(
        "-o", "--output",
        default="./png_output",
        help="Output directory for PNG files (default: ./png_output).",
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Recursively search subdirectories for .dcm files.",
    )
    parser.add_argument(
        "--no-windowing",
        action="store_true",
        help="Disable DICOM windowing; use min-max normalisation instead.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    batch_convert(
        input_path    = Path(args.input),
        output_dir    = Path(args.output),
        use_windowing = not args.no_windowing,
        recursive     = args.recursive,
    )


if __name__ == "__main__":
    main()
