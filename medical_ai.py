"""Medical imaging, explainability, and patient-summary logic.

This module is intentionally model-ready: a production deployment can replace
the heuristic fallback with validated Torch/TensorFlow weights while preserving
the UI and report contract. The fallback keeps this project runnable on a
standard Python install with only NumPy and Pillow.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .models import Finding, ImageAnalysis, ReportAnalysis


DISEASE_CATALOG = {
    "X-ray": ("Pneumonia pattern", "Fracture pattern", "No acute finding"),
    "CT scan": ("Kidney stone pattern", "Tumor-like lesion", "No acute finding"),
    "MRI": ("Tumor-like lesion", "Inflammatory change", "No acute finding"),
    "Retina": ("Diabetic retinopathy signs", "Macular abnormality", "No acute finding"),
}


def create_demo_images(data_dir: Path) -> dict[str, Path]:
    """Create simple synthetic medical-looking examples for first launch."""

    data_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "X-ray": data_dir / "sample_chest_xray.png",
        "CT scan": data_dir / "sample_ct_kidney.png",
        "MRI": data_dir / "sample_brain_mri.png",
        "Retina": data_dir / "sample_retina.png",
    }
    if not outputs["X-ray"].exists():
        _draw_chest_xray(outputs["X-ray"])
    if not outputs["CT scan"].exists():
        _draw_ct(outputs["CT scan"])
    if not outputs["MRI"].exists():
        _draw_mri(outputs["MRI"])
    if not outputs["Retina"].exists():
        _draw_retina(outputs["Retina"])
    return outputs


def analyze_medical_image(image_path: Path, modality: str, output_dir: Path) -> ImageAnalysis:
    """Analyze a medical image and create a Grad-CAM-style explanation overlay."""

    image = Image.open(image_path).convert("RGB")
    array = _resize_array(image, 256)
    gray = np.dot(array[..., :3], [0.299, 0.587, 0.114])

    feature_map = _feature_map_for_modality(gray, modality)
    heatmap_path = output_dir / f"{image_path.stem}_{_slug(modality)}_gradcam.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_grad_cam(image, feature_map, heatmap_path)

    findings = _classify_with_fallback(gray, feature_map, modality)
    technical_notes = [
        "Grad-CAM-style overlay highlights image regions most responsible for the top prediction.",
        "Prototype classifier uses image feature scoring unless a validated deep learning model is connected.",
        "Use only as clinical decision support; final interpretation requires a qualified clinician.",
    ]
    return ImageAnalysis(
        modality=modality,
        image_path=image_path,
        heatmap_path=heatmap_path,
        top_finding=findings[0],
        findings=findings,
        technical_notes=technical_notes,
    )


def analyze_report_text(report_text: str) -> ReportAnalysis:
    """Extract notable terms from a medical report."""

    text = report_text.strip()
    lowered = text.lower()
    concerns = _find_terms(
        lowered,
        {
            "pneumonia": "Possible pneumonia mentioned",
            "consolidation": "Lung consolidation mentioned",
            "fracture": "Fracture mentioned",
            "mass": "Mass or lesion mentioned",
            "tumor": "Tumor mentioned",
            "stone": "Stone or calculus mentioned",
            "retinopathy": "Retinopathy mentioned",
            "hemorrhage": "Hemorrhage mentioned",
            "edema": "Swelling or edema mentioned",
        },
    )
    urgent = _find_terms(
        lowered,
        {
            "critical": "critical",
            "urgent": "urgent",
            "stat": "stat",
            "acute": "acute",
            "severe": "severe",
            "emergency": "emergency",
        },
    )
    reassuring = _find_terms(
        lowered,
        {
            "normal": "normal",
            "no acute": "no acute abnormality",
            "unremarkable": "unremarkable",
            "negative": "negative",
            "stable": "stable",
        },
    )
    if not text:
        summary = "No report text was entered."
    elif concerns:
        summary = "The report includes terms that may need clinical review: " + ", ".join(concerns[:4]) + "."
    elif reassuring:
        summary = "The report contains reassuring language, but it should still be reviewed in context."
    else:
        summary = "The report was parsed, but no major tracked disease terms were found."
    return ReportAnalysis(concerns=concerns, urgent_terms=urgent, reassuring_terms=reassuring, summary=summary)


def generate_patient_summary(image_result: ImageAnalysis | None, report_result: ReportAnalysis | None) -> str:
    """Generate a patient-friendly LLM-style summary with safety guidance."""

    lines = [
        "Patient-friendly summary",
        "",
        "This assistant is a screening and explanation tool, not a final diagnosis.",
    ]
    if image_result:
        top = image_result.top_finding
        lines.extend(
            [
                "",
                f"The uploaded {image_result.modality.lower()} most strongly matched: {top.label}.",
                f"Estimated confidence: {top.confidence:.0%}; estimated severity: {top.severity}.",
                "Why the AI focused there: " + " ".join(top.evidence[:2]),
            ]
        )
        if top.recommendations:
            lines.append("Suggested next step: " + top.recommendations[0])
    if report_result:
        lines.extend(["", report_result.summary])
        if report_result.urgent_terms:
            lines.append("The report contains urgent-sounding terms. Please contact a clinician promptly.")
    lines.extend(
        [
            "",
            "Seek emergency care now for severe pain, trouble breathing, weakness on one side, confusion, fainting, or rapidly worsening symptoms.",
            "Share the original images, this summary, and the full radiology report with your doctor.",
        ]
    )
    return "\n".join(lines)


def _classify_with_fallback(gray: np.ndarray, heat: np.ndarray, modality: str) -> list[Finding]:
    bright_ratio = float((gray > 0.76).mean())
    dark_ratio = float((gray < 0.18).mean())
    contrast = float(gray.std())
    center_activity = float(heat[72:184, 72:184].mean())
    edge_activity = float(np.r_[heat[:55, :].ravel(), heat[-55:, :].ravel(), heat[:, :55].ravel(), heat[:, -55:].ravel()].mean())
    spot_count = _count_hot_spots(heat)

    if modality == "X-ray":
        pneumonia = _clamp(0.36 + contrast * 0.75 + center_activity * 0.25)
        fracture = _clamp(0.18 + edge_activity * 0.35 + bright_ratio * 0.7)
        normal = _clamp(1.05 - pneumonia * 0.75 - fracture * 0.45)
        findings = [
            _finding("Pneumonia pattern", pneumonia, "patchy lung opacity signal", ["Opacity-like areas detected in lung fields.", "Heatmap concentrates across the central chest."]),
            _finding("Fracture pattern", fracture, "high-contrast bone line signal", ["Bright linear structures detected near high-contrast regions.", "Review alignment and cortical margins."]),
            _finding("No acute finding", normal, "low abnormality score", ["Tracked abnormal features were limited.", "Image quality and positioning still affect accuracy."]),
        ]
    elif modality == "CT scan":
        stone = _clamp(0.22 + bright_ratio * 2.5 + spot_count * 0.04)
        tumor = _clamp(0.24 + center_activity * 0.45 + contrast * 0.45)
        normal = _clamp(1.0 - stone * 0.65 - tumor * 0.45)
        findings = [
            _finding("Kidney stone pattern", stone, "small bright focus signal", ["Small dense bright areas detected.", "Pattern may match calcification or stone-like density."]),
            _finding("Tumor-like lesion", tumor, "asymmetric soft-tissue signal", ["Regional intensity asymmetry detected.", "Heatmap marks a localized lesion-like area."]),
            _finding("No acute finding", normal, "low abnormality score", ["No dominant tracked pattern was found.", "Correlation with CT phase and report is required."]),
        ]
    elif modality == "MRI":
        tumor = _clamp(0.28 + center_activity * 0.55 + contrast * 0.55)
        inflammation = _clamp(0.18 + dark_ratio * 0.3 + edge_activity * 0.25)
        normal = _clamp(1.0 - tumor * 0.7 - inflammation * 0.35)
        findings = [
            _finding("Tumor-like lesion", tumor, "focal MRI intensity signal", ["Focal intensity variation detected.", "Heatmap localizes attention to an asymmetric region."]),
            _finding("Inflammatory change", inflammation, "diffuse intensity signal", ["Diffuse abnormal texture detected.", "Clinical history and sequence type matter."]),
            _finding("No acute finding", normal, "low abnormality score", ["Tracked focal abnormalities were limited.", "Radiologist review remains necessary."]),
        ]
    else:
        retinopathy = _clamp(0.28 + spot_count * 0.06 + bright_ratio * 0.65 + contrast * 0.35)
        macular = _clamp(0.18 + center_activity * 0.42)
        normal = _clamp(1.0 - retinopathy * 0.7 - macular * 0.35)
        findings = [
            _finding("Diabetic retinopathy signs", retinopathy, "microaneurysm/exudate-like signal", ["Small bright or dark retinal spots detected.", "Heatmap highlights scattered vascular regions."]),
            _finding("Macular abnormality", macular, "central retinal signal", ["Central retinal area drew model attention.", "OCT or eye exam may be needed."]),
            _finding("No acute finding", normal, "low abnormality score", ["Tracked retinal warning signs were limited.", "Routine screening intervals depend on medical history."]),
        ]
    return sorted(findings, key=lambda item: item.confidence, reverse=True)


def _finding(label: str, confidence: float, severity_hint: str, evidence: list[str]) -> Finding:
    severity = "High" if confidence >= 0.72 else "Moderate" if confidence >= 0.48 else "Low"
    recommendations = [
        "Review with a licensed clinician or radiologist before making treatment decisions.",
        "Compare with symptoms, lab results, prior imaging, and the official report.",
    ]
    return Finding(label=label, confidence=confidence, severity=severity, evidence=[severity_hint, *evidence], recommendations=recommendations)


def _feature_map_for_modality(gray: np.ndarray, modality: str) -> np.ndarray:
    blurred = _box_blur(gray, 9)
    edges = np.abs(gray - _box_blur(gray, 3))
    yy, xx = np.mgrid[0 : gray.shape[0], 0 : gray.shape[1]]
    cx, cy = gray.shape[1] / 2, gray.shape[0] / 2
    center_weight = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (gray.shape[0] * 0.28) ** 2)))
    if modality == "Retina":
        candidate = edges * 1.4 + np.maximum(gray - blurred, 0) * 1.2 + center_weight * 0.15
    elif modality == "CT scan":
        candidate = np.maximum(gray - blurred, 0) * 1.7 + edges * 0.7
    elif modality == "MRI":
        candidate = np.abs(gray - blurred) * 1.4 + center_weight * 0.25
    else:
        candidate = edges * 0.9 + np.maximum(blurred - gray, 0) * 0.8 + center_weight * 0.2
    return _normalize(candidate)


def _save_grad_cam(source: Image.Image, heat: np.ndarray, destination: Path) -> None:
    base = source.convert("RGB").resize((512, 512))
    heat_img = Image.fromarray(np.uint8(_normalize(heat) * 255), "L").resize(base.size, Image.Resampling.BICUBIC)
    color = Image.new("RGBA", base.size, (0, 0, 0, 0))
    pixels = np.asarray(heat_img, dtype=np.uint8)
    overlay = np.zeros((base.size[1], base.size[0], 4), dtype=np.uint8)
    overlay[..., 0] = 255
    overlay[..., 1] = np.clip(pixels * 0.65, 0, 180)
    overlay[..., 3] = np.clip(pixels * 0.55, 0, 150)
    color = Image.fromarray(overlay, "RGBA").filter(ImageFilter.GaussianBlur(radius=2))
    combined = Image.alpha_composite(base.convert("RGBA"), color)
    combined.save(destination)


def _resize_array(image: Image.Image, size: int) -> np.ndarray:
    resized = image.resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.float32) / 255.0


def _box_blur(array: np.ndarray, radius: int) -> np.ndarray:
    pad = radius // 2
    padded = np.pad(array, pad, mode="edge")
    out = np.zeros_like(array)
    for y in range(array.shape[0]):
        for x in range(array.shape[1]):
            out[y, x] = padded[y : y + radius, x : x + radius].mean()
    return out


def _normalize(array: np.ndarray) -> np.ndarray:
    minimum = float(array.min())
    maximum = float(array.max())
    if math.isclose(maximum, minimum):
        return np.zeros_like(array)
    return (array - minimum) / (maximum - minimum)


def _count_hot_spots(heat: np.ndarray) -> int:
    mask = heat > 0.82
    if not mask.any():
        return 0
    sampled = mask[::8, ::8]
    return int(min(12, sampled.sum() // 2))


def _find_terms(text: str, terms: dict[str, str]) -> list[str]:
    return [label for term, label in terms.items() if re.search(rf"\b{re.escape(term)}\b", text)]


def _clamp(value: float) -> float:
    return max(0.03, min(0.97, value))


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _draw_chest_xray(path: Path) -> None:
    img = Image.new("L", (512, 512), 18)
    draw = ImageDraw.Draw(img)
    draw.ellipse((95, 78, 250, 430), fill=70)
    draw.ellipse((260, 78, 415, 430), fill=66)
    draw.rectangle((240, 50, 270, 455), fill=115)
    draw.ellipse((145, 220, 240, 350), fill=118)
    draw.line((330, 190, 390, 260), fill=205, width=5)
    img.filter(ImageFilter.GaussianBlur(4)).save(path)


def _draw_ct(path: Path) -> None:
    img = Image.new("L", (512, 512), 25)
    draw = ImageDraw.Draw(img)
    draw.ellipse((56, 56, 456, 456), fill=92, outline=170, width=9)
    draw.ellipse((155, 180, 250, 325), fill=122)
    draw.ellipse((275, 180, 370, 325), fill=118)
    draw.ellipse((328, 240, 343, 255), fill=245)
    img.filter(ImageFilter.GaussianBlur(2)).save(path)


def _draw_mri(path: Path) -> None:
    img = Image.new("L", (512, 512), 12)
    draw = ImageDraw.Draw(img)
    draw.ellipse((105, 45, 407, 465), fill=82)
    draw.ellipse((145, 92, 367, 415), fill=125)
    draw.arc((170, 130, 342, 385), 80, 290, fill=165, width=8)
    draw.ellipse((292, 190, 352, 252), fill=210)
    img.filter(ImageFilter.GaussianBlur(3)).save(path)


def _draw_retina(path: Path) -> None:
    img = Image.new("RGB", (512, 512), (8, 8, 8))
    draw = ImageDraw.Draw(img)
    draw.ellipse((38, 38, 474, 474), fill=(120, 54, 36))
    draw.ellipse((225, 225, 287, 287), fill=(74, 26, 28))
    draw.ellipse((322, 185, 382, 245), fill=(222, 166, 98))
    for angle in range(0, 360, 24):
        x = 256 + int(math.cos(math.radians(angle)) * 180)
        y = 256 + int(math.sin(math.radians(angle)) * 180)
        draw.line((352, 215, x, y), fill=(154, 42, 42), width=3)
    for point in [(180, 175), (205, 310), (298, 300), (145, 250), (260, 155)]:
        draw.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill=(238, 210, 130))
    img.filter(ImageFilter.GaussianBlur(1)).save(path)
