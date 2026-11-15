"""Shared data models for medical image and report analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


SUPPORTED_MODALITIES = ("X-ray", "CT scan", "MRI", "Retina")


@dataclass(frozen=True)
class Finding:
    """Single model finding with explainability metadata."""

    label: str
    confidence: float
    severity: str
    evidence: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ImageAnalysis:
    """Result of analyzing one medical image."""

    modality: str
    image_path: Path
    heatmap_path: Path
    top_finding: Finding
    findings: list[Finding]
    technical_notes: list[str]


@dataclass(frozen=True)
class ReportAnalysis:
    """Structured information extracted from a clinical report."""

    concerns: list[str]
    urgent_terms: list[str]
    reassuring_terms: list[str]
    summary: str

