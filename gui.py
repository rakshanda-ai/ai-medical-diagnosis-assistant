"""Tkinter interface for the AI Medical Diagnosis Assistant."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from .medical_ai import analyze_medical_image, analyze_report_text, create_demo_images, generate_patient_summary
from .models import ImageAnalysis, ReportAnalysis, SUPPORTED_MODALITIES


class MedicalDiagnosisApp(tk.Tk):
    """Desktop prototype for imaging analysis, Grad-CAM, and report summaries."""

    def __init__(self) -> None:
        super().__init__()
        self.title("AI Medical Diagnosis Assistant")
        self.geometry("1320x820")
        self.minsize(1120, 720)
        self.configure(bg="#101418")

        self.base_dir = Path(__file__).resolve().parent
        self.output_dir = self.base_dir / "outputs"
        self.image_path: Path | None = None
        self.image_result: ImageAnalysis | None = None
        self.report_result: ReportAnalysis | None = None
        self._image_photo: ImageTk.PhotoImage | None = None
        self._heatmap_photo: ImageTk.PhotoImage | None = None

        self.modality = tk.StringVar(value="X-ray")

        self._build_styles()
        self._build_layout()
        self._load_demo()

    def _build_styles(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Root.TFrame", background="#101418")
        style.configure("Panel.TFrame", background="#182026", relief="flat")
        style.configure("Title.TLabel", background="#101418", foreground="#f5f7fa", font=("Segoe UI", 24, "bold"))
        style.configure("Subtle.TLabel", background="#101418", foreground="#9fb0bd", font=("Segoe UI", 10))
        style.configure("PanelTitle.TLabel", background="#182026", foreground="#f5f7fa", font=("Segoe UI", 14, "bold"))
        style.configure("Body.TLabel", background="#182026", foreground="#dce5eb", font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background="#182026", foreground="#9fb0bd", font=("Segoe UI", 9))
        style.configure("Metric.TLabel", background="#182026", foreground="#ffffff", font=("Segoe UI", 20, "bold"))
        style.configure("Warn.TLabel", background="#182026", foreground="#ffd166", font=("Segoe UI", 9, "bold"))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=(12, 8))
        style.configure("TCombobox", padding=6)

    def _build_layout(self) -> None:
        header = ttk.Frame(self, style="Root.TFrame", padding=(22, 16, 22, 8))
        header.pack(fill="x")
        ttk.Label(header, text="AI Medical Diagnosis Assistant", style="Title.TLabel").pack(side="left")
        ttk.Label(header, text="Imaging triage + Grad-CAM + report summary", style="Subtle.TLabel").pack(side="right")

        main = ttk.Frame(self, style="Root.TFrame", padding=(22, 10, 22, 22))
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=1, minsize=285)
        main.columnconfigure(1, weight=2, minsize=430)
        main.columnconfigure(2, weight=2, minsize=430)
        main.rowconfigure(0, weight=1)

        self._build_controls(main)
        self._build_images(main)
        self._build_results(main)

    def _build_controls(self, parent: ttk.Frame) -> None:
        panel = ttk.Frame(parent, style="Panel.TFrame", padding=18)
        panel.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        panel.columnconfigure(0, weight=1)
        ttk.Label(panel, text="Case Inputs", style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 12))

        ttk.Label(panel, text="Imaging type", style="Muted.TLabel").grid(row=1, column=0, sticky="w")
        ttk.Combobox(panel, textvariable=self.modality, values=SUPPORTED_MODALITIES, state="readonly").grid(
            row=2, column=0, sticky="ew", pady=(4, 12)
        )
        ttk.Button(panel, text="Upload Medical Image", style="Accent.TButton", command=self._choose_image).grid(row=3, column=0, sticky="ew")
        ttk.Button(panel, text="Analyze Case", style="Accent.TButton", command=self._run_analysis).grid(row=4, column=0, sticky="ew", pady=(10, 16))

        ttk.Label(panel, text="Medical report text", style="Muted.TLabel").grid(row=5, column=0, sticky="w")
        self.report_text = tk.Text(
            panel,
            wrap="word",
            height=12,
            bg="#0d1115",
            fg="#f5f7fa",
            insertbackground="#f5f7fa",
            relief="flat",
            padx=12,
            pady=10,
            font=("Segoe UI", 10),
        )
        self.report_text.grid(row=6, column=0, sticky="nsew", pady=(5, 12))
        panel.rowconfigure(6, weight=1)
        ttk.Label(
            panel,
            text="For education and triage support only. Do not use as a replacement for radiology or medical care.",
            style="Warn.TLabel",
            wraplength=245,
        ).grid(row=7, column=0, sticky="ew")

    def _build_images(self, parent: ttk.Frame) -> None:
        panel = ttk.Frame(parent, style="Panel.TFrame", padding=18)
        panel.grid(row=0, column=1, sticky="nsew", padx=(0, 14))
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(2, weight=1)
        panel.rowconfigure(4, weight=1)
        ttk.Label(panel, text="Image + Grad-CAM", style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(panel, text="Original image", style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=(12, 5))
        self.image_preview = tk.Label(panel, bg="#0d1115", fg="#dce5eb", text="Medical image preview", bd=0)
        self.image_preview.grid(row=2, column=0, sticky="nsew")
        ttk.Label(panel, text="Explainability overlay", style="Muted.TLabel").grid(row=3, column=0, sticky="w", pady=(14, 5))
        self.heatmap_preview = tk.Label(panel, bg="#0d1115", fg="#dce5eb", text="Grad-CAM preview", bd=0)
        self.heatmap_preview.grid(row=4, column=0, sticky="nsew")

    def _build_results(self, parent: ttk.Frame) -> None:
        panel = ttk.Frame(parent, style="Panel.TFrame", padding=18)
        panel.grid(row=0, column=2, sticky="nsew")
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(4, weight=1)
        ttk.Label(panel, text="AI Findings", style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w")
        self.top_label = ttk.Label(panel, text="Ready", style="Metric.TLabel")
        self.top_label.grid(row=1, column=0, sticky="w", pady=(12, 0))
        self.confidence_label = ttk.Label(panel, text="", style="Body.TLabel", wraplength=380)
        self.confidence_label.grid(row=2, column=0, sticky="ew", pady=(2, 12))
        ttk.Label(panel, text="Decision-support report", style="Muted.TLabel").grid(row=3, column=0, sticky="w")
        self.results = tk.Text(
            panel,
            wrap="word",
            bg="#0d1115",
            fg="#f5f7fa",
            insertbackground="#f5f7fa",
            relief="flat",
            padx=14,
            pady=12,
            font=("Segoe UI", 10),
        )
        self.results.grid(row=4, column=0, sticky="nsew")
        self.results.configure(state="disabled")

    def _load_demo(self) -> None:
        demos = create_demo_images(self.base_dir / "sample_data")
        self.image_path = demos[self.modality.get()]
        self.report_text.insert(
            "1.0",
            "Chest radiograph demonstrates patchy right lower-zone opacity. No pleural effusion. "
            "Recommend clinical correlation for pneumonia.",
        )
        self._show_image(self.image_path, self.image_preview, "image")
        self._run_analysis()

    def _choose_image(self) -> None:
        path = filedialog.askopenfilename(title="Select medical image", filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
        if path:
            self.image_path = Path(path)
            self._show_image(self.image_path, self.image_preview, "image")
            self._run_analysis()

    def _run_analysis(self) -> None:
        if not self.image_path:
            messagebox.showwarning("Missing Image", "Please upload a medical image first.")
            return
        try:
            self.image_result = analyze_medical_image(self.image_path, self.modality.get(), self.output_dir)
            self.report_result = analyze_report_text(self.report_text.get("1.0", "end"))
            self._show_image(self.image_result.heatmap_path, self.heatmap_preview, "heatmap")
        except Exception as exc:
            messagebox.showerror("Analysis Error", str(exc))
            return
        self._render_results()

    def _render_results(self) -> None:
        assert self.image_result is not None
        top = self.image_result.top_finding
        self.top_label.configure(text=top.label)
        self.confidence_label.configure(text=f"{top.confidence:.0%} confidence | {top.severity} estimated severity")
        lines = [
            "IMAGE MODEL PREDICTIONS",
            *[f"{item.label}: {item.confidence:.0%} confidence, {item.severity} severity" for item in self.image_result.findings],
            "",
            "EXPLAINABILITY",
            f"Grad-CAM overlay saved to: {self.image_result.heatmap_path}",
            *top.evidence,
            "",
            "REPORT PARSING",
            self.report_result.summary if self.report_result else "No report analysis available.",
            "",
            "PATIENT SUMMARY",
            generate_patient_summary(self.image_result, self.report_result),
            "",
            "TECHNICAL NOTES",
            *self.image_result.technical_notes,
        ]
        self.results.configure(state="normal")
        self.results.delete("1.0", "end")
        self.results.insert("1.0", "\n".join(lines))
        self.results.configure(state="disabled")

    def _show_image(self, path: Path, label: tk.Label, target: str) -> None:
        image = Image.open(path).convert("RGB")
        image.thumbnail((470, 245))
        photo = ImageTk.PhotoImage(image)
        if target == "image":
            self._image_photo = photo
        else:
            self._heatmap_photo = photo
        label.configure(image=photo, text="")

