"""Application entry point for the AI Medical Diagnosis Assistant."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ai_medical_diagnosis_assistant.gui import MedicalDiagnosisApp
else:
    from .gui import MedicalDiagnosisApp


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    app = MedicalDiagnosisApp()
    app.mainloop()


if __name__ == "__main__":
    main()

