# AI Medical Diagnosis Assistant

A desktop medical imaging prototype that analyzes X-rays, CT scans, MRI images, retinal images, and clinical report text. It produces disease-risk predictions, Grad-CAM-style visual explanations, and a patient-friendly summary.

## Features

- Upload medical images for X-ray, CT scan, MRI, or retina workflows.
- Screens for pneumonia-like lung opacity, fracture-like high contrast lines, tumor-like lesions, kidney stone-like dense spots, and diabetic-retinopathy-like retinal signs.
- Generates a Grad-CAM-style heatmap overlay that highlights regions influencing the top prediction.
- Parses medical report text for concerning, urgent, and reassuring language.
- Creates a patient-friendly summary with safety guidance and clinician-review reminders.
- Includes generated sample images so the dashboard works on first launch.

## Run

```powershell
pip install -r requirements_ai_medical_diagnosis.txt
python run_ai_medical_diagnosis_assistant.py
```

## Model Note

The app is structured for validated deep learning weights, but ships with a lightweight fallback classifier so it can run locally without downloading large medical models. Production use would require clinically validated datasets, regulatory review, privacy controls, calibration, audit logging, and radiologist oversight.

This is decision-support software for education and prototyping only. It is not a medical device and must not be used as a final diagnosis or treatment recommendation.

