from fpdf import FPDF
import os

pdf = FPDF()
pdf.add_page()

# Title
pdf.set_font("Arial", 'B', 16)
pdf.cell(200, 10, txt="Steel Defect Detection - Project Files Report", ln=True, align='C')
pdf.ln(10)

# Content
pdf.set_font("Arial", size=12)

files_summary = [
    ("src/models.py", "Defines the AI models (U-Net, FPN, DeepLabV3+)."),
    ("src/dataset.py", "Loads images and mask data for training."),
    ("src/config.py", "Stores hyperparameters (batch size, learning rate)."),
    ("src/utils.py", "Helper functions for metrics and encoding."),
    ("train.py", "Main script to train the models."),
    ("predict.py", "Runs inference on local images."),
    ("api/app.py", "FastAPI backend server for the web app."),
    ("static/", "Frontend UI files (HTML, CSS, JS)."),
    ("app.py", "Gradio app for Hugging Face Spaces free-tier."),
    ("notebooks/02_live_demo_colab.ipynb", "Public Colab web demo."),
    ("space_static/", "Static landing page deployed to Hugging Face."),
    ("Dockerfile", "Docker container configuration."),
    ("deploy.py", "Automation script for pushing code and models.")
]

for file_path, desc in files_summary:
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(80, 10, txt=f"- {file_path}: ", ln=False)
    pdf.set_font("Arial", size=12)
    pdf.cell(100, 10, txt=desc, ln=True)

output_path = os.path.join(os.getcwd(), "Project_Report.pdf")
pdf.output(output_path)
print(f"Generated PDF: {output_path}")
