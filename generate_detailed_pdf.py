from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
import os

def create_comprehensive_report(filename):
    doc = SimpleDocTemplate(filename, pagesize=letter,
                            rightMargin=50, leftMargin=50,
                            topMargin=50, bottomMargin=50)
    
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        'CustomTitle', parent=styles['Title'], fontName='Helvetica-Bold',
        fontSize=22, spaceAfter=20, textColor=HexColor("#1A202C")
    )
    heading1_style = ParagraphStyle(
        'Heading1', parent=styles['Heading1'], fontName='Helvetica-Bold',
        fontSize=16, spaceBefore=15, spaceAfter=10, textColor=HexColor("#2D3748")
    )
    heading2_style = ParagraphStyle(
        'Heading2', parent=styles['Heading2'], fontName='Helvetica-Bold',
        fontSize=13, spaceBefore=10, spaceAfter=6, textColor=HexColor("#4A5568")
    )
    normal_style = ParagraphStyle(
        'CustomNormal', parent=styles['Normal'], fontName='Helvetica',
        fontSize=11, leading=16, textColor=HexColor("#2D3748"), spaceAfter=10
    )
    bullet_style = ParagraphStyle(
        'BulletStyle', parent=normal_style, spaceAfter=4
    )

    story = []

    # Title
    story.append(Paragraph("Steel Defect Detection: Comprehensive Architecture & Implementation Report", title_style))
    story.append(Spacer(1, 12))

    # 1. Executive Summary
    story.append(Paragraph("1. Executive Summary", heading1_style))
    story.append(Paragraph(
        "This project was developed for the Severstal Steel Defect Detection initiative. The goal is to automatically locate and segment surface defects on sheet steel using advanced computer vision techniques. "
        "Originally consisting of scattered Jupyter notebooks and ad-hoc scripts, the codebase has been completely refactored into a modular, highly scalable, and production-ready deep learning pipeline. "
        "The final system leverages a robust 3-model ensemble, a highly optimized REST API, a modern 'Glassmorphism' web interface, and zero-cost cloud deployment workflows.",
        normal_style
    ))

    # 2. Machine Learning Methodology
    story.append(Paragraph("2. Machine Learning Methodology", heading1_style))
    story.append(Paragraph(
        "To achieve maximum accuracy and generalization, the system utilizes a heterogeneous ensemble of three state-of-the-art segmentation architectures. Each model is designed to capture different levels of spatial features and contextual information:",
        normal_style
    ))
    
    models_list = [
        ListItem(Paragraph("<b>U-Net with ResNet-34:</b> Excels at capturing fine-grained spatial details through its symmetric encoder-decoder structure and skip connections.", bullet_style)),
        ListItem(Paragraph("<b>FPN (Feature Pyramid Network) with SE-ResNeXt50:</b> Highly effective at detecting defects at multiple scales by merging high-level semantic features with low-level spatial features.", bullet_style)),
        ListItem(Paragraph("<b>DeepLabV3+ with EfficientNet-B3:</b> Utilizes Atrous Spatial Pyramid Pooling (ASPP) to capture multi-scale context, making it particularly strong at segmenting large, amorphous defect patches.", bullet_style))
    ]
    story.append(ListFlowable(models_list, bulletType='bullet', spaceAfter=10))

    story.append(Paragraph("<b>Post-Processing & Noise Reduction:</b>", heading2_style))
    story.append(Paragraph(
        "Raw model probabilities are averaged at the sigmoid level. To eliminate false positives and isolated noise pixels, a deterministic Connected Component Analysis (CCA) is applied. "
        "Any predicted defect component with an area smaller than a class-specific pixel threshold (e.g., 300 to 2000 pixels) is automatically discarded. This significantly improves precision.",
        normal_style
    ))

    # 3. Codebase Architecture
    story.append(Paragraph("3. Codebase Architecture & File Responsibilities", heading1_style))
    story.append(Paragraph(
        "The repository follows best practices for production AI systems, strictly separating the machine learning logic, the backend API, and the frontend user interface.",
        normal_style
    ))

    # Source code
    story.append(Paragraph("A. Core Machine Learning Module (src/)", heading2_style))
    src_list = [
        ListItem(Paragraph("<b>src/models.py:</b> The model factory. Responsible for dynamically instantiating the ensemble architectures using the `segmentation_models_pytorch` framework.", bullet_style)),
        ListItem(Paragraph("<b>src/dataset.py:</b> Contains the PyTorch `Dataset` classes. It manages image loading, applies heavy data augmentations via `albumentations`, and parses Kaggle's Run-Length Encoded (RLE) strings into strict binary tensors.", bullet_style)),
        ListItem(Paragraph("<b>src/config.py:</b> Centralizes all project hyperparameters (batch sizes, learning rates, epochs, image dimensions) using Python `dataclasses`.", bullet_style)),
        ListItem(Paragraph("<b>src/utils.py:</b> Houses mathematical utilities, specifically the implementation of the Dice Coefficient metric used to evaluate segmentation overlap.", bullet_style))
    ]
    story.append(ListFlowable(src_list, bulletType='bullet', spaceAfter=10))

    # Training and Inference
    story.append(Paragraph("B. Execution Scripts", heading2_style))
    exec_list = [
        ListItem(Paragraph("<b>train.py:</b> The primary training loop. Built for speed and efficiency, it implements Automatic Mixed Precision (AMP) via `torch.cuda.amp` to reduce VRAM usage and accelerate training times.", bullet_style)),
        ListItem(Paragraph("<b>predict.py:</b> A standalone offline inference engine. Designed to process batches of local images through the ensemble and output overlay visualizations.", bullet_style))
    ]
    story.append(ListFlowable(exec_list, bulletType='bullet', spaceAfter=10))

    # API and Frontend
    story.append(Paragraph("C. Production Application", heading2_style))
    app_list = [
        ListItem(Paragraph("<b>api/app.py:</b> A lightning-fast REST API built on FastAPI. It loads the multi-gigabyte ensemble into memory asynchronously, exposes a `/api/predict` endpoint for binary image uploads, and serves the static frontend.", bullet_style)),
        ListItem(Paragraph("<b>static/ (HTML/CSS/JS):</b> The user-facing frontend. Features a modern, 'Glassmorphism' dark-mode aesthetic. It allows users to drag-and-drop steel images and renders the API's segmentation masks directly onto an HTML5 canvas in real-time.", bullet_style))
    ]
    story.append(ListFlowable(app_list, bulletType='bullet', spaceAfter=10))

    # 4. Cloud Deployment Strategy
    story.append(Paragraph("4. Cloud Deployment & Automation", heading1_style))
    story.append(Paragraph(
        "Because the ensemble weights exceed standard Git repository limits (approx. 760 MB), a distributed deployment strategy was implemented using GitHub and the Hugging Face Hub.",
        normal_style
    ))
    
    cloud_list = [
        ListItem(Paragraph("<b>deploy.py & scratch_hf.py:</b> Custom automation scripts that authenticate via API tokens to programmatically create GitHub repositories, push code, and upload the heavy `.pth` weights directly to the Hugging Face Hub.", bullet_style)),
        ListItem(Paragraph("<b>app.py (Root):</b> A native Gradio application designed specifically to run on Hugging Face Spaces free-tier hardware.", bullet_style)),
        ListItem(Paragraph("<b>space_static/index.html:</b> A highly optimized, responsive static landing page hosted on Hugging Face. It serves as the public face of the project, explaining the architecture and directing users to the live demo.", bullet_style)),
        ListItem(Paragraph("<b>notebooks/02_live_demo_colab.ipynb:</b> An interactive Google Colab notebook. It dynamically downloads the weights from Hugging Face, starts the FastAPI server, and exposes it to the public internet using Cloudflare Tunnels (`cloudflared`). This provides a 100% free, highly available public demonstration link.", bullet_style)),
        ListItem(Paragraph("<b>generate_colab_nb.py:</b> A CI/CD-style script that programmatically generates the Colab notebook to ensure the demo always stays perfectly synchronized with the repository code.", bullet_style)),
        ListItem(Paragraph("<b>Dockerfile & docker-compose.yml:</b> Standardized container definitions. Allows any developer to spin up the entire API, frontend, and environment dependencies locally with a single command.", bullet_style))
    ]
    story.append(ListFlowable(cloud_list, bulletType='bullet', spaceAfter=10))

    # Conclusion
    story.append(Paragraph("Conclusion", heading1_style))
    story.append(Paragraph(
        "This project represents a complete, end-to-end machine learning lifecycle. It transitions from raw data exploration and model training to robust API development, premium UI/UX design, and zero-cost distributed cloud deployment. The resulting system is fully prepared for real-world industrial application and peer review.",
        normal_style
    ))

    # Build PDF
    doc.build(story)
    
if __name__ == "__main__":
    output_path = os.path.join(os.getcwd(), "Comprehensive_Project_Report.pdf")
    create_comprehensive_report(output_path)
    print(f"Generated PDF: {output_path}")
