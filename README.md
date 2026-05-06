MobileDiffusion Efficient AI Project
=====================================================================
Overview
=====================================================================
This project explores efficiency tradeoffs in diffusion-based text-to-image generation models inspired by the MobileDiffusion paper. 
The main goal of our experiment was to test how changing the number of diffusion sampling steps affects:

- Image generation speed (latency)
- GPU memory usage
- Text-image alignment quality using CLIP score
- Overall visual image quality

The experiment was implemented using Stable Diffusion v1.5 in Google Colab with GPU acceleration on a Tesla T4 GPU.
=====================================================================
Project Motivation
=====================================================================
Diffusion models can generate highly realistic images from text prompts, but they usually require many denoising steps to produce high-quality results. 
Because of this, image generation can become slow and computationally expensive, especially for mobile or lower-resource devices.

The MobileDiffusion paper focuses on improving diffusion efficiency by reducing computational cost while still maintaining good image quality. 
Inspired by this idea, our project evaluates how reducing sampling steps affects speed, memory usage, and output quality.
=====================================================================
Experiment Description
=====================================================================
Images were generated using Stable Diffusion with different numbers of sampling steps:

- 1 step
- 4 steps
- 8 steps
- 20 steps
- 50 steps

For each configuration, we recorded:

- Latency (generation time in seconds)
- Peak GPU memory usage
- CLIP score for text-image alignment

Several prompts were tested to compare how sampling steps affect both efficiency and image quality.
=====================================================================
Main Findings
=====================================================================
Our experiments showed several clear trends:

- Increasing sampling steps significantly increased image generation time.
- GPU memory usage stayed relatively constant across different step counts.
- CLIP scores improved quickly at lower step counts and then stabilized at higher values.
- Very low sampling steps produced noisy or distorted images, while higher step counts produced clearer and more realistic outputs.

Overall, the experiment showed that reducing sampling steps makes image generation much faster, but image quality begins to decrease at very low step counts. This highlights the tradeoff between computational efficiency and visual quality in diffusion models.
=====================================================================
Repository Structure
=====================================================================
mobilediffusion-efficient-ai/
│
├── notebooks/
│   └── sampling_experiment_github_clean.ipynb
│
├── results/
│   ├── figures/
│   ├── images/
│   └── metrics/
│
├── requirements.txt
├── README.md
└── src/
=====================================================================
Figures Included
=====================================================================
The repository includes:

- Latency vs Sampling Steps graph
- GPU Memory Usage vs Sampling Steps graph
- CLIP Score vs Sampling Steps graph
- Example generated image comparisons across different sampling steps
=====================================================================
Technologies Used
=====================================================================
- Python
- PyTorch
- Hugging Face Diffusers
- Stable Diffusion v1.5
- CLIP
- Google Colab
- Pandas
- Matplotlib
=====================================================================
Setup Instructions
=====================================================================
- Install Dependencies
   pip install -r requirements.txt
- Run the Notebook
  Open the notebook inside the notebooks/ folder using:
  - Google Colab
  - Jupyter Notebook
** GPU acceleration is recommended for running the diffusion model efficiently.
=====================================================================
Attribution and References
=====================================================================
This project was inspired by the MobileDiffusion paper and uses publicly available pretrained diffusion models from Hugging Face.

Models Used

Stable Diffusion:
- runwayml/stable-diffusion-v1-5
CLIP:
- openai/clip-vit-base-patch32
Libraries Used
- PyTorch
- Hugging Face Diffusers
- Transformers
This project was developed for educational and research purposes as part of a machine learning course project.
=====================================================================
Notes
-Some additional experiments and extensions may be added later as part of ongoing work related to MobileDiffusion-style optimization and efficiency analysis.
-The notebook was optimized for Github rendering compatibility. If needed, the notebook can also be opened directly in Google Colab. 
