# MH-WMG: A Multi-Head Wavelet-Based MobileNet with Gated Linear Attention for Power Grid Fault Diagnosis

This repository accompanies our academic paper **"MH-WMG: A Multi-Head Wavelet-Based MobileNet with Gated Linear Attention for Power Grid Fault Diagnosis"**, published in *MDPI Applied Sciences*.
The project provides code, datasets, and models to reproduce the experiments described in the paper.

> The full article is open-source and can be reached at [MDPI](https://www.mdpi.com/3536174).

---

## 📖 Project Overview
MH-WMG is a novel deep learning architecture that integrates **wavelet-based preprocessing**, a **MobileNet backbone**, and **gated linear attention** to perform **fault diagnosis in power grids**.  
The pipeline combines time-series and wavelet scalogram images to achieve high accuracy and interpretability in classifying fault location, type, and distance.

---

## 🚀 How to Run

1. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

2. **Download datasets**  
   Run the downloader script to fetch both image and time-series data from Google Drive:  
   ```bash
   python3 data_downloader.py
   ```
   You can find the full description of the dataset inside the `/data` folder.

3. **Train the model**  
   Execute the main training script:  
   ```bash
   python3 MH-WMG.py
   ```

   > The same workflow is also available as a Jupyter Notebook under the `notebooks/` directory (`MH-WMG.ipynb`).

---

## 📂 Repository Structure

```
MH-WMG/
│
├── MH-WMG.py               		# Main training and evaluation script
├── data_downloader.py      		# Downloads datasets from Google Drive
├── requirements.txt        		# Required Python packages
├── notebooks/
│   └── MH-WMG.ipynb        		# Notebook version of the main script
├── notebooks/
│   └── KundurTwoAreaSystemModified.slx # Related Simulink/Matlab Kundur Two Area System
│   └── runSystem.m        		# Short Circuit Faults Application on The System
└── README.md               		# Project documentation
```

---

## 📜 Citation
If you use the **data** or the **pipeline**, please cite the paper below.

**APA** <br>
Alkhanafseh, Y., Akinci, T. C., Martinez-Morales, A. A., Seker, S., & Ekici, S. (2025). MH-WMG: A Multi-Head Wavelet-Based MobileNet with Gated Linear Attention for Power Grid Fault Diagnosis. *Applied Sciences, 15*(20), 10878. https://doi.org/10.3390/app152010878

**BibTeX**
```bibtex
@article{Alkhanafseh2025MHW_MG,
  title   = {MH-WMG: A Multi-Head Wavelet-Based MobileNet with Gated Linear Attention for Power Grid Fault Diagnosis},
  author  = {Alkhanafseh, Yousef and Akinci, Tahir Çetin and Martinez-Morales, Alfredo A. and Seker, Serhat and Ekici, Sami},
  journal = {Applied Sciences},
  year    = {2025},
  volume  = {15},
  number  = {20},
  pages   = {10878},
  doi     = {10.3390/app152010878}
}
```

---

## 📝 License
This code is released for academic and research purposes only. Please check the license terms in this repository.

---

## ✉️ Contact
For questions or collaborations, please contact:
**Yousef Alkhanafseh** – [alkhanafseh15@gmail.com]

---
