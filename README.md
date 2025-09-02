# JPEG-Compression-Pipeline with Truncation

**Dept. of Electronics and Electrical Communication Engineering** **Indian Institute of Technology Kharagpur** **Course:** EC69211 – Image and Video Processing Laboratory  

**Team Members** 
- M A Rama Murthy – 21EC39016  
- B H S Sagar – 21EC39038  

---

## Project Overview
This project implements a **modified JPEG compression pipeline**. Along with the standard JPEG stages, a **truncation step** is added that discards a fixed number of **high-frequency DCT coefficients**. This improves the **compression ratio** by creating sparser data while preserving the **visual quality** of images.

---

## Pipeline Stages
1. **Color Space Conversion (RGB → YCbCr)**
2. **Chrominance Subsampling (Cb, Cr downsampling)**
3. **Block-wise DCT (8×8 blocks)**
4. **Quantization + Truncation (retain first N coefficients)**
5. **Zigzag Scanning + Run-Length Encoding**
6. **Huffman/Entropy Coding**
7. **Decoding adapted for truncated data**

---

## Requirements
Install dependencies:
```bash
pip install opencv-python matplotlib numpy
````

-----

## Usage

### Single Image

```bash
python jpeg_truncation.py
```

You will be prompted for the image path, block size (e.g., 8), number of retained coefficients, and color/grayscale option.

Outputs: Original vs Compressed image + Compression ratio + PSNR

### Folder of Images

```bash
python jpeg_truncation.py
```

Enter a folder path and image type (color/grayscale).

Outputs: **PSNR vs Compression Ratio graph**

-----

## Results

  * **Compression Ratio** examples: **8.3, 28.7, 39, 41**
  * Quality is measured using **PSNR**.
  * There is a trade-off: **Higher compression → Lower PSNR**.

-----

## Notes

  * Supports both **color** and **grayscale** images.
  * Only works for **8-bit images** (JPEG limitation).
  * Loses detail in **sharp edges/text** at high compression.

-----

## Authors

  * **Medavarapu Atchutha Rama Murthy** – 21EC39016
  * **Banisetty Hema Sai Sagar** – 21EC39038

```
```
