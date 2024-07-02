# Gan-n

## Project Summary

### Overview

This project implements a Generative Adversarial Network (GAN) to generate synthetic data for imbalanced datasets. The generated data is intended to augment the minority class in binary classification tasks, improving model performance.

### Key Components

1. **Data Preprocessing**

   - Load and preprocess data from a CSV file.
   - Separate minority and majority classes.
   - Use SMOTEEN to generate additional samples for the minority class.

2. **GAN Architecture**

   - **Generator:** Maps random noise vectors (latent space) to synthetic data samples.
   - **Discriminator:** Distinguishes between real and synthetic data samples.

3. **Training the GAN**

   - Train the generator and discriminator iteratively.
   - Use the Binary Cross-Entropy loss function.
   - Optimize using Adam optimizer.

4. **Main Script**
   - Load and preprocess data.
   - Train the GAN.
   - Save the trained generator model.

### Files and Directories

- **src/**: Contains modules for GAN training and data preprocessing.
- **models/**: Directory to save trained GAN models.
- **Datasets/**: Contains the CSV data files for training.

### How to Run

1. **Set Up Environment**

   - Install dependencies: `pip install -r requirements.txt`

2. **Train the GAN**
   - Run the main script: `python3 main.py`
   - Specify the target column and data file path in the script.

### Example Usage

```bash
python3 main.py
```

### Dependencies

- PyTorch
- NumPy
- SMOTEEN (Imbalanced-learn)

### Goals

- Address class imbalance in datasets.
- Improve classification performance by augmenting training data with synthetic samples.
- Utilizes docker and DVC for data version controlling, re-experimenation and containerization
