# Team-VERTEX
This project builds a semantic segmentation model to classify and detect objects in real-world scenarios. Through dataset preprocessing, model training, and evaluation, we target high accuracy and reproducibility, with clear documentation and visual storytelling for impact.


# 1. Project Overview

This project implements a Deep Learning-based Semantic Segmentation model trained on a synthetic desert dataset generated using Duality AI’s Falcon Digital Twin platform.

The objective is to accurately classify each pixel in an image into one of the following 10 classes:

- Trees
- Lush Bushes
- Dry Grass
- Dry Bushes
- Ground Clutter
- Flowers
- Logs
- Rocks
- Landscape
- Sky

The system is designed to simulate real-world off-road autonomy applications such as Unmanned Ground Vehicles (UGVs), where accurate terrain understanding is critical.

---

# 2. Problem Statement

Manual segmentation of terrain data is time-consuming and inconsistent.  
UGVs require reliable pixel-level scene understanding to navigate safely in harsh environments.

This project aims to:

- Train a robust semantic segmentation model
- Achieve high Intersection over Union (IoU)
- Generalize to unseen desert environments
- Provide reproducible and well-documented results

---

# 3. Functional Requirements

The system contains the following functional modules:

### 3.1 Data Processing Module
- Load training, validation, and test datasets
- Apply preprocessing and augmentation
- Maintain strict separation between train/val/test sets

### 3.2 Model Training Module
- Train segmentation model on provided dataset
- Save model checkpoints
- Log training and validation metrics

### 3.3 Evaluation Module
- Perform inference on unseen test images
- Calculate IoU score
- Generate prediction outputs
- Provide loss and performance metrics

---

# 4. Non-Functional Requirements

- Performance: Efficient training and inference
- Reliability: Stable results across runs
- Maintainability: Modular folder structure
- Scalability: Can extend to larger datasets
- Reproducibility: Fixed seed and documented parameters
- Error Handling: Validation of file paths and dataset splits

---

# 5. System Architecture

Input Image  
↓  
Preprocessing  
↓  
Segmentation Model (CNN / UNet / DeepLab)  
↓  
Pixel-wise Classification  
↓  
Evaluation (IoU, Loss Metrics)  
↓  
Output Segmented Image  

---

# 6. Dataset Description

Source: Duality AI Falcon Digital Twin Platform  

Dataset Includes:
- Train images (RGB + segmentation masks)
- Validation images
- Test images (unseen environment)

Important:
Test images were NOT used for training (as per competition guidelines).

---

# 7. Model Details

Model Used: [UNet / DeepLabV3 / Custom CNN]  
Loss Function: Cross Entropy Loss  
Optimizer: Adam  
Learning Rate: 0.001  
Batch Size: 8  
Epochs: 50  

Evaluation Metric:
- Mean Intersection over Union (IoU)

---

# 8. Environment & Dependencies

## 8.1 Python Version
Python 3.9 recommended

## 8.2 Create Environment

bash conda create -n segmentation python=3.9 -y
conda activate segmentation

## 8.3 Install Dependencies
pip install torch torchvision numpy opencv-python matplotlib tqdm

# 9. Project Structure
Offroad_Segmentation/
│
├── train.py
├── test.py
├── models/
├── dataset/
│   ├── train/
│   ├── val/
│   ├── testImages/
├── checkpoints/
├── runs/
├── README.md
├── statement.md

10. Step-by-Step Instructions to Run
Step 1: Activate Environment
conda activate segmentation

Step 2: Train the Model
python train.py

This will: Train the model
           Save checkpoints in /runs or /checkpoints
           Display loss and validation metrics

Step 3: Test the Model on Unseen Images
python test.py

This will:  Generate predictions
            Compute IoU score
            Save segmented output images

# 11. How to Reproduce Final Results
To reproduce the reported results:
Use the exact dataset split provided.

Do NOT use test images for training.

Set random seed to 42.


Train using:
python train.py --epochs 50 --batch_size 8 --lr 0.001

Run:
python test.py

Expected Output:
IoU Score: [Insert Your Final IoU]
Loss Curve: Steady decreasing trend
Segmented output images in output folder

# 12. Expected Outputs & Interpretation
After training and testing:
Generated Outputs:
Model weights (.pth file)
Predicted segmentation images
Console metrics (IoU, loss)
Interpretation Guide:
Higher IoU → Better overlap with ground truth
Decreasing loss → Proper learning
High confusion in small classes → Possible class imbalance
Overfitting → Training loss ↓ but validation loss ↑

# 13. Testing Approach
Evaluated only on unseen test environment
Measured Mean IoU
Analyzed failure cases (e.g., misclassification of Logs vs Rocks)
Compared multiple hyperparameter settings

# 14. Challenges Faced
Class imbalance
Occlusion of small objects
Overfitting during early training
OpenMP environment configuration issues
Solutions:
Data augmentation
Learning rate tuning
Clean conda environment setup
Model checkpointing

# 15. Future Enhancements
Domain adaptation for real-world images
Attention-based segmentation models
Self-supervised pretraining
Model compression for edge deployment
16. References
Duality AI Falcon Platform
PyTorch Documentation
UNet Paper (Ronneberger et al.)
DeepLabV3 Paper
