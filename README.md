# Brain Intracranial Hemorrhage Detection and Segmentation

## Project Overview

This project aims to detect and segment **intracranial hemorrhages** in brain CT images. Intracranial hemorrhage is a critical condition caused by bleeding within the brain due to trauma or stroke, and accurate detection and segmentation are essential for diagnosis and treatment planning.

Using a Convolutional Neural Network (CNN) for detection and **ResUNet** for segmentation, this project provides a method for localizing hemorrhages in CT scans with high precision. The project employs **Dice Loss** and **Dice Coefficient** for measuring the performance of the segmentation.

## Dataset

Used dataset from https://www.kaggle.com/datasets/vbookshelf/computed-tomography-ct-images that include images of **brain CT scans**.

## Novelty

This project applies **ResUNet** for hemorrhage segmentation, which is an advanced version of the U-Net architecture, incorporating residual connections to improve training efficiency and segmentation accuracy. By utilizing **Dice Loss** instead of standard loss functions, the project achieves better performance on small and imbalanced hemorrhage regions.

## Requirements

To run this project, you will need the following packages:
- TensorFlow
- Keras
- OpenCV
- Numpy
- Matplotlib

You can install the necessary dependencies using:
pip install -r requirements.txt


## How to Run

1. Clone the repository:
```bash
git clone https://github.com/your-username/brain-hemorrhage-detection.git
```


3. Place the CT image dataset in the `data/` directory.

4. Open and run the Jupyter notebook:
```bash
jupyter notebook brain-stroke-detection-using-ct-images.ipynb
```

6. Train the model and evaluate the segmentation results.

## Results

- The model outputs segmentation maps for CT images, where the hemorrhage regions are highlighted.
- Metrics such as **Dice Coefficient** are used to assess segmentation performance.
<img width="927" alt="image" src="https://github.com/user-attachments/assets/bc238ed2-911e-4198-9750-8731f1543dae">
<img width="938" alt="image" src="https://github.com/user-attachments/assets/da973404-fd85-44e3-8aa3-8bc37c32d229">
<img width="938" alt="image" src="https://github.com/user-attachments/assets/532d8c2d-5db7-45dc-9152-927c5029ceb1">

## Jacard Coefficient and Jacard Loss
- Jacard Coefficient: Measures the similarity between two sets by dividing the size of their intersection by the size of their union. It is used in tasks like clustering, where higher values indicate greater similarity.
- Jacard Loss: A loss function derived from the Jaccard Coefficient, commonly used in image segmentation tasks. It penalizes the model based on how much the predicted output deviates from the actual target.
<img width="938" alt="image" src="https://github.com/user-attachments/assets/8765c065-94f9-4400-9216-d80c070ae7c4">

## Future Scope

- **Multi-class Hemorrhage Segmentation**: Extend the project to handle different types of intracranial hemorrhages (subdural, epidural, etc.).
- **3D CT Analysis**: Expand from 2D to 3D segmentation for more accurate hemorrhage detection across multiple CT slices.
- **Transfer Learning**: Apply transfer learning from pre-trained models to improve performance with limited data.
- **Attention Mechanisms**: Incorporate attention gates to improve the model's focus on hemorrhage regions.
- **GAN-based Data Augmentation**: Use GANs to generate synthetic CT scans for training, helping to address data scarcity.
