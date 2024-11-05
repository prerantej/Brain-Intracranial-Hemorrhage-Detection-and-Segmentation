# Brain Intracranial Hemorrhage Detection and Segmentation

## Project Overview

## Abstract

This study uses machine learning on medical imaging and demographic data to improve the accuracy of bleeding detection. The effectiveness of six models—Logistic Regression, Random Forest, SVM, Gradient Boosting, XGBoost, and a Neural Network—in detecting hemorrhages was assessed. According to the results, XGBoost and Neural Networks were the best options for practical use since they had the highest accuracy and the lowest log loss.

## Introduction
For better patient outcomes and prompt care, hemorrhage identification is essential. Accurately interpreting imaging data, however, takes time and skill, which may postpone therapy. By automatically examining data patterns, machine learning provides a way to speed up this process and identify hemorrhages quickly and precisely. This study uses a large dataset of medical and demographic data to evaluate six machine learning algorithms in order to determine the best model for bleeding diagnosis.

## Contributions
•	Model Evaluation: Compared six machine learning models for hemorrhage detection.
•	Data Integration: Merged medical imaging data with demographic details to improve prediction accuracy.
•	Regularization & Validation: Applied regularization and cross-validation for robust model performance.
•	Optimal Model Selection: Identified XGBoost and Neural Networks as top-performing models for accuracy and reliability.

## Literature Review
Current machine learning research in medical diagnostics focuses on individual models such as SVM and Random Forests. Although ensemble and neural network methods have shown great accuracy in imaging-based classification, they are frequently not interpretable. Research has also demonstrated that by capturing more patient-specific risk factors, combining demographic information with imaging features might improve prediction models even more.

## Proposed Methodology 
Our approach incorporates patient demographics and imaging data. Normalization, encoding categorical variables, and handling missing values are all part of the data preprocessing workflow. Stratified k-fold cross-validation was used to train and assess six machine learning models, with regularization strategies used to reduce overfitting. Accuracy and log loss measurements were used to evaluate performance.
<img width="927" alt="image" src="https://github.com/user-attachments/assets/966ef12f-43d9-4195-8e9a-a0c11b7a5cfa">

## Experimental Setup and Results
A combined dataset of demographic and imaging data was used to train all models. Each model's hyperparameters were adjusted, and training was carried out on a workstation with enough processing capability. The following are each model's primary metrics:
<img width="938" alt="image" src="https://github.com/user-attachments/assets/195ba107-7d41-4c44-8001-af67cf5cc860">
<img width="938" alt="image" src="https://github.com/user-attachments/assets/532d8c2d-5db7-45dc-9152-927c5029ceb1">

## Comparison with State-of-the-Art
Our use of demographic information and imaging data resulted in a significant increase in prediction accuracy when compared to earlier research, especially with the XGBoost and Neural Network models. Prior research has shown that ensemble techniques perform well; nevertheless, our study is unique in that it combines thorough data preprocessing and robust evaluation to achieve state-of-the-art hemorrhage detection findings.

## Conclusion & Future Scope
According to this study, XGBoost and Neural Network models perform better than conventional methods in bleeding detection, with minimal log loss and high accuracy. To further enhance generalization and predictive power in various clinical contexts, future research could investigate deeper neural networks, include more clinical characteristics, and test the models on bigger datasets.

## References
1.	Liao, X., & Lu, Y. (2020). Machine learning-based hemorrhage detection in medical images: Techniques and applications. IEEE Access, 8, 12456-12465.
2.	Guan, Q., & Huang, Y. (2020). Ensemble learning approaches in medical image classification. Computers in Biology and Medicine, 123, 103905.
3.	Shao, Y., Xia, Z., & Yu, Y. (2018). The integration of imaging and non-imaging features for improved hemorrhage diagnosis. Journal of Biomedical Informatics, 84, 64-75.
4.	Lundervold, A. S., & Lundervold, A. (2019). An overview of deep learning in medical imaging focusing on MRI. Zeitschrift für Medizinische Physik, 29(2), 102-127.
5.	Litjens, G., Kooi, T., Bejnordi, B. E., et al. (2017). A survey on deep learning in medical image analysis. Medical Image Analysis, 42, 60-88.


## Dataset

Used dataset from https://www.kaggle.com/datasets/vbookshelf/computed-tomography-ct-images that include images of **brain CT scans**.

## Novelty

This study uniquely combines medical imaging data with demographic information to improve hemorrhage detection accuracy, a rarely explored integration in existing literature. By evaluating six diverse machine learning models under the same conditions, we systematically identify optimal models for clinical use. The robust validation techniques applied enhance the generalization of results, positioning this approach as a reliable tool for real-world diagnostics.

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
https://colab.research.google.com/drive/1yk18AdVvXUlvv6pGj8ITJBVcd2579CrK?usp=sharing
```

6. Train the model and evaluate the segmentation results.

## Results

- The model outputs segmentation maps for CT images, where the hemorrhage regions are highlighted.

## Future Scope

- **Multi-class Hemorrhage Segmentation**: Extend the project to handle different types of intracranial hemorrhages (subdural, epidural, etc.).
- **3D CT Analysis**: Expand from 2D to 3D segmentation for more accurate hemorrhage detection across multiple CT slices.
- **Transfer Learning**: Apply transfer learning from pre-trained models to improve performance with limited data.
- **Attention Mechanisms**: Incorporate attention gates to improve the model's focus on hemorrhage regions.
- **GAN-based Data Augmentation**: Use GANs to generate synthetic CT scans for training, helping to address data scarcity.
