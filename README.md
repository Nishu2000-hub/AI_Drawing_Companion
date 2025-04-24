# Interactive AI Drawing Companion

## Project Overview

The aim of this project is to develop an interactive game where users draw objects in real time, and an AI system attempts to guess what has been drawn. This project combines computer vision, deep learning, and human-computer interaction to create an engaging and educational experience. In addition to recognizing a wide range of hand-drawn sketches, the system is also designed to handle cases where the drawing does not belong to any of the known categories—a challenge known as out-of-distribution (OOD) detection. Addressing these cases adds a valuable research component to the project.

## High-Level System Architecture

### 1. Data Preprocessing and Augmentation

- **Primary Data Source:**  
  The project leverages the [Quick, Draw! dataset](https://quickdraw.withgoogle.com/data) by Google, which contains over 50 million hand-drawn sketches across a variety of categories. The dataset is originally in vector form, and a preprocessing step is needed to convert these sketches into raster images compatible with convolutional neural networks (CNNs).

- **Preprocessing Techniques:**  
  - **Normalization:** Standardizing the input data.
  - **Resizing:** Ensuring all sketches are of uniform dimensions.
  - **Data Augmentation:** Applying transformations like rotations, translations, and scaling to increase the robustness of the model.

### 2. Neural Network Architecture

- **Core Recognition Engine:**  
  A deep convolutional neural network (CNN) will serve as the primary recognition engine. Due to the abstract and often variable nature of hand-drawn sketches, the architecture may also incorporate additional layers to capture sequential stroke information if the drawing sequence data is used.

- **Real-Time Performance:**  
  Given the interactive nature of the game, the neural network must be optimized for low-latency inference.
### 3. Out-of-Bound (OOD) Detection

Handling drawings that do not belong to any of the known categories is a central research aspect of this project. Several methods will be explored:

- **Confidence Thresholding:**  
  - After the network outputs class probabilities (typically through a softmax layer), a confidence threshold will be set.
  - If the maximum probability is below the threshold, the input will be flagged as “unknown.”
  

### 4. Interactive Feedback Mechanism

The system is designed to provide immediate and clear feedback to users:

- **Recognized Drawings:**  
  The system displays the predicted label along with a confidence score.

- **Unknown Drawings:**  
  If the drawing does not meet the criteria for any known category, the system can output an “unknown” label and prompt the user to refine their drawing or try another object.

This interactive feedback loop is crucial for maintaining user engagement and ensuring that the application is both challenging and enjoyable.

## Datasets and Data Requirements

### Primary Dataset

- **Quick, Draw! Dataset:**  
  This dataset is the backbone of the project, offering a diverse and extensive collection of sketches in hundreds of categories. Its variety is ideal for training a robust recognition model. However, due to the inherent variability in drawing styles, extensive preprocessing and data augmentation are required to ensure that the model can generalize effectively.

### Additional Data for Out-of-Bound Cases

To effectively address the challenge of OOD detection, additional data sources or methods may be needed:

- **Synthetic or Noise Images:**  
  - Generating random noise or abstract patterns that do not belong to any known category can serve as negative examples.
  
- **Alternative Sketch Datasets:**  
  - Other publicly available sketch datasets can provide examples of drawings that differ stylistically from those in the Quick, Draw! dataset.
  
- **User-Contributed Drawings:**  
  - In later phases, collecting sketches from actual users in a controlled environment can provide real-world examples of out-of-bound inputs.


## Key Features and Technical Considerations

### Feature Extraction

- **Visual Features:**  
  The model will extract key visual features such as edges, contours, and stroke sequences. These features are essential for recognizing the abstract representations typical of hand-drawn sketches.
- **Automated vs. Engineered Features:**  
  Depending on the architecture, these features might be learned automatically via convolutional layers or be explicitly engineered during preprocessing.

### Real-Time Inference

- **Performance Optimization:**  
The application demands real-time inference capabilities. To meet these requirements, techniques such as efficient model architectures and possible model quantization will be investigated.

### Evaluation Metrics

- **Standard Metrics:**  
  The classification performance will be evaluated using accuracy metrics.
- **OOD Metrics:**  
  Additional metrics such as the Area Under the Receiver Operating Characteristic (AUROC), precision, and recall for unknown detection will be crucial in assessing the effectiveness of the out-of-bound detection strategies.


  # Interactive AI Drawing Companion

## Part 2: Data Acquisition and Partitioning

### Overview

For this phase, we have acquired the **Sketch-RNN QuickDraw Dataset**, a subset of the Google Quick, Draw! dataset, to serve as our training and validation data. The dataset contains simplified vector sketches in `.npz` files, each corresponding to a specific category (e.g., `cat.npz`, `dog.npz`). Each file typically includes three arrays: `train`, `valid`, and `test`, providing a pre-made split for training, validation, and testing known categories. We will also collect or generate an **"unknown"** subset later in the semester to evaluate our model's ability to detect out-of-distribution inputs.

---

### 1. Data Source

- **Dataset Name:** [QuickDraw Dataset](https://quickdraw.withgoogle.com/data)  
- **Associated References/Papers:**
  - *Ha, D., Eck, D. (2017). A Neural Representation of Sketch Drawings.*  
    Available via the Google Research Blog and [Magenta Project](https://github.com/magenta/magenta).
  - Official Quick, Draw! documentation on [Data](https://quickdraw.withgoogle.com/data) and the Sketch-RNN model.
  - [ Open World Object Detection](https://openaccess.thecvf.com/content/CVPR2021/papers/Joseph_Towards_Open_World_Object_Detection_CVPR_2021_paper.pdf)

---

### 2. Differences Between the Training and Validation Subsets

1. **Purpose of Each Subset**  
   - **Training (train):** Used to optimize the model’s learnable parameters (weights, biases). The dataset includes tens of thousands of sketches per category to ensure robust coverage of drawing variations.
   - **Validation (valid):** Helps monitor overfitting and guides hyperparameter tuning. Although drawn from the same categories as the training set, it is kept separate so that performance metrics are not biased by overexposure to training data.

2. **Data Distribution**  
   - Both `train` and `valid` subsets share the same label categories (e.g., “cat”, “bicycle”), but the validation set is typically much smaller (around 2,500 samples per category) compared to the training set (around 70,000). This distribution ensures the validation set remains a representative, yet sufficiently distinct, sample for evaluating generalization.

3. **Hyperparameter Tuning and Early Stopping**  
   - We use the validation subset to adjust learning rates, network architectures, and regularization parameters. When validation performance stops improving or begins to degrade, we can apply early stopping to avoid overfitting.

---

### 3. Number of Distinct Objects and Samples

- **Categories:** Approximately 345 different classes (e.g., “cat,” “dog,” “apple,” “airplane”).  
- **Samples per Category:**  
  - **Training:** ~70,000 sketches  
  - **Validation:** ~2,500 sketches  
  - **Test (Known Categories):** ~2,500 sketches  

Because we are focusing on a subset of these categories initially, we may only download specific `.npz` files (e.g., `cat.npz`, `dog.npz`) to keep our project scope manageable. The full dataset, however, offers extensive variety for future experimentation.

---

### 4. Characterization of Samples

1. **Format and Resolution**  
   - **Vector Data:** Each `.npz` file stores sketches in a sequence of `[dx, dy, pen_state]`, representing stroke offsets and pen-lift events.  
   - **Simplification Process:**  
     1. Aligned to the top-left corner (min x, y = 0).  
     2. Uniformly scaled so the maximum coordinate is 255 (fitting a 256×256 region).  
     3. Resampled at ~1 pixel spacing.  
     4. Ramer–Douglas–Peucker (RDP) algorithm applied (epsilon=2.0) to remove redundant points.

2. **Sensors and Ambient Conditions**  
   - Data comes from a web-based game; there is no significant variation in lighting or sensor type since all drawings are captured digitally.

3. **Stroke vs. Image Representation**  
   - While the dataset is stored as vector strokes, we can convert them to raster images (e.g., 28×28 or 64×64) if a convolutional neural network is preferred.

4. **Example Usage**  
   - **RNN/LSTM** approaches can process stroke-by-stroke input.  
   - **CNN** approaches require rendering or rasterizing the vector data into images.

---

### 5. "Unknown" Subset for Final Evaluation

In addition to the known categories, we will prepare an “unknown” partition that is kept untouched until final evaluation. This subset will include:

- **Sketches from Unused Categories:** If we choose only a subset of the 345 classes for training, any classes we omit can serve as “unknown” categories.  
- **External or Synthetic Data:** We may incorporate sketches from other sources (e.g. or random noise patterns, or create our own samples as vectors) to represent out-of-distribution samples.

The purpose is to test the model’s out-of-bound detection—verifying whether it can correctly identify inputs that do not match any known category.

---

### 6. Sample Exploration
![.nz file contains](sample_dataset/Figure1.png)
![vector stroke representation contains](sample_dataset/Figure2.png)

# Part 3: First Update

## Introduction

This update documents our progress on the Interactive AI Drawing Companion project, summarizing our work so far, presenting preliminary experimental results, and outlining the challenges we have encountered. We have pushed our current codebase to our GitHub repository, which includes data processing scripts, a pipeline for converting Sketch-RNN vector data into raster images, a script to split these images into training, validation, and test sets, and a baseline CNN model trained on the processed images. Our current results show that with a limited dataset per category, our model achieves a Test Loss of approximately 1.39 and a Test Accuracy of around 26%. Although these results are modest, they serve as an essential starting point for further refinement and exploration.

This update describes:
- Our data acquisition and preprocessing process,
- The current state of our baseline model and experimental outcomes,
- Specific challenges we face in terms of generalization, overfitting, real-time inference, and out-of-bound (OOD) detection,
- Our plans for addressing these challenges in future iterations.

## Data Acquisition and Preprocessing

### Dataset Overview

We are using the Sketch-RNN QuickDraw Dataset, which is a preprocessed subset of Google’s Quick, Draw! dataset. The dataset is provided in compressed `.npz` files—one per category (e.g., `cat.npz`, `dog.npz`)—each containing three splits:
- **train:** ~70,000 sketches per category.
- **valid:** ~2,500 sketches per category.
- **test:** ~2,500 sketches per category.

Our initial work focused on processing these files to convert the vector sketches into raster images. The conversion process adheres to the following steps:
1. **Alignment:** The sketches are aligned so that the minimum x and y coordinates are at 0.
2. **Scaling:** The sketches are uniformly scaled to fit within a 256×256 coordinate region.
3. **Resampling:** Strokes are resampled to ensure a consistent spacing (approximately 1 pixel).
4. **Simplification:** The Ramer–Douglas–Peucker (RDP) algorithm is applied with an epsilon value of 2.0 to reduce noise and redundant points.

We successfully generated 100 images per category (from the ‘train’ split) and stored them in a `processed_images` folder, organized by category. To facilitate model training using standard deep learning libraries (e.g., Keras), we then re-split these images into a new dataset directory using a 60/20/20 split (train/valid/test).

### Directory Structure

After splitting, our dataset is organized as follows:
dataset/ train/ cat/ cat_00000.png cat_00001.png ... dog/ dog_00000.png ... valid/ cat/ cat_00060.png ... test/ cat/ cat_00080.png ...

This structure allows us to easily use Keras’ `ImageDataGenerator` for model training and evaluation.

## Baseline Model Development

### CNN Architecture

We built a simple Convolutional Neural Network (CNN) to serve as a baseline for sketch classification. Our architecture consists of the following layers:
- **Input Layer:** Accepts grayscale images resized to 64×64 (downscaled from 256×256 for faster training).
- **Convolutional Layers:** Three Conv2D layers with increasing numbers of filters (32, 64, 128) and ReLU activations.
- **Pooling Layers:** MaxPooling2D layers are interleaved between convolutional layers to reduce spatial dimensions.
- **Fully Connected Layers:** A flattening layer followed by a dense layer with 128 neurons and dropout regularization.
- **Output Layer:** A dense layer with softmax activation to predict the probability distribution over the classes.

### Model Training and Evaluation

We used Keras’ `ImageDataGenerator` to load our dataset from the structured directories. Data augmentation (rotations, shifts, horizontal flips) was applied to the training data to improve generalization. The model was trained for 10 epochs with a batch size of 32 using the Adam optimizer and categorical crossentropy as the loss function.

After training, the model was evaluated on the test set, achieving:
- **Test Loss:** ~1.39
- **Test Accuracy:** ~26%

These results indicate that, while the model is able to learn basic features from the limited dataset, its accuracy is currently quite low. This is expected given the limited number of samples per category and the inherent complexity of hand-drawn sketches.

## Challenges Encountered

### 1. Subject-Disjoint Data Splitting

As professor has highlighted the importance of subject-disjoint splits to improve generalization. The Sketch-RNN dataset metadata not have explicit user IDs, which complicates this task. We are considering the following approaches:
- **Using Proxy Variables:** Leveraging metadata such as `countrycode` and `timestamp` to infer groups of sketches that might originate from the same subject or session.
- **Future Data Collection:** Exploring possibilities for integrating additional metadata in future versions of the dataset to enable true subject-disjoint splits.

### 2. Real-Time Inference Latency

A significant goal of our project is to provide real-time probabilistic feedback as users draw. Although our current work is focused on offline model training, our eventual aim is to integrate the model into a real-time interface. Current challenges include:
- **Preprocessing Overhead:** Converting drawing inputs into a format suitable for the model in real time.
- **Model Inference Time:** Ensuring the CNN model runs quickly enough to provide instant feedback. We are exploring asynchronous processing, model quantization, and lighter network architectures to reduce latency.

### 3. Out-of-Bound (OOD) Detection

Integrating OOD detection into our system is crucial for handling sketches that fall outside the known categories. Although our current model does not yet incorporate OOD detection, we are actively researching several methods:
- **Confidence Thresholding:** Flagging inputs as “unknown” when the softmax probability of the top class falls below a threshold.
- **Advanced Methods:** Exploring techniques such as OpenMax and Monte Carlo dropout to quantify uncertainty.
- **Autoencoder-Based Methods:** Considering training an autoencoder on known sketches and using reconstruction error to detect anomalies.

Balancing these techniques with real-time requirements remains an ongoing challenge.

### 4. User Interface and Interactive Feedback

While our primary focus so far has been on data processing and model training, developing an intuitive user interface is also a priority. Our planned UI will:
- **Display Live Predictions:** Show a dynamically updating probability distribution as users draw.
- **Provide Clear Feedback:** Clearly indicate recognized classes and flag uncertain or unknown inputs.
- **Integrate OOD Detection:** Eventually incorporate OOD indicators to prompt users when a drawing falls outside the known categories.

Designing and testing this UI will require iterative refinement based on user feedback.

## Future Work

Based on our current progress and the challenges encountered, our next steps include:

1. **Subject-Disjoint Splitting:**  
   - Further investigate the use of proxy variables (e.g., countrycode, timestamp) to create subject-disjoint splits.
   - Plan for future dataset enhancements that include explicit subject metadata.
2. **Model Improvements:**  
   - Experiment with deeper or hybrid architectures (e.g., combining CNNs and RNNs) to better capture the sequential nature of stroke data.
   - Continue hyperparameter tuning and incorporate additional regularization techniques.
    
3. **User Interface Development:**  
   - Develop a prototype UI that displays real-time predictions and gathers user feedback.
   - Iteratively refine the UI based on usability testing.
4. **Real-Time Inference Optimization:**  
   - Optimize the preprocessing pipeline to reduce latency.
   - Explore techniques such as model quantization and asynchronous processing to ensure the model can run in real time.

5. **Advanced OOD Detection:**  
   - Prototype multiple OOD detection methods and integrate the most promising approach into the model.
   - Evaluate OOD performance using dedicated metrics like AUROC


# 🖌️Part4

## 📘 Overview

This repository presents a full implementation of an interactive sketch recognition system capable of:

1. **Classifying** hand-drawn sketches from an initial set of 50 categories with high accuracy.  
2. **Detecting unknown** (out-of-distribution) sketches using energy-based scoring and softmax confidence thresholds.  
3. **Learning new classes on the fly** via a one-shot incremental learning pipeline (using exemplar replay and knowledge distillation) without catastrophic forgetting.  

All components—from data preprocessing and training to incremental updates and a Tkinter-based GUI—are provided.


## 🧪 Training & Validation Accuracy

We used a two-layer LSTM (512 hidden units, dropout 0.5) trained on stroke-sequence data from Google’s Quick, Draw! Sketch-RNN dataset, subsampled to 50 classes.

**Dataset Split per class:**
- **Train**: 70,000 samples  
- **Validation**: 2,500 samples  
- **Test**: 2,500 samples  

### Base Model Performance

| Dataset     | Accuracy  |
|-------------|-----------|
| **Train**   | 94.31 %   |
| **Validation** | 92.57 %   |

> 🔍 **Interpretation**:  
> A ~2% train-validation gap indicates mild overfitting. The validation accuracy above 90% shows that the model generalizes well on unseen sketches.

---

##  Incremental Learning Evaluation

Two configurations were tested for incremental learning:

### 🔸 Configuration 1: Naïve
- **Memory**: 20 exemplars per class  
- **New classes per step**: 10  
- **Final test accuracy**: **35.7%**

> ⚠ Severe catastrophic forgetting occurred. The model failed to retain earlier knowledge.

### 🔹 Configuration 2: Improved
- **Memory**: 50 exemplars per class  
- **New classes per step**: 5  
- **Final test accuracy**: **~60.0%**

>  The larger memory and smaller task size helped significantly retain prior knowledge.

---


## 📏 Evaluation Metrics & Observations

### ✅ 1. Classification Accuracy
- **Formula**:  
  ```python
  accuracy = correct_predictions / total_samples
  ```

---

### ✅ 2. Confusion Matrix
- **Why**: To visualize misclassifications and track forgetting patterns.
- **Usage**: Plotted as a heatmap, saved at `plots/confusion_matrix.png`.

---

### ✅ 3. ROC AUC for OOD Detection
- **Why**: Suitable for binary classification (known vs. unknown).
- **AUC Achieved**: **0.87**

---

### 🔧 4. (Planned) Expected Calibration Error (ECE)
- **Purpose**: To assess how well softmax confidence aligns with true accuracy.
- **Status**: To be evaluated in future updates.

---

##  Commentary on Observed Accuracy

- **Base accuracy (92.6%)**: Strong generalization and minimal overfitting.
- **Naïve incremental (35.7%)**: Significant forgetting due to limited memory and large task increments.
- **Improved incremental (60%)**: Substantial recovery with better memory management and task design.
- **OOD detection (AUC = 0.87)**: A strong baseline that can be further improved with advanced scoring techniques.

---

## 💡 Ideas for Further Improvements

### 🔄 1. Contrastive Clustering Loss
- **Inspired by**: Open-world recognition methods.
- **Goal**: Encourage intra-class compactness and inter-class separation in feature space.

---

### 🔁 2. Tune Knowledge Distillation Weight (λ)
- **Strategy**: Increase λ (e.g., 2.0–5.0) to strengthen old model behavior retention.

---

### 🔀 3. Replay Data Augmentation
- **Techniques**: Apply rotation, jitter, or noise to exemplars in memory.
- **Benefit**: Prevents overfitting to static replay samples.



### Once clone, you can run : 
git clone <repo>
cd sketchrnnnpz
pip install -r requirements.txt

# Single validation sample
python single_valid.py --index 0

# Launch interactive GUI
python inference.py

# Full evaluation & plots
python evaluate.py
