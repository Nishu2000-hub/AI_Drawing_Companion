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

## Conclusion

This project seeks to blend advanced neural network techniques with interactive design principles to create a unique and engaging drawing game. By leveraging the extensive Quick, Draw! dataset and incorporating advanced out-of-bound detection methods—such as confidence thresholding, uncertainty estimation, the system is designed to handle both typical and novel inputs robustly. As the project evolves, additional refinements and potential multimodal integrations may be considered to further enhance both the technical performance and the overall user experience.

