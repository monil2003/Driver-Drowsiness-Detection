# Driver Drowsiness Detection using Deep Learning

A multi-stage research project on **real-time driver drowsiness detection** using facial landmarks, behavioral features, and deep learning models. The work explores both temporal deep-learning approaches and lightweight feature-based classifiers across multiple benchmark datasets.

## Project Overview

The project investigates non-intrusive driver drowsiness detection using camera-based facial and eye-movement information. It progressed through two stages:

1. **Hybrid CNN-LSTM detection** using MediaPipe facial landmarks and behavioral features such as Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and head pose.
2. **Multi-dataset evaluation with ANN and TabNet**, focusing on lightweight feature-based classification and generalization across NTHU-DDD, UTA-RLDD, and YAW-DD.

The research emphasizes practical, real-time detection while reducing the computational overhead associated with processing complete image representations.

## Research Contributions

### Real-Time Driver Drowsiness Detection

The initial study developed a hybrid **CNN-LSTM** approach for real-time drowsiness detection. MediaPipe FaceMesh was used to extract facial landmarks, from which EAR, MAR, and head pose/movement features were derived. CNN layers capture spatial patterns while the LSTM models temporal dependencies such as prolonged eye closure, yawning, and head movement.

The final MediaPipe + EAR + MAR + Head Pose + CNN-LSTM configuration achieved **98.89% accuracy** and a **98.00% F1-score**.

### Multi-Dataset and Low-Resource Evaluation

The extended study evaluated lightweight feature-based models across:

- **NTHU-DDD** — binary classification
- **UTA-RLDD** — ternary classification
- **YAW-DD** — ternary classification

Facial landmarks were extracted using MediaPipe FaceMesh, followed by extraction of pupil coordinates, EAR, and MAR. Feature vectors were standardized before classification using a custom **Artificial Neural Network (ANN)** and **TabNet Classifier**.

The best result was obtained using TabNet on UTA-RLDD: **95.56% accuracy**, with an **AUC of 0.99** and F1-scores above **0.92** across the classes.

## Methodology

```text
Input Images / Video
        |
        v
MediaPipe FaceMesh
        |
        v
Facial Landmark Extraction
        |
        +-- Pupil Coordinates
        +-- Eye Aspect Ratio (EAR)
        +-- Mouth Aspect Ratio (MAR)
        +-- Head Pose / Movement
        |
        v
Feature Scaling / Temporal Processing
        |
        +-- CNN-LSTM
        +-- ANN
        +-- TabNet
        |
        v
Drowsiness Classification
```

MediaPipe FaceMesh extracts **468 facial landmarks**, which are used to construct behavioral features.

## Datasets

### NTHU-DDD
Binary classification:
- Alert
- Drowsy

### UTA-RLDD
Three-class classification:
- Alert
- Low vigilant
- Drowsy

### YAW-DD
Three-class classification:
- Normal
- Talking
- Yawning

## Results

### CNN-LSTM Study

| Metric | Result |
|---|---:|
| Accuracy | **98.89%** |
| F1-score | **98.00%** |

### Multi-Dataset Study

| Dataset | Model | Accuracy |
|---|---|---:|
| NTHU-DDD | ANN | 92.89% |
| NTHU-DDD | TabNet | 88.16% |
| YAW-DD | ANN | 79.85% |
| YAW-DD | TabNet | 77.91% |
| UTA-RLDD | ANN | 91.15% |
| UTA-RLDD | TabNet | **95.56%** |

The extended study found that the YAW-DD **talking** class was difficult to distinguish because its behavioral features can overlap with both alert and drowsy states.

## Repository Structure

```text
BTech_Project/
├── Major Project/
│   ├── 1140_1_Monil_Desai_16th_ICCCNT_2025.pdf
│   ├── 16th_ICCCNT_2025_paper_1140.pdf
│   ├── yawdd_pupil_mar_model.h5
│   └── Models/
│       ├── eye_state_model_nthuddd.h5
│       ├── eye_state_model_utarldd.h5
│       ├── eye_state_model_utarldd_2.h5
│       ├── eye_state_model_utarldd_3_mar.h5
│       ├── eye_state_model_yawdd.h5
│       ├── nthuddd_eye_state_model.h5
│       └── realtime_yawdd.ipynb
│
└── Minor Project/
    ├── Research_Paper.pdf
    ├── Research Conference.pptx
    ├── Acceptance Letter - ICUIS918.pdf
    ├── Presentation_Certificate.pdf
    ├── drowsiness_detection_model.h5
    ├── drowsiness_detection_model2.h5
    ├── FM9B3TC-alarm.mp3
    └── doi
```

## Research Publications

### Initial Study

**Real-Time Driver Drowsiness Detection Using Hybrid CNN-LSTM Model with Facial Feature and Behavioral Analysis**

The initial work presents the MediaPipe + CNN-LSTM approach for real-time drowsiness detection.

### Extended Study

**Multi-Dataset Evaluation of Eye Movement-Based Drowsiness Detection Using ANN and TabNetClassifier**

The extended work evaluates ANN and TabNet across NTHU-DDD, UTA-RLDD, and YAW-DD, with the best reported result of 95.56% accuracy on UTA-RLDD using TabNet.

The repository contains the associated conference papers, acceptance documentation, and presentation certificate.

## Acknowledgements

I would like to acknowledge the contributions of my teammate **Khushal Kathad** to the development and completion of this project. His contributions to the implementation, experimentation, analysis, and research discussions supported the overall project and its associated publications.

I am also grateful to **Prof. Nandini Modi** for her guidance and supervision throughout the research work.

The project was completed collaboratively, with responsibilities distributed across different parts of the implementation, experimentation, analysis, and research workflow.

## Project Timeline

- **Minor Project:** Aug 2024 – Dec 2024
- **Major Project:** Aug 2024 – Jun 2025

## Notes

The repository primarily contains trained model artifacts, an experimentation notebook, research papers, and project documentation. Dataset files are not included in the repository.
