

````markdown
# 🥔 Potato Leaf Late Blight Detection, Severity Analysis & Disease Progression Monitoring

> **AI-Based Computer Vision Project | Under Development**

An end-to-end computer vision project focused on detecting **late blight disease in potato leaves**, localizing diseased regions, estimating disease severity, and eventually monitoring disease progression over time.

The project is being developed as a research-oriented AI/ML pipeline that progresses from **image-based disease detection** toward **lesion localization, severity estimation, temporal monitoring, and future disease-risk analysis**.

---

## 📌 Project Overview

Potato late blight is a major disease affecting potato crops. Early identification of infected regions is important for monitoring disease development and supporting timely crop-management decisions.

The goal of this project is to develop a computer vision system that can automatically analyze potato leaf images and provide quantitative information about disease presence and progression.

The planned pipeline is:

```text
Potato Leaf Image
       │
       ▼
Disease Detection
       │
       ▼
Lesion Localization
       │
       ▼
Disease Severity Estimation
       │
       ▼
Multi-Date Disease Monitoring
       │
       ▼
Disease Progression Analysis
       │
       ▼
Environmental Risk Analysis
       │
       ▼
Future Forecasting
````

---

# 🎯 Objectives

The project is being developed around the following objectives.

### 1. Late Blight Detection

Develop a deep-learning model capable of detecting visible late-blight regions in potato leaf images.

### 2. Lesion Localization

Locate visible diseased regions using object-detection bounding boxes.

### 3. Disease Severity Estimation

Estimate the percentage of the leaf affected by disease using lesion segmentation and pixel-level analysis.

### 4. Disease Progression Monitoring

Track changes in disease severity across multiple observations or dates.

### 5. Environmental Risk Analysis

In future stages, investigate the relationship between disease development and environmental variables such as:

* Temperature
* Relative humidity
* Rainfall
* Leaf wetness
* Observation date/time

### 6. Future Forecasting

Explore machine-learning/time-series approaches for predicting disease-risk or progression trends when sufficient temporal and environmental data are available.

---

# 🧠 Project Architecture

The complete planned system is divided into several stages:

```text
                    ┌─────────────────────┐
                    │  Potato Leaf Image  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  YOLO Detection     │
                    │  Late Blight        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Lesion Localization │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Segmentation        │
                    │ Diseased Pixels     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Severity Estimation │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Multi-Date Tracking │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Progression/Risk    │
                    │ Analysis             │
                    └─────────────────────┘
```

---

# 🛠️ Technology Stack

## Programming

* Python

## Computer Vision

* OpenCV
* Pillow

## Data Analysis

* NumPy
* Pandas
* Matplotlib
* Scikit-learn

## Deep Learning

* Ultralytics YOLO
* PyTorch

## Annotation

* CVAT (Computer Vision Annotation Tool)

## Development

* Jupyter Notebook
* Visual Studio Code
* Git
* GitHub

---

# 📂 Project Structure

The project is being organized as follows:

```text
potato-leaf-vision/
│
├── data/
│   ├── raw/
│   │   ├── healthy/
│   │   ├── early_blight/
│   │   └── late_blight/
│   │
│   ├── train/
│   ├── val/
│   └── test/
│
├── yolo_dataset/
│   ├── data.yaml
│   ├── train.txt
│   └── labels/
│       └── train/
│
├── notebooks/
│   └── 01_dataset_analysis.ipynb
│
├── src/
│   ├── preprocessing/
│   ├── detection/
│   ├── segmentation/
│   ├── severity/
│   └── monitoring/
│
├── models/
│
├── results/
│   ├── figures/
│   ├── predictions/
│   └── metrics/
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

> Some directories represent planned future components and may not contain implementation yet.

---

# 📊 Dataset

The initial dataset contains potato leaf images belonging to three broad categories:

```text
Healthy
Early Blight
Late Blight
```

The dataset was organized into:

```text
data/raw/
│
├── healthy/
├── early_blight/
└── late_blight/
```

The project currently focuses specifically on **late-blight detection**.

---

# 🔍 Dataset Preparation

The initial dataset-preparation stage includes:

* Organizing images by disease class
* Checking image dimensions
* Visualizing representative samples
* Checking for corrupted images
* Checking for duplicate images
* Inspecting class distribution
* Preparing training/validation/test splits

The analysis is being performed using Python, Pillow, Pandas, Matplotlib, and Scikit-learn.

---

# 🏷️ Object Detection Annotation

For the object-detection stage, **CVAT** is being used to manually annotate visible late-blight regions.

A dedicated CVAT project was created:

```text
potato_late_blight_detection
```

The annotation configuration is:

```text
Label:
late_blight

Shape:
Rectangle
```

The rectangle annotations represent visible late-blight affected regions.

---

# 📸 Current Annotation Dataset

A subset of **95 late-blight images** has been selected for the object-detection annotation stage.

Current progress:

```text
Total selected images       : 95
Manually annotated images   : 12
Remaining images            : 83
```

The current 12 annotated images are being used to create an initial detection dataset.

> The initial 12-image dataset is a prototype training set. It is not considered sufficient for a final robust model.

---

# 🔄 Annotation Strategy

Instead of manually drawing bounding boxes on all 95 images from the beginning, the project follows an iterative annotation strategy.

```text
             12 Manual Annotations
                      │
                      ▼
              Initial YOLO Model
                      │
                      ▼
            Predictions on New Images
                      │
                      ▼
              Human Verification
                      │
                      ▼
             Corrected Annotations
                      │
                      ▼
              Expanded Dataset
                      │
                      ▼
                Retrain YOLO
```

This follows an iterative object-detection workflow in which labeled data is used to train an initial model, predictions are generated on new images, difficult cases are identified, and additional corrected examples are incorporated into later training cycles. ([Ultralytics Docs][1])

---

# 📦 YOLO Dataset Export

The annotations were exported from CVAT using:

```text
Ultralytics YOLO Detection 1.0
```

This format is appropriate because the current annotations are **bounding boxes** rather than segmentation masks or classification-only labels. CVAT's Ultralytics YOLO Detection export uses normalized bounding-box coordinates and provides the dataset configuration and image/label structure required for YOLO detection. ([CVAT Documentation][2])

The current exported annotation package contains:

```text
yolo_dataset/
│
├── data.yaml
├── train.txt
└── labels/
    └── train/
```

The corresponding image files still need to be placed/linked correctly with the exported labels before training.

A complete Ultralytics YOLO dataset uses images together with their corresponding annotation `.txt` files and a dataset YAML configuration. ([CVAT Documentation][2])

---

# 📝 YOLO Annotation Format

Each annotated image has a corresponding YOLO annotation file.

The detection annotation follows the general format:

```text
class_id center_x center_y width height
```

The coordinates are normalized relative to the image dimensions. ([CVAT Documentation][2])

For example:

```text
0 0.52 0.47 0.18 0.21
```

where:

```text
0     → late_blight class ID
0.52  → bounding-box center X
0.47  → bounding-box center Y
0.18  → bounding-box width
0.21  → bounding-box height
```

---

# 🤖 YOLO Detection Model

The next major implementation stage is training a YOLO object-detection model on the manually annotated late-blight images.

The project will use a pretrained YOLO model as the starting point and fine-tune it on the custom dataset.

Ultralytics documents pretrained-model training for custom detection datasets using a dataset YAML configuration. ([Ultralytics Docs][3])

Conceptually:

```text
Pretrained YOLO
       │
       ▼
Custom Late-Blight Dataset
       │
       ▼
Fine-Tuning
       │
       ▼
Late-Blight Detection Model
```

---

# 🚧 Current Development Status

| Component                      | Status         |
| ------------------------------ | -------------- |
| Project initialization         | ✅ Completed    |
| Python environment             | ✅ Completed    |
| Dataset collection             | ✅ Completed    |
| Dataset organization           | ✅ Completed    |
| Basic dataset analysis         | 🔄 In Progress |
| CVAT project creation          | ✅ Completed    |
| Late-blight label creation     | ✅ Completed    |
| Rectangle annotation setup     | ✅ Completed    |
| 95-image late-blight subset    | ✅ Completed    |
| Manual annotation              | 🔄 In Progress |
| 12 images annotated            | ✅ Completed    |
| CVAT YOLO export               | ✅ Completed    |
| YOLO dataset preparation       | 🔄 In Progress |
| Initial YOLO training          | ⏳ Next         |
| Prediction on remaining images | ⏳ Planned      |
| Annotation correction          | ⏳ Planned      |
| Dataset expansion              | ⏳ Planned      |
| Segmentation                   | ⏳ Planned      |
| Severity estimation            | ⏳ Planned      |
| Temporal monitoring            | ⏳ Planned      |
| Environmental analysis         | ⏳ Planned      |
| Forecasting                    | ⏳ Planned      |
| Dashboard                      | ⏳ Planned      |

---

# 🧪 Model Training

The first model will be considered an **initial prototype** because the currently annotated dataset is small.

The training pipeline will follow:

```text
YOLO Dataset
     │
     ▼
Training Images
     │
     ▼
Pretrained YOLO
     │
     ▼
Fine-Tuning
     │
     ▼
Validation
     │
     ▼
Best Model
```

The Ultralytics training workflow supports specifying a custom dataset YAML and training parameters such as epochs and image size. ([Ultralytics Docs][4])

Example training command:

```bash
yolo detect train model=yolo26n.pt data=yolo_dataset/data.yaml epochs=50 imgsz=640
```

> The exact model and training parameters may be adjusted after inspecting the available hardware and dataset size.

---

# 🔎 Model Prediction

After the first model is trained, it will be used to generate predictions on the remaining unannotated late-blight images.

The workflow will be:

```text
Unseen Image
     │
     ▼
YOLO Model
     │
     ▼
Predicted Bounding Boxes
     │
     ▼
Human Verification
     │
     ├── Correct Prediction
     │
     └── Correct / Add / Remove Box
```

The corrected predictions can then become additional training data.

---

# 📈 Detection Evaluation

The detection model will be evaluated using standard object-detection metrics such as:

### Precision

Measures the proportion of predicted detections that are correct.

### Recall

Measures how many relevant disease regions were detected.

### mAP@50

Mean Average Precision at IoU threshold 0.50.

### mAP@50–95

Mean Average Precision averaged across multiple IoU thresholds.

### F1 Score

Harmonic mean of precision and recall.

Ultralytics validation provides metrics including mAP50, mAP50-95, precision, recall, and related per-image detection metrics. ([Ultralytics Docs][4])

---

# 🧩 Planned Segmentation Stage

Bounding boxes provide approximate localization.

For more precise disease quantification, the next stage will investigate **lesion segmentation**.

The planned pipeline is:

```text
Leaf Image
    │
    ▼
Late-Blight Detection
    │
    ▼
Lesion Region
    │
    ▼
Segmentation
    │
    ▼
Disease Pixels
```

This will allow the project to estimate the actual affected area more precisely than bounding boxes alone.

---

# 📐 Disease Severity Estimation

After obtaining the diseased region and leaf region, the planned severity calculation is:

```text
Disease Severity (%) =
--------------------------------------- × 100
       Diseased Leaf Area
```

More explicitly:

```text
Disease Severity (%) =
(Diseased Pixel Area / Total Leaf Pixel Area) × 100
```

This value can then be used to represent the proportion of the leaf affected by visible disease.

The exact severity thresholds/categories will be defined separately and will not be treated as biological standards without appropriate validation.

---

# 📅 Disease Progression Monitoring

A future stage of the project will analyze disease progression across multiple observations.

Example:

```text
Day 1
   │
   ▼
Day 4
   │
   ▼
Day 7
   │
   ▼
Day 10
```

For each observation, the system can record:

```text
Date
Disease Area
Disease Severity
Number of Lesions
Detection Confidence
Environmental Conditions
```

The resulting data can be visualized as:

```text
Disease Severity
      │
      │                ●
      │          ●
      │      ●
      │  ●
      └──────────────────────► Time
```

This will allow disease progression to be studied quantitatively.

---

# 🌦️ Environmental Data Integration

A future version may combine image-based disease measurements with environmental variables.

Potential variables include:

```text
Temperature
Humidity
Rainfall
Leaf Wetness
Date / Time
Location
```

The goal is to investigate whether changes in environmental conditions correspond with changes in observed disease progression.

---

# 🔮 Future Forecasting

If sufficient temporal data becomes available, the project may explore machine-learning and time-series approaches for disease-risk or progression analysis.

Possible approaches include:

* Linear Regression baseline
* Random Forest
* Gradient Boosting
* LSTM
* GRU
* Temporal Neural Networks

The forecasting stage will only be developed after obtaining an adequate time-series dataset.

---

# 📊 Planned Dashboard

A future dashboard will provide an interactive view of the model output.

Potential dashboard components:

```text
┌─────────────────────────────────────┐
│     POTATO LATE BLIGHT MONITOR      │
├─────────────────────────────────────┤
│                                     │
│  Leaf Image                         │
│  ┌─────────────────────────────┐    │
│  │   Disease Detection         │    │
│  │   [Bounding Boxes]          │    │
│  └─────────────────────────────┘    │
│                                     │
│  Disease Severity: XX %             │
│  Detection Confidence: XX %         │
│                                     │
│  Progression                        │
│  ────────────────╱────────          │
│                                     │
│  Environmental Conditions           │
│  Temperature: XX °C                 │
│  Humidity: XX %                     │
│  Rainfall: XX mm                    │
│                                     │
└─────────────────────────────────────┘
```

---

# 🔬 Research-Oriented Extensions

The longer-term version of the project may investigate:

* Multimodal disease detection
* UAV/drone imagery
* Hyperspectral or thermal imagery
* Environmental sensing
* Time-series disease modelling
* Explainable AI
* Edge AI
* Disease-risk forecasting
* Model monitoring and retraining

These extensions are future research directions rather than currently implemented components.

---

# 🧠 Explainability

An explainability stage is planned to help understand which parts of the image influence model predictions.

Possible future methods include:

* Grad-CAM
* Saliency maps
* Detection visualization
* Confidence analysis

The goal is to make the model's visual predictions easier to inspect.

---

# 🔁 Model Improvement Strategy

The project follows an iterative development loop:

```text
Collect Data
     ↓
Annotate
     ↓
Train
     ↓
Validate
     ↓
Find Errors
     ↓
Add / Correct Data
     ↓
Retrain
     ↓
Validate Again
```

This is particularly important because the initial dataset is small.

Future datasets should ideally contain variation in:

* Lighting
* Background
* Camera/device
* Leaf orientation
* Disease stage
* Cultivar
* Field conditions
* Image quality

This will help evaluate whether the model generalizes beyond the initial image set.

---

# ⚠️ Current Limitations

The current project has several limitations.

### Small Initial Detection Dataset

Only 12 images have currently been manually annotated.

Therefore, the first YOLO model should be considered a prototype.

### Limited Annotation Coverage

83 of the selected 95 late-blight images are not yet manually annotated.

### Image-Domain Limitations

Initial images may not represent all real-world field conditions.

### No Temporal Dataset Yet

Disease progression analysis requires multiple observations over time.

### No Environmental Dataset Yet

Environmental-risk analysis requires corresponding environmental measurements.

### No Final Severity Model Yet

Severity estimation is a planned stage and has not yet been completed.

---

# 📌 Current Milestone

### Completed

```text
✅ Project created
✅ Python environment configured
✅ Dataset organized
✅ Potato disease classes identified
✅ Late-blight subset selected
✅ 95 images selected for detection work
✅ CVAT project created
✅ late_blight label created
✅ Rectangle annotation configured
✅ 12 images manually annotated
✅ Ultralytics YOLO Detection export completed
```

### Currently Working On

```text
🔄 Preparing YOLO dataset
🔄 Training first YOLO prototype
```

### Next

```text
⬜ Generate predictions on remaining images
⬜ Correct predictions
⬜ Expand annotation dataset
⬜ Retrain YOLO
⬜ Evaluate detection performance
```

---

# 🚀 Complete Planned Roadmap

```text
PHASE 1
Dataset Preparation
        ↓
PHASE 2
Exploratory Data Analysis
        ↓
PHASE 3
CVAT Annotation
        ↓
PHASE 4
YOLO Object Detection
        ↓
PHASE 5
Automatic Prediction + Human Correction
        ↓
PHASE 6
Dataset Expansion + Retraining
        ↓
PHASE 7
Lesion Segmentation
        ↓
PHASE 8
Disease Severity Estimation
        ↓
PHASE 9
Multi-Date Disease Monitoring
        ↓
PHASE 10
Environmental Risk Analysis
        ↓
PHASE 11
Disease Progression / Forecasting
        ↓
PHASE 12
Dashboard & Deployment
```

---

# 📚 Research Motivation

The project is motivated by research into the use of artificial intelligence for potato late-blight detection, physiological/molecular monitoring, and disease forecasting.

The broader research direction is to move beyond simple disease classification toward an integrated system combining:

```text
Computer Vision
       +
Disease Localization
       +
Severity Quantification
       +
Temporal Monitoring
       +
Environmental Data
       +
Forecasting
```

---

# 👩‍💻 Development

This project is being developed as an AI/ML and computer-vision research/portfolio project.

**Current focus:** Potato leaf late-blight object detection using YOLO.

**Status:** 🚧 Under Development

---

# 📄 License

This project is intended for educational, research, and portfolio purposes.

Dataset licensing and third-party model/software licenses should be checked separately before redistribution or commercial use.

Ultralytics provides different licensing options for its software/models; users should review the applicable Ultralytics license before using the technology in a particular project or product. ([Ultralytics Docs][1])

---

# ⭐ Project Pipeline Summary

```text
                    POTATO LEAF
                         │
                         ▼
                ┌────────────────┐
                │ Disease        │
                │ Detection      │
                └───────┬────────┘
                        │
                        ▼
                ┌────────────────┐
                │ Lesion         │
                │ Localization   │
                └───────┬────────┘
                        │
                        ▼
                ┌────────────────┐
                │ Segmentation   │
                └───────┬────────┘
                        │
                        ▼
                ┌────────────────┐
                │ Severity       │
                │ Estimation     │
                └───────┬────────┘
                        │
                        ▼
                ┌────────────────┐
                │ Progression    │
                │ Monitoring     │
                └───────┬────────┘
                        │
                        ▼
                ┌────────────────┐
                │ Risk Analysis  │
                └───────┬────────┘
                        │
                        ▼
                ┌────────────────┐
                │ Forecasting    │
                └────────────────┘
```

> **From detecting a diseased leaf to quantitatively monitoring how the disease progresses over time.**


[1]: https://docs.ultralytics.com/yolov5/tutorials/train-custom-data?utm_source=chatgpt.com "Train YOLOv5 on Custom Data | Ultralytics"
[2]: https://docs.cvat.ai/docs/dataset_management/formats/format-yolo-ultralytics/?utm_source=chatgpt.com "Ultralytics YOLO | CVAT"
[3]: https://docs.ultralytics.com/datasets/detect?utm_source=chatgpt.com "Object Detection Datasets Overview | Ultralytics"
[4]: https://docs.ultralytics.com/tasks/detect?utm_source=chatgpt.com "Object Detection with Ultralytics YOLO"
