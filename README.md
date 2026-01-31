# Intrapartum Ultrasound Grand Challenge (IUGC) 2024

This repository contains the code and tools for the MICCAI 2024 Intrapartum Ultrasound Grand Challenge. The challenge focuses on ultrasound image analysis for intrapartum applications, including classification and segmentation tasks.

## Project Structure

```
IUGC2024-main/
├── baseline/
│   ├── starting_tool_kit/
│   │   ├── utils/
│   │   │   ├── augmentation.py
│   │   │   ├── criterion.py
│   │   │   ├── dataset_classification.py
│   │   │   ├── dataset_segmentation.py
│   │   │   ├── evaluator.py
│   │   │   ├── loss.py
│   │   │   └── metric.py
│   │   ├── README.md
│   │   ├── requirements.txt
│   │   └── train_seg.py
│   ├── utils/
│   │   ├── augmentation.py
│   │   ├── criterion.py
│   │   ├── dataset_classification.py
│   │   ├── dataset_segmentation.py
│   │   ├── evaluator.py
│   │   ├── loss.py
│   │   └── metric.py
│   ├── model.py
│   ├── resize_video.py
│   ├── train_cls.py
│   └── train_seg.py
├── eval/
│   ├── cal_aop.py
│   └── ellipse.py
├── latex_template/
│   └── main.tex
├── model_encapsulation/
│   └── baseline.py
├── starting_tool_kit/
│   ├── utils/
│   │   ├── augmentation.py
│   │   ├── criterion.py
│   │   ├── dataset_classification.py
│   │   ├── dataset_segmentation.py
│   │   ├── evaluator.py
│   │   ├── loss.py
│   │   └── metric.py
│   ├── README.md
│   ├── dataset_sample/
│   ├── requirements.txt
│   └── train_seg.py
└── submit_code/
    ├── metadata
    ├── model.pickle
    └── model.py
```

## Implementation Instructions

### Prerequisites

First, install the required dependencies:

```bash
pip install -r starting_tool_kit/requirements.txt
```

The requirements include:
- numpy
- opencv-python
- SimpleITK
- scikit-learn

### Getting Started

The challenge provides a starting toolkit in the `starting_tool_kit` directory. This includes:

- **Dataset Sample**: Contains positive and negative samples for training in the `dataset_sample` folder.
- **Utilities**: Various helper functions in the `utils` folder:
  - `augmentation.py`: Classes for data augmentation in classification and segmentation tasks
  - `criterion.py`: Customizable combinations of loss functions for training
  - `dataset_classification.py`: Class to read data for classification tasks
  - `dataset_segmentation.py`: Class to read labeled data for segmentation tasks
  - `evaluator.py`: Classes to evaluate model performance with comprehensive assessments
  - `loss.py`: Common loss functions
  - `metric.py`: Metric class for validation results during training

### Training Models

#### Segmentation Model

Use the provided `train_seg.py` template to train segmentation models:

```bash
cd starting_tool_kit
python train_seg.py
```

The training script includes:
- Hyperparameters configuration (batch size, epochs, learning rate, image size)
- Model loading and optimizer setup
- Training loop with validation
- Checkpoint saving based on validation scores

#### Classification Model

Use the `train_cls.py` in the `baseline` directory for classification tasks:

```bash
cd baseline
python train_cls.py
```

### Evaluation

The evaluation scripts are located in the `eval/` directory:
- `cal_aop.py`: Calculates AOP (Angle of Progression) metrics
- `ellipse.py`: Ellipse fitting utilities

### Model Submission

For submission, place your trained model in the `submit_code/` directory:
- `model.py`: Your model implementation
- `model.pickle`: Serialized model weights
- `metadata`: Metadata file for your submission

### Key Features

- **Multi-task Learning**: The challenge supports simultaneous training for both classification and segmentation tasks
- **Data Augmentation**: Comprehensive augmentation techniques for improved model generalization
- **Flexible Loss Functions**: Customizable combinations of various loss functions
- **Comprehensive Evaluation**: Multiple metrics and evaluation strategies for robust model assessment

### Customization

You can customize:
- Network architectures in `model.py`
- Training hyperparameters in the training scripts
- Data augmentation strategies in `utils/augmentation.py`
- Loss functions in `utils/criterion.py`
- Evaluation metrics in `utils/metric.py`

## License

This project is part of the MICCAI 2024 Intrapartum Ultrasound Grand Challenge.