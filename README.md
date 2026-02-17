# Visual Question Answering (VQA)

A Visual Question Answering project implementing and comparing multiple state-of-the-art models: BLIP, BLIP-2, and ViLT.

## Models Implemented

- **BLIP** (Bootstrapping Language-Image Pre-training)
- **BLIP-2** (Bootstrapping Language-Image Pre-training v2)
- **ViLT** (Vision-and-Language Transformer)

## Project Structure

```
VQA2/
├── compare_models.py      # Model comparison and evaluation script
├── run_blip.py           # BLIP model inference
├── run_blip2.py          # BLIP-2 model inference
├── run_vilt.py           # ViLT model inference
├── vqa_utils.py          # Utility functions for VQA tasks
└── README.md
```

## Dataset Setup

This project uses the **VQA v2.0 dataset** (MS COCO). The dataset files are not included in this repository due to their large size.

### Download Instructions

1. **Questions:**
   - Training: [v2_OpenEnded_mscoco_train2014_questions.json](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip)
   - Validation: [v2_OpenEnded_mscoco_val2014_questions.json](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip)

2. **Annotations:**
   - Training: [v2_mscoco_train2014_annotations.json](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip)
   - Validation: [v2_mscoco_val2014_annotations.json](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip)

3. **Images:**
   - Training: [train2014/](http://images.cocodataset.org/zips/train2014.zip) (~13 GB)
   - Validation: [val2014/](http://images.cocodataset.org/zips/val2014.zip) (~6 GB)

### Setup Steps

```bash
# Create project directory structure
cd VQA2

# Download and extract questions
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
unzip v2_Questions_Train_mscoco.zip
unzip v2_Questions_Val_mscoco.zip

# Download and extract annotations
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip
unzip v2_Annotations_Train_mscoco.zip
unzip v2_Annotations_Val_mscoco.zip

# Download and extract images
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
unzip train2014.zip
unzip val2014.zip

# Clean up zip files
rm *.zip
```

After downloading, your directory structure should look like:
```
VQA2/
├── train2014/                                      # Training images
├── val2014/                                        # Validation images
├── v2_OpenEnded_mscoco_train2014_questions.json
├── v2_OpenEnded_mscoco_val2014_questions.json
├── v2_mscoco_train2014_annotations.json
├── v2_mscoco_val2014_annotations.json
├── compare_models.py
├── run_blip.py
├── run_blip2.py
├── run_vilt.py
└── vqa_utils.py
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install torch torchvision
pip install transformers
pip install Pillow
pip install datasets
```

## Usage

### Run Individual Models

```bash
# Run BLIP model
python run_blip.py

# Run BLIP-2 model
python run_blip2.py

# Run ViLT model
python run_vilt.py
```

### Compare Models

```bash
# Run comparison across all models
python compare_models.py
```

## References

- VQA Dataset: [visualqa.org](https://visualqa.org/)
- BLIP: [Salesforce/BLIP](https://github.com/salesforce/BLIP)
- BLIP-2: [Salesforce/LAVIS](https://github.com/salesforce/LAVIS)
- ViLT: [dandelin/ViLT](https://github.com/dandelin/vilt)

## License

This project is for educational purposes.