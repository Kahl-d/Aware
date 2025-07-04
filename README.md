# 🧠 Aware: Cultural Capital Theme Classification

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.20+-green.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-orange.svg)](https://arxiv.org/)

> **Aware**: A multi-stage pipeline for identifying Cultural Capital Themes (CCTs) in student essays using Domain-Adaptive Pre-training, Essay-Aware Architecture, and Multi-Label Classification.

## 📖 Overview

**Aware** is an innovative framework designed to automatically identify Cultural Capital Themes in student essays, particularly within STEM classrooms. Based on Yosso's (2005) framework of community cultural wealth, this system recognizes 11 distinct cultural capital themes that students bring to educational settings.

### 🎯 Problem Statement

Identifying Cultural Capital themes in student essays presents unique challenges:
- **Context Dependency**: Meaning depends on surrounding narrative
- **Subtle Expressions**: Themes are often implied rather than explicit
- **Theme Overlap**: Multiple themes can co-occur in single sentences
- **Domain Specificity**: Student essay language differs from general text

### 🚀 Solution: The "Aware" Framework

Our framework addresses these challenges through three key components:

1. **Domain Awareness (DAPT)**: Domain-Adaptive Pre-training on student essays
2. **Context Awareness**: Essay-aware architecture preserving narrative flow
3. **Theme Awareness**: Multi-label classification handling theme overlap

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Domain-Adapt  │    │  Essay-Aware     │    │ Multi-Label     │
│   Pre-training  │───▶│  Architecture    │───▶│ Classification  │
│   (DAPT)        │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
   Adapts model to         Preserves essay         Handles theme
   student essay           context and             overlap with
   language patterns       narrative flow          multi-label
```

### 🔧 Key Components

#### 1. Domain-Adaptive Pre-training (DAPT)
- **Model**: DeBERTa-v3-large
- **Process**: Continues training on unlabeled student essays
- **Goal**: Adapt to STEM classroom vocabulary and reflective journaling style

#### 2. Essay-Aware Architecture
- **Essay Reconstruction**: Groups sentences into full essays (max 1024 tokens)
- **Attention Pooling**: Learns weighted token representations
- **BiLSTM Context**: Captures inter-sentence relationships
- **Character-to-Token Mapping**: Preserves sentence boundaries

#### 3. Multi-Label Classification
- **11 Cultural Capital Themes** + neutral class
- **Focal Loss**: Handles class imbalance
- **Optimal Threshold Tuning**: Per-category optimization
- **Ensemble Learning**: 5-fold cross-validation

## 📊 Cultural Capital Themes

| Theme | Description | Example |
|-------|-------------|---------|
| **Aspirational** | Future goals and ambitions | "I want to become a doctor" |
| **Attainment** | Academic/career achievements | "I earned my degree" |
| **Community Consciousness** | Social responsibility | "I want to help my community" |
| **Familial** | Family-related motivations | "My parents sacrificed for me" |
| **Filial Piety** | Respect for parents/family | "I must make my family proud" |
| **First Generation** | First-gen college experiences | "I'm the first in my family" |
| **Navigational** | Navigating academic systems | "I learned to navigate college" |
| **Perseverance** | Overcoming challenges | "Despite obstacles, I persisted" |
| **Resistance** | Resistance to systems | "I challenge traditional norms" |
| **Social** | Social connections | "My friends supported me" |
| **Spiritual** | Spiritual/philosophical | "I believe everything happens for a reason" |

## 📈 Performance Results

| Model | Macro F1 | Micro F1 | Hamming Loss |
|-------|----------|----------|--------------|
| **Base Model** | 0.5135 | 0.8050 | 0.0370 |
| **Essay-Aware** | **0.5329** | **0.8255** | **0.0328** |

### Detailed Performance by Theme

| Theme | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Aspirational | 0.79 | 0.86 | 0.83 | 245 |
| Attainment | 0.61 | 0.57 | 0.59 | 171 |
| Community Consciousness | 0.44 | 0.31 | 0.36 | 13 |
| Familial | 0.52 | 0.84 | 0.64 | 50 |
| Filial Piety | 0.70 | 0.78 | 0.74 | 55 |
| First Generation | 0.76 | 0.73 | 0.74 | 22 |
| Navigational | 0.44 | 0.50 | 0.47 | 94 |
| Perseverance | 0.00 | 0.00 | 0.00 | 23 |
| Resistance | 0.07 | 0.17 | 0.10 | 6 |
| Social | 0.50 | 0.46 | 0.48 | 41 |
| Spiritual | 0.44 | 0.66 | 0.53 | 35 |

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/aware.git
cd aware

# Create virtual environment
python -m venv aware_env
source aware_env/bin/activate  # On Windows: aware_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📁 Project Structure

```
aware/
├── 📄 README.md                 # This file
├── 📄 requirements.txt          # Python dependencies
├── 📄 LICENSE                   # MIT License
├── 📄 .gitignore               # Git ignore file
├── 📄 Aware-Paper.docx         # Research paper
│
├── 🔧 multi-base/              # Base model implementation
│   ├── base_model.py           # Standard sentence-level classification
│   ├── train.csv               # Training data
│   ├── test.csv                # Test data
│   ├── run_job.sh              # SLURM job script
│   └── proto/                  # Model outputs
│
├── 🧠 multi-ea/                # Essay-aware implementation
│   ├── essay_aware_model.py    # Essay-aware architecture
│   ├── train_essay_aware.csv   # Training data
│   ├── test_essay_aware.csv    # Test data
│   ├── run_job.sh              # SLURM job script
│   └── models_essay_aware/     # Model outputs
│
├── 🔄 dapt/                    # Domain-adaptive pre-training
│   ├── train.py                # DAPT training script
│   ├── test.py                 # DAPT evaluation script
│   ├── combined_essays.csv     # Essay corpus for DAPT
│   ├── run_job.sh              # SLURM job script
│   └── deberta-v3-large-essays-adapted-final/  # Adapted model
│
├── 📊 data/                    # Dataset files
│   ├── Attainment_essayaware_train.csv
│   └── Attainment_essayaware_test.csv
│
└── 📝 logs/                    # Training logs
    ├── training_log_essay_aware.log
    └── training_proto.log
```

## 🚀 Usage

### 1. Domain-Adaptive Pre-training
```bash
cd dapt
python train.py
```

### 2. Base Model Training
```bash
cd multi-base
python base_model.py
```

### 3. Essay-Aware Model Training
```bash
cd multi-ea
python essay_aware_model.py
```

### 4. Model Evaluation
```bash
cd dapt
python test.py
```

## 📊 Dataset

### Overview
- **1,499 unique essays** from STEM classrooms
- **10,921 unique sentences** (average 7 sentences per essay)
- **2,858 sentences (26.17%)** contain at least one CCT
- **8,063 sentences (73.83%)** are neutral (class_0)

### Data Collection
- **Source**: San Francisco State University Physics & Astronomy Department
- **Method**: Structured journaling activities with prompts like "Why are you here?"
- **Processing**: Anonymized, cleaned, and restructured for computational modeling

## 🔬 Research Impact

### Educational Applications
- **Scalable CCT identification**: Automates expert-level annotation
- **Equity insights**: Helps educators understand student cultural assets
- **STEM education**: Supports underrepresented students
- **Policy development**: Evidence-based educational interventions

### Technical Contributions
- **First computational framework** for Cultural Capital Theme identification
- **Novel essay-aware architecture** preserving narrative context
- **Domain adaptation approach** for educational text
- **Multi-label classification strategy** for overlapping themes

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@article{aware2024,
  title={Aware: Cultural Capital Theme Classification in Student Essays},
  author={Khan, Khalid and Coble, Kim and Tran, Kien},
  journal={arXiv preprint},
  year={2024}
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dr. Kim Coble** (San Francisco State University) - Research collaboration
- **Dr. Kien Tran** - Cultural Capital framework expertise
- **Student Researchers** - Data collection and annotation
- **SFSU Physics & Astronomy Department** - Institutional support

## 📞 Contact

- **Author**: Khalid Khan
- **Email**: kkhan@sfsu.edu
- **Institution**: San Francisco State University
- **Research Area**: Educational Technology, NLP, Equity in STEM

---

<div align="center">
  <p><strong>Empowering educators to recognize and value student cultural capital through AI</strong></p>
  <p>⭐ Star this repository if you find it useful!</p>
</div> 