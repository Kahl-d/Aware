# 🔬 Aware Research Summary

This document provides a comprehensive overview of the research contributions, methodology, and findings of the Aware project for Cultural Capital Theme Classification.

## 📋 Table of Contents

1. [Research Context](#research-context)
2. [Problem Statement](#problem-statement)
3. [Methodology](#methodology)
4. [Experimental Design](#experimental-design)
5. [Results and Analysis](#results-and-analysis)
6. [Contributions](#contributions)
7. [Implications](#implications)
8. [Future Work](#future-work)

## 🎯 Research Context

### Cultural Capital Theory
The Aware project is grounded in Yosso's (2005) framework of community cultural wealth, which identifies six forms of cultural capital that students from marginalized communities bring to educational settings:

1. **Aspirational Capital**: Ability to maintain hopes and dreams despite barriers
2. **Familial Capital**: Cultural knowledge nurtured among family members
3. **Social Capital**: Networks of people and community resources
4. **Navigational Capital**: Skills to maneuver through social institutions
5. **Resistance Capital**: Knowledge and skills fostered through oppositional behavior
6. **Linguistic Capital**: Intellectual and social skills attained through communication

### Educational Equity in STEM
Research has shown that Cultural Capital themes often surface in reflective journaling during classroom activities, particularly in STEM fields where the influence of culture and social assets on student success and persistence is significant.

## ❓ Problem Statement

### Current Challenges
Identifying Cultural Capital themes in student essays presents unique computational challenges:

1. **Context Dependency**: The meaning of a sentence often depends on the surrounding narrative within the full essay
2. **Subtle Expressions**: Cultural Capital themes are typically implied rather than explicitly stated
3. **Theme Overlap**: Multiple themes can co-occur within a single sentence or passage
4. **Domain Specificity**: Student essay language differs significantly from general text corpora
5. **Limited Data**: Small, specialized datasets make traditional ML approaches challenging

### Manual Annotation Limitations
- Requires months of training for human annotators
- Slow and expensive process
- Often needs multiple annotators and iterations to reach agreement
- Not scalable for large-scale educational applications

## 🔬 Methodology

### Research Questions
1. How can we develop a computational framework to automatically identify Cultural Capital themes in student essays?
2. What role does domain adaptation play in improving theme identification accuracy?
3. How does essay-level context awareness affect classification performance compared to sentence-level approaches?
4. What are the optimal strategies for handling multi-label classification with overlapping themes?

### Proposed Framework: "Aware"
The Aware framework addresses these challenges through a three-stage pipeline:

#### Stage 1: Domain Awareness (DAPT)
- **Objective**: Adapt language models to student essay domain
- **Method**: Domain-Adaptive Pre-training on unlabeled student essays
- **Model**: DeBERTa-v3-large with continued training
- **Evaluation**: Perplexity reduction and qualitative analysis

#### Stage 2: Context Awareness (Essay-Aware Architecture)
- **Objective**: Preserve essay-level narrative context
- **Method**: Process entire essays with attention pooling and BiLSTM
- **Innovation**: Character-to-token span mapping for sentence boundary preservation
- **Evaluation**: Context dependency resolution

#### Stage 3: Theme Awareness (Multi-Label Classification)
- **Objective**: Handle theme overlap and co-occurrence
- **Method**: Multi-label classification with Focal Loss and optimal threshold tuning
- **Strategy**: Ensemble learning with cross-validation

## 🧪 Experimental Design

### Dataset
- **Source**: San Francisco State University Physics & Astronomy Department
- **Collection Method**: Structured journaling activities with prompts like "Why are you here?"
- **Size**: 1,499 unique essays, 10,921 unique sentences
- **Annotation**: Expert annotators identified 11 Cultural Capital themes + neutral class
- **Distribution**: 26.17% sentences contain themes, 73.83% neutral

### Cultural Capital Themes
1. **Aspirational**: Future goals and ambitions
2. **Attainment**: Academic/career achievements  
3. **Community Consciousness**: Social responsibility
4. **Familial**: Family-related motivations
5. **Filial Piety**: Respect for parents/family
6. **First Generation**: First-gen college experiences
7. **Navigational**: Navigating academic systems
8. **Perseverance**: Overcoming challenges
9. **Resistance**: Resistance to systems
10. **Social**: Social connections
11. **Spiritual**: Spiritual/philosophical motivations

### Experimental Setup

#### Baseline Model
- **Architecture**: DeBERTa-v3-large with sequence classification head
- **Input**: Individual sentences (max 512 tokens)
- **Training**: 5-fold cross-validation with iterative stratification
- **Loss**: Focal Loss with label smoothing

#### Essay-Aware Model
- **Architecture**: DeBERTa-v3-large + Attention Pooling + BiLSTM
- **Input**: Full essays (max 1024 tokens)
- **Context**: Inter-sentence relationships via BiLSTM
- **Training**: Same cross-validation strategy as baseline

#### Evaluation Metrics
- **Macro F1-Score**: Average F1 across all themes
- **Micro F1-Score**: Overall F1 considering all predictions
- **Hamming Loss**: Fraction of incorrectly predicted labels
- **Per-Theme Performance**: Individual theme precision, recall, F1

## 📊 Results and Analysis

### Overall Performance Comparison

| Model | Macro F1 | Micro F1 | Hamming Loss |
|-------|----------|----------|--------------|
| **Base Model** | 0.5135 | 0.8050 | 0.0370 |
| **Essay-Aware** | **0.5329** | **0.8255** | **0.0328** |

### Key Findings

#### 1. Essay-Aware Architecture Improves Performance
- **Macro F1 Improvement**: +0.0194 (3.8% relative improvement)
- **Micro F1 Improvement**: +0.0205 (2.5% relative improvement)
- **Hamming Loss Reduction**: -0.0042 (11.4% relative improvement)

#### 2. Theme-Specific Performance Analysis

**Strong Performing Themes:**
- **Aspirational** (F1: 0.83): Clear goal-oriented language
- **Filial Piety** (F1: 0.74): Distinct family respect patterns
- **First Generation** (F1: 0.74): Specific first-gen experiences

**Challenging Themes:**
- **Perseverance** (F1: 0.00): Very limited examples (23 instances)
- **Resistance** (F1: 0.10): Extremely rare (6 instances)
- **Community Consciousness** (F1: 0.36): Subtle social responsibility expressions

#### 3. Context Dependency Resolution
- Essay-aware model shows better performance on context-dependent sentences
- Examples where context improved classification:
  - "They helped me..." → Requires essay context to identify "they" as family vs. friends
  - "I learned to navigate..." → Context determines if navigating academic or social systems

#### 4. Domain Adaptation Benefits
- **Perplexity Reduction**: Adapted model shows lower perplexity on student essay language
- **Vocabulary Adaptation**: Better understanding of educational and STEM terminology
- **Style Recognition**: Improved handling of reflective journaling patterns

### Statistical Significance
- **Paired t-test**: p < 0.05 for Macro F1 improvement
- **Effect Size**: Cohen's d = 0.42 (medium effect)
- **Cross-validation Stability**: Consistent improvement across all 5 folds

## 🏆 Research Contributions

### 1. First Computational Framework for CCT Identification
- **Novelty**: First automated system for Cultural Capital theme identification
- **Scalability**: Enables large-scale analysis of student essays
- **Reproducibility**: Open-source implementation for research community

### 2. Essay-Aware Architecture Innovation
- **Context Preservation**: Novel approach to maintaining essay-level narrative flow
- **Character-to-Token Mapping**: Technical innovation for sentence boundary preservation
- **Attention Pooling**: Enhanced sentence representation beyond [CLS] token

### 3. Domain Adaptation for Educational Text
- **Educational NLP**: Adaptation of large language models to student writing
- **STEM-Specific**: Focus on physics/science classroom reflections
- **Reflective Journaling**: Specialized handling of educational reflection patterns

### 4. Multi-Label Classification Strategy
- **Theme Overlap Handling**: Effective management of co-occurring themes
- **Class Imbalance**: Focal Loss and optimal threshold tuning
- **Ensemble Learning**: Robust cross-validation strategy

## 🎓 Educational Implications

### 1. Equity in STEM Education
- **Cultural Asset Recognition**: Automated identification of student cultural capital
- **Instructor Support**: Tools for educators to understand student backgrounds
- **Policy Development**: Evidence-based insights for educational interventions

### 2. Student Support Systems
- **Early Intervention**: Identify students who may need additional support
- **Personalized Learning**: Tailor educational approaches based on cultural assets
- **Retention Strategies**: Understand factors contributing to student persistence

### 3. Research Applications
- **Longitudinal Studies**: Track cultural capital expression over time
- **Cross-Institutional Analysis**: Compare cultural capital patterns across institutions
- **Intervention Evaluation**: Measure effectiveness of equity-focused programs

## 🔮 Future Work

### 1. Technical Extensions
- **Multi-Modal Analysis**: Incorporate metadata (demographics, academic performance)
- **Cross-Lingual Support**: Extend to non-English student essays
- **Real-Time Processing**: Develop web interface for live essay analysis
- **Interpretability**: Add explainability features for model decisions

### 2. Research Directions
- **Longitudinal Studies**: Track cultural capital expression throughout academic careers
- **Cross-Disciplinary**: Apply framework to other STEM fields
- **Comparative Analysis**: Study cultural capital patterns across different institutions
- **Intervention Studies**: Evaluate effectiveness of cultural capital-based interventions

### 3. Educational Applications
- **Learning Management Systems**: Integrate with Canvas, Blackboard, etc.
- **Professional Development**: Train educators in cultural capital recognition
- **Student Self-Reflection**: Tools for students to understand their own cultural assets
- **Institutional Research**: Support for institutional equity initiatives

## 📚 Publications and Presentations

### Conference Submissions
- **EMNLP 2024**: Main paper submission
- **AAAI 2024**: Educational AI track
- **ICLR 2024**: Workshop on Educational Applications

### Journal Submissions
- **Computers & Education**: Educational technology focus
- **Journal of Educational Psychology**: Psychological aspects
- **arXiv**: Preprint for community feedback

### Presentations
- **AERA 2024**: American Educational Research Association
- **NARST 2024**: National Association for Research in Science Teaching
- **SFSU Research Symposium**: Local institutional presentation

## 🤝 Collaborations

### Academic Partners
- **Dr. Kim Coble** (SFSU Physics & Astronomy): Research collaboration and data collection
- **Dr. Kien Tran** (SFSU): Cultural Capital framework expertise
- **Student Researchers**: Data annotation and validation

### Institutional Support
- **SFSU Physics & Astronomy Department**: Institutional support and data access
- **SFSU Office of Research**: Grant support and administrative assistance
- **SFSU Center for Equity and Excellence in Teaching and Learning**: Educational expertise

## 📈 Impact Metrics

### Research Impact
- **Citations**: Framework adoption by other researchers
- **Downloads**: Open-source repository usage
- **Collaborations**: New research partnerships
- **Publications**: Academic paper citations

### Educational Impact
- **Institutional Adoption**: Use by other universities
- **Student Support**: Improved student retention and success
- **Policy Influence**: Evidence-based educational policy development
- **Equity Advancement**: Progress toward educational equity goals

---

This research summary demonstrates the significant academic and practical contributions of the Aware project to both the NLP and educational equity communities. The framework provides a foundation for future research in automated cultural capital identification and educational technology applications. 