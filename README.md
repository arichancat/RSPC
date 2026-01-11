# Relational Stress and Psychiatry Corpus (RSPC)

The **Relational Stress and Psychiatry Corpus (RSPC)** is a clinically grounded, relationship-aware mental-health dataset designed to study psychological distress as an interactional and temporal phenomenon. Unlike prior datasets that treat mental health as an individual-level signal, RSPC explicitly models **relational stressors**, **psychiatric symptom categories**, and **relationship phase dynamics** within long-distance romantic relationships (LDRs).

RSPC enables the study of how interpersonal uncertainty, communication breakdowns, and digital interaction patterns contribute to psychiatric symptom expression over time.

---

## 📊 Dataset Overview

RSPC consists of **1,799 Reddit posts** collected from long-distance relationship discussion forums. Each post is manually annotated along three complementary axes:

1. **Psychiatric Symptom Categories**  
   Multi-label annotations grounded in **DSM-5-TR** and **ICD-11**, allowing comorbid symptom modeling.

2. **Relational Stress Triggers**  
   Relationship-specific stressors such as commitment ambiguity, communication gaps, and reunion-related stress.

3. **Temporal Relationship Phase**  
   Coarse-grained phase labels capturing progression within the long-distance relationship lifecycle.

This tri-axial annotation enables joint reasoning over clinical symptoms, relational causes, and temporal context.

---

## 🧠 Annotation Schema

### Psychiatric Symptom Categories (Task 1)
Multi-label classification over five categories:

- Adjustment Disorder (ADJ)  
- Generalized Anxiety Disorder (GAD)  
- Social Anxiety Disorder (SAD)  
- Major Depressive Disorder (MDD)  
- Insomnia

### Relational Stress Triggers (Task 2)
Multi-label trigger detection over eight categories:

- Commitment Ambiguity  
- Lack of Communication  
- Reunion / Separation Stress  
- Trust and Fidelity Concerns  
- Jealousy and Insecurity  
- Silence or Communication Gaps  
- Social Media Surveillance  
- Time-Zone Mismatch

### Temporal Relationship Phases (Task 3)
Single-label classification:

- Anticipation  
- Separation  
- Reunion  
- Unknown

---

## 🧪 Benchmark Tasks

RSPC supports three supervised learning tasks:

**Task 1 — Multi-Label Psychiatric Symptom Classification**  
Predict psychiatric symptom categories from text, explicitly modeling comorbidity and symptom overlap.

**Task 2 — Relational Trigger Detection**  
Identify relationship-specific stressors that act as precursors or amplifiers of psychological distress.

**Task 3 — Temporal Phase Prediction**  
Predict the relationship phase associated with a post, capturing temporal dynamics of long-distance relationships.

---

## 📈 Baselines and Experiments

Initial benchmark experiments include:

- TF-IDF baselines  
- Transformer-based models (BERT, RoBERTa, ClinicalBERT)  

Current results highlight strong performance on frequent categories, while **clinically specific and minority labels remain challenging**, motivating further research into context-aware and temporally informed modeling.

Ongoing work includes:
- Multi-seed evaluation for statistical robustness  
- Fine-tuning of state-of-the-art transformer models  
- Expanded experimentation for Task 2 and Task 3  
- Explainability and error analysis

---

## ⚖️ Ethical Considerations

To ensure ethical compliance and user privacy:

- **All usernames and direct identifiers have been removed**
- Posts are anonymized and stored only as textual content
- No attempt is made to re-identify individuals
- The dataset is intended **strictly for non-clinical, research purposes**

⚠️ *Note:* This section will be finalized and refined prior to public dataset release.

---

## 📄 Intended Use

RSPC is designed for:

- Computational mental-health research  
- Social and affective computing  
- Clinical NLP benchmarking  
- Relationship-aware stress modeling  

The dataset **must not** be used for diagnosis, treatment, or clinical decision-making.

---

## 📌 Release Status

- Dataset annotations: **Complete**
- Task 1 benchmarks: **Available**
- Task 2 & Task 3 benchmarks: **In progress**
- Multi-seed evaluation: **Ongoing**
- Full dataset release: **Planned upon paper acceptance**

---

## 📚 Citation

If you use RSPC in your research, please cite the accompanying paper (citation to be added upon acceptance).

---

## 📬 Contact

For questions, collaboration, or access requests, please contact the authors via the repository issue tracker.

---

**RSPC — Modeling Mental Health as a Relational, Contextual, and Temporal Phenomenon**
