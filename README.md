# Investigating the Utility of Machine Learning Algorithms in Glycan Specificity of Influenza A

This repository accompanies the MSc dissertation *"Investigating the Utility of Machine Learning Algorithms in Glycan-Specificity in Influenza A"* (King’s College London, 2024). It contains the research report and Python code used to build and evaluate predictive models for influenza A glycan specificity.

---

## Overview

Zoonotic influenza A viruses, especially H5 and H7 subtypes, pose significant pandemic risks due to their potential to cross species barriers. The hemagglutinin (HA) protein plays a central role in host specificity by binding to sialic acid receptors.  
This project evaluates the ability of machine learning (ML) algorithms to predict glycan specificity (avian α2,3 vs human α2,6 binding) using viral sequence data, and identifies key mutations associated with receptor adaptation.

---

## Repository Structure

```text
├── data/
│   ├── alignments/                # Multiple sequence alignments (H3, H5, H7)
│   └── original/                  # Original raw sequence data
│
├── results/
│   └── Utility of ML in Glycan Specificity.pdf   # Report of results (figures, metrics, discussion)
│
├── test-installs-notebooks/
│   └── Utility of ML in Glycan Specificity.ipynb # Jupyter notebook with full pipeline
│
├── pdb/                           # Structural/3D data 
│
└── README.md                      # Project documentation                                 
```

---

## Overview

This project investigates whether machine learning (ML) algorithms can predict glycan specificity (avian α2,3 vs human α2,6 binding) from influenza A HA protein sequences.  

Goals:
- Compare ML classifiers (Logistic Regression, SVC, Random Forest, XGBoost).  
- Identify key HA mutations linked to host specificity.  
- Evaluate predictive accuracy and feature importance.  

---
## Workflow

1. **Data Collection**  
   - HA sequences from GISAID EpiFlu (H3, H5, H7).  
   - Cleaned, aligned (MUSCLE), trimmed (SignalP).  

2. **Feature Engineering**  
   - Mutation encoding (Burke & Smith numbering).  
   - Polynomial features for co-occurrence.  
   - Feature selection with ANOVA F-test.  

3. **Model Training**  
   - Implemented Logistic Regression, SVC, Random Forest, XGBoost.  
   - GridSearchCV with 5-fold CV, weighted F1 as primary metric.  

4. **Evaluation**  
   - Metrics: Accuracy, Precision, Recall, F1.  
   - Feature importance analysis across models.  

---
## Results

- **Random Forest** achieved the highest balance of accuracy (0.731) and weighted F1 (0.622).  
- Logistic Regression had the best cross-validation weighted F1 (0.534).  
- XGBoost and SVC performed less consistently due to class imbalance.  
- **Key mutations identified**: T192I, Q226L, S227N — critical for receptor adaptation.  

Full results with figures are available in:  
`results/Utility of ML in Glycan Specificity.pdf`

---

## Usage

### Requirements
- Python 3.9+  
- Libraries: `numpy`, `pandas`, `biopython`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`  

---

## Citation
Sümer, S. Z. (2024). Investigating the Utility of Machine Learning Algorithms in Glycan-Specificity in Influenza A. MSc Dissertation, King’s College London.

Data Sources & Tools
	•	GISAID. EpiFlu Database. https://www.gisaid.org/
	•	Edgar, R. C. (2004). MUSCLE: multiple sequence alignment with high accuracy and high throughput. Nucleic Acids Res, 32(5), 1792–1797.
	•	Almagro Armenteros, J. J. et al. (2019). SignalP 5.0 improves signal peptide predictions using deep neural networks. Nat Biotechnol, 37, 420–423.
	•	Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. JMLR, 12, 2825–2830.
	•	Chen, T. & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. KDD ’16, 785–794.

This list is not exhaustive. The full set of references is available in the dissertation (.pdf/.md file).
