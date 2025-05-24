# SCRMBL

**SCRMBL** (Statistical Clustering for Robust Model-Based Learning) is a prototype framework for experimenting with probabilistic models and clustering techniques for analyzing complex, structured datasets. The repository emphasizes Bayesian-style analysis and reproducible research, leveraging tools such as `pymc`, `numpyro`, and flexible notebook workflows.

---

## 🧠 Motivation

The SCRMBL project was initiated to explore:

* Hierarchical and probabilistic modeling strategies
* Clustering algorithms under uncertainty
* Bayesian inference using modern probabilistic programming frameworks
* Model evaluation across synthetic and real-world datasets

It is built to enable fast iteration, comparison of methods, and rigorous diagnostics for modeling decisions.

---

## 📁 Project Structure

```
SCRMBL/
├── data/               # Input datasets and generation scripts
├── experiments/        # Core notebooks running different modeling experiments
├── models/             # Reusable model definitions (e.g., NumPyro, PyMC)
├── utils/              # Shared utilities: plotting, sampling, metrics
├── requirements.txt    # Python dependencies
└── README.md           # Project overview
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/c-daly/SCRMBL.git
cd SCRMBL
```

### 2. Install dependencies

Use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

## 📊 Usage

Start with one of the notebooks in `experiments/`, such as:

* `baseline_clustering.ipynb`: Classical clustering vs. probabilistic approaches
* `hierarchical_modeling.ipynb`: Building and sampling from custom Bayesian models
* `evaluation_metrics.ipynb`: Quantitative comparison across experiments

To run them:

```bash
jupyter notebook
```

---

## 🧠 Features

* Bayesian clustering using PyMC and NumPyro
* Synthetic data generation for testing model robustness
* Modular design for easy model reuse
* Visualization of posteriors and clustering outcomes
* Comparison of inference performance across toolkits

---

## 📚 References

This project draws inspiration from:

* Gelman et al., *Bayesian Data Analysis*
* Tutorials and examples from PyMC and NumPyro communities
* Research on statistical learning under uncertainty

Specific citations are included in relevant notebooks.

---

## 📝 License

This project is licensed under the MIT License.
