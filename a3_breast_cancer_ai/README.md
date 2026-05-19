# A3 Breast Cancer AI Triage Prototype

This repository supports the UTS 36121 Assessment 3 report draft.

## Problem

The prototype explores an AI-assisted decision-support workflow for estimating whether a breast mass is malignant or benign from cell-nucleus features extracted from fine-needle aspirate images.

## Dataset

The code uses the Breast Cancer Wisconsin Diagnostic dataset distributed through scikit-learn. It contains 569 observations and 30 continuous features.

## AI paradigms included

- Structural knowledge representation: a shallow interpretable decision tree/rule model.
- Probabilistic reasoning: Gaussian Naive Bayes and predicted malignancy probabilities.
- Predictive modelling: logistic regression and random forest.
- Multi-layer network: MLP classifier.
- Responsible AI framing: the output is treated as triage support, not autonomous diagnosis.

## Run

Create a virtual environment if preferred, then install the required packages:

```powershell
python -m pip install -r requirements.txt
python .\src\run_experiment.py
```

On macOS or Linux, use:

```bash
python -m pip install -r requirements.txt
python ./src/run_experiment.py
```

Outputs are written to the `outputs/` directory.
