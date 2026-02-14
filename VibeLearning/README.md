# Loan Default Prediction

End-to-end machine learning project to predict loan defaults.

## Project Structure

- `data/`: Contains the generated dataset.
- `src/`: Source code for the project.
    - `data_generator.py`: Generates synthetic data.
    - `preprocessing.py`: handles data cleaning and feature engineering.
    - `train.py`: Trains and evaluates models.
    - `predict.py`: Runs inference on new data.
- `models/`: Scaled models and preprocessing pipelines.

## Setup

1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Generate Data**:
   ```bash
   python src/data_generator.py
   ```

2. **Train Model**:
   ```bash
   export PYTHONPATH=.
   python src/train.py
   ```

3. **Inference**:
   ```bash
   export PYTHONPATH=.
   python src/predict.py
   ```
