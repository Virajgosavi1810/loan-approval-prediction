# Loan Approval Prediction

A Streamlit application that explores loan application data and predicts approval outcomes with Decision Tree and Random Forest classifiers.

## Features

- Upload a loan approval CSV file
- Data preprocessing and asset feature engineering
- Exploratory charts and correlation analysis
- Decision Tree and Random Forest classification
- Confusion matrices, classification reports, and evaluation metrics

## Run locally

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run LAP.py
```

Use `loan_approval_dataset.csv` with the uploader to run the analysis.

## Project files

- `LAP.py`: Streamlit application
- `loan_approval_dataset.csv`: Sample dataset
- `Loan Approval Prediction.ipynb`: Exploratory notebook
- `short description.md`: Project background and data dictionary
- `requirements.txt`: Python dependencies

## Notes

The optional `process.jpg` and `approved.jpg` illustrations are displayed when present. The app runs without them.

## License

This project is provided for educational and demonstration purposes.