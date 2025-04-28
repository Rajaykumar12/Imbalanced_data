# Insurance Claims Classification on Imbalanced Data

## Project Overview
This project demonstrates handling imbalanced classification for insurance claims data using Python. It includes exploratory data analysis, data balancing techniques, and machine learning model implementation.

## Technologies Used
- Python
- Libraries: pandas, scikit-learn, matplotlib, seaborn

## Key Steps

### 1. Data Loading and Initial Analysis
```python
import pandas as pd
data = pd.read_csv("Insurance claims data.csv")
```

### 2. Exploratory Data Analysis
- Visualized distribution of claim status
- Analyzed numerical features:
  - Subscription length
  - Vehicle age
  - Customer age
- Examined categorical features:
  - Region code
  - Segment
  - Fuel type

### 3. Handling Class Imbalance
- Used oversampling technique to balance the dataset
- Achieved equal distribution of both classes (54,844 entries each)

### 4. Feature Selection
- Implemented Random Forest Classifier for feature importance
- Encoded categorical variables using LabelEncoder
- Identified most significant predictors for insurance claims

### 5. Model Building
- Used Random Forest Classifier
- Features included:
  - Customer demographics
  - Vehicle information 
  - Subscription details
- Split data: 70% training, 30% testing

### 6. Results
- Built classification model with balanced data
- Evaluated performance on original imbalanced dataset
- Visualized classification accuracy using pie chart

## Key Findings
- Successfully balanced the dataset using oversampling
- Identified important features affecting insurance claims
- Created robust model for predicting insurance claim likelihood

## Future Improvements
- Test other balancing techniques (SMOTE, undersampling)
- Experiment with different algorithms
- Include feature engineering
- Perform hyperparameter tuning

## Usage
1. Load the insurance claims dataset
2. Run the Jupyter notebook
3. Follow the analysis and modeling steps
4. Evaluate results on new data

## Repository Structure
```
├── classification_on_imbalanced_data.ipynb
├── Insurance claims data.csv
└── README.md
```
