# Predicting and Explaining Traffic Violations Using Machine Learning and XAI Techniques

This project applies machine learning models and Explainable Artificial Intelligence (XAI) techniques to predict and interpret traffic violations. Using Decision Tree and Logistic Regression models, the study evaluates prediction accuracy and provides transparency into the models’ decision-making processes with XAI tools such as LIME and SHAP.

## Overview

Traffic violations significantly contribute to accidents, property damage, and traffic delays. This project aims to:
1. Predict traffic violation severity using machine learning.
2. Enhance model interpretability using XAI techniques like LIME and SHAP.
3. Identify key features driving predictions (e.g., vehicle type, weather, driver demographics).

### Key Features:
- **Models Used:** Decision Tree, Logistic Regression.
- **XAI Tools:** LIME (Local Interpretable Model-agnostic Explanations), SHAP (SHapley Additive exPlanations).
- **Performance Metrics:** Accuracy, precision, recall, F1 score, ROC-AUC, and Cohen's Kappa.

## Methodology

### Data Preprocessing
1. **Handling Missing Data:** 
   - Dropped features with >20% missing values.
   - Imputed numerical and categorical missing values using the median and mode, respectively.
2. **Encoding Categorical Variables:** 
   - Transformed variables like vehicle type and violation type into numerical form using Label Encoding.
3. **Scaling Numerical Features:** 
   - Used StandardScaler to normalize features like age and driving experience for consistent model performance.

### Model Selection
- **Decision Tree:** Offers high interpretability and decision transparency but can overfit without careful tuning.
- **Logistic Regression:** Performs well with linearly separable data and provides insights into feature relationships.

### XAI Techniques
- **LIME:** Provides localized explanations for individual predictions by perturbing input data.
- **SHAP:** Offers both global and local interpretability by quantifying each feature's contribution to model predictions.

## Results

1. **Model Performance:**
   - **Decision Tree:** 
     - Accuracy: 75.6%
     - F1 Score: 76.0%
     - ROC-AUC: 59.9%
   - **Logistic Regression:**
     - Accuracy: 84.1%
     - F1 Score: 76.8%
     - ROC-AUC: 64.6%
   - Logistic Regression outperformed Decision Tree in accuracy and recall but struggled with class imbalances.

2. **Key Insights:**
   - **Decision Tree:** Influenced by features like pedestrian movement, road surface conditions, and number of vehicles.
   - **Logistic Regression:** Relied on driver demographics, weather conditions, and vehicle type for predictions.

3. **XAI Contributions:**
   - SHAP revealed global feature importance, showing consistent trends across the dataset.
   - LIME provided localized insights, highlighting feature impacts on individual predictions.

## Conclusion

The study demonstrates the potential of combining machine learning with XAI techniques for traffic violation prediction. While Logistic Regression showed higher accuracy, the Decision Tree excelled in interpretability. XAI tools like LIME and SHAP significantly enhance model transparency, fostering trust and enabling actionable insights.

## Future Work

- Explore advanced models like Random Forest or Gradient Boosting for improved performance.
- Enrich datasets with diverse traffic scenarios to reduce class imbalances.
- Integrate contextual data (e.g., real-time traffic and weather conditions) for more accurate predictions.

## Usage

### Requirements
- Python 3.x
- Libraries: NumPy, Pandas, Scikit-learn, SHAP, LIME, Matplotlib.

### Steps to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/MuhammadAyob/Explaining-Traffic-Violations-Using-Machine-Learning-and-xAI.git
   cd Explaining-Traffic-Violations-Using-Machine-Learning-and-xAI

## License

This project is for demonstration purposes. Usage or distribution requires explicit permission.
