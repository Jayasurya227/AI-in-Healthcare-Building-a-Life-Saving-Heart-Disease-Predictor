# AI in Healthcare: Building a Life-Saving Heart Disease Predictor 🩺

This project aims to build and evaluate various machine learning models to predict the presence and severity of heart disease based on patient clinical data. The analysis uses the UCI Heart Disease dataset, accessed via Kaggle Hub.

This serves as a comprehensive example of a classification workflow, suitable for a data science portfolio, covering data loading, exploratory data analysis (EDA), preprocessing using Scikit-learn pipelines, model training (comparing several algorithms), and detailed evaluation using classification metrics.

**Dataset:** `heart_disease_uci.csv` (from Kaggle Hub dataset `redwankarimsony/heart-disease-data`)
**Target Variable:** `num` (0 = No Disease, 1-4 = Increasing severity of disease) - Treated as a multi-class classification problem.
**Focus:** Demonstrating EDA for classification, robust preprocessing with Pipelines, training multiple classification models, and evaluating performance using relevant metrics like the classification report and confusion matrix.
**Repository:** [https://github.com/Jayasurya227/AI-in-Healthcare-Building-a-Life-Saving-Heart-Disease-Predictor](https://github.com/Jayasurya227/AI-in-Healthcare-Building-a-Life-Saving-Heart-Disease-Predictor)

***

## Key Techniques & Concepts Demonstrated

Based on the analysis within the notebook (`4_AI_in_Healthcare_Building_a_Life_Saving_Heart_Disease_Predictor.ipynb`), the following key machine learning concepts and techniques are applied:

* **Classification Fundamentals:** Understanding the goal of predicting discrete categories (severity levels 0-4).
* **Exploratory Data Analysis (EDA):**
    * Analyzed the distribution of the multi-class target variable (`num`).
    * Visualized relationships between key features (e.g., `age`, `thalach`, `cp`, `sex`) and the target variable using histograms, box plots, and count plots.
    * Calculated and visualized the correlation matrix for numerical features.
* **Data Preprocessing within Pipelines:**
    * **Imputation:** Used `SimpleImputer` to handle missing values (mean strategy for numerical, most frequent for categorical).
    * **Feature Scaling:** Applied `StandardScaler` to numerical features.
    * **Categorical Encoding:** Used `OneHotEncoder` (with `drop='first'`) to convert categorical features into a numerical format.
    * **Pipeline & ColumnTransformer:** Systematically combined preprocessing steps and applied them to the correct feature types.
* **Model Building & Comparison:** Trained and evaluated four different classification algorithms wrapped in Scikit-learn Pipelines:
    * **Logistic Regression** (Baseline Linear Model)
    * **Random Forest Classifier** (Ensemble Tree-based Model)
    * **Support Vector Machine (SVM)** (SVC)
    * **K-Nearest Neighbors (KNN)**
* **Model Evaluation (Multi-Class):**
    * Compared model performance using `classification_report`, which includes **precision, recall, f1-score,** and **accuracy** for each class and overall.
    * Visualized the **Confusion Matrix** for the best-performing model (SVM in this case) to understand class-specific prediction errors (True Positives, False Positives, True Negatives, False Negatives).
* **Feature Importance:** Extracted and visualized feature importances from the Random Forest model to identify the most influential predictors (e.g., `ca`, `thalach`, `thal`, `cp`).

***

## Analysis Workflow

The notebook follows a structured classification workflow:

1.  **Setup & Data Loading:** Importing libraries and loading the dataset using the Kaggle Hub API.
2.  **Exploratory Data Analysis (EDA):**
    * Initial inspection (`info()`, `describe()`, `isnull().sum()`).
    * Analyzing the distribution of the target variable (`num`).
    * Visualizing relationships between key features (`age`, `thalach`, `cp`, `sex`) and the target.
    * Plotting a correlation heatmap for numerical features.
3.  **Data Preprocessing (using Pipeline & ColumnTransformer):**
    * Defining feature lists (categorical and numerical).
    * Creating separate preprocessing pipelines for numerical (impute mean, scale) and categorical (impute mode, one-hot encode) features.
    * Combining these using `ColumnTransformer`.
    * Splitting data into training and testing sets (`train_test_split`) with stratification based on the target variable `y`.
4.  **Model Building:**
    * Creating full pipelines for Logistic Regression, Random Forest, SVM, and KNN by combining the preprocessor with the respective classifier.
    * Training each pipeline on the training data (`X_train`, `y_train`).
5.  **Model Evaluation:**
    * Generating predictions for each model on the test set (`X_test`).
    * Printing the `classification_report` for each model to compare precision, recall, f1-score, and accuracy across classes.
    * Generating and plotting the confusion matrix for the best model (SVM).
6.  **Feature Importance:** (Using the trained Random Forest pipeline)
    * Extracting feature names after one-hot encoding via the preprocessor step.
    * Getting feature importances from the classifier step.
    * Plotting the top 10 most important features.
7.  **Conclusion:** Summarizing the process, findings, and evaluation results.

***

## Technologies Used

* **Python**
* **Pandas & NumPy:** For data loading, manipulation, and analysis.
* **Matplotlib & Seaborn:** For data visualization.
* **Scikit-learn:** For data splitting (`train_test_split`), preprocessing (`StandardScaler`, `OneHotEncoder`, `SimpleImputer`, `ColumnTransformer`, `Pipeline`), modeling (`LogisticRegression`, `RandomForestClassifier`, `SVC`, `KNeighborsClassifier`), and evaluation (`classification_report`, `confusion_matrix`).
* **Kaggle Hub:** For dataset download (`kagglehub`).
* **Jupyter Notebook / Google Colab:** For the interactive analysis environment.

***

## How to Run the Project

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Jayasurya227/AI-in-Healthcare-Building-a-Life-Saving-Heart-Disease-Predictor.git](https://github.com/Jayasurya227/AI-in-Healthcare-Building-a-Life-Saving-Heart-Disease-Predictor.git)
    cd AI-in-Healthcare-Building-a-Life-Saving-Heart-Disease-Predictor
    ```
2.  **Install dependencies:**
    (It is recommended to use a virtual environment)
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn kagglehub jupyter
    ```
3.  **Set up Kaggle API (Optional but Recommended for Kaggle Hub):**
    * Ensure you have your `kaggle.json` API token configured if needed for consistent `kagglehub` access (though Colab often handles this).
4.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook "4_AI_in_Healthcare_Building_a_Life_Saving_Heart_Disease_Predictor.ipynb"
    ```
    *(Run the cells sequentially. The notebook handles dataset download via `kagglehub`.)*

***

## Author & Portfolio Use

* **Author:** Jayasurya227
* **Portfolio:** This project ([https://github.com/Jayasurya227/AI-in-Healthcare-Building-a-Life-Saving-Heart-Disease-Predictor](https://github.com/Jayasurya227/AI-in-Healthcare-Building-a-Life-Saving-Heart-Disease-Predictor)) demonstrates a complete multi-class classification workflow, including advanced preprocessing with pipelines and comparison of multiple models. It is suitable for showcasing skills on GitHub, resumes/CVs, LinkedIn, and during data science interviews, particularly for roles involving predictive modeling in domains like healthcare.
* **Notes:** Recruiters can observe the systematic approach to data preparation, model implementation using pipelines, thorough evaluation across multiple metrics and models, and interpretation of results through feature importance and confusion matrices. The analysis addresses a real-world problem with significant implications.
