# Heart-Disease-Prediction
Dataset¶
The dataset used in this analysis contains information about various factors that could influence heart disease. The columns in the dataset are:

**Age**: Age of the individual (years).
**Gender**: Gender of the individual (Male/Female).
**Cholesterol**: Cholesterol level in mg/dL.
**Blood Pressure**: Systolic blood pressure in mmHg.
**Heart Rate**: Heart rate in beats per minute.
**Smoking**: Smoking status (Never/Former/Current).
**Alcohol Intake**: Alcohol intake frequency (None/Moderate/Heavy).
**Exercise Hours**: Hours of exercise per week.
**Family History**: Family history of heart disease (Yes/No).
**Diabetes**: Diabetes status (Yes/No).
**Obesity**: Obesity status (Yes/No).
**Stress Level**: Stress level on a scale of 1 to 10.
**Blood Sugar**: Fasting blood sugar level in mg/dL.
**Exercise Induced Angina**: Presence of exercise-induced angina (Yes/No).
**Chest Pain Type**: Type of chest pain experienced (Typical Angina/Atypical Angina/Non-anginal Pain/Asymptomatic).
**Heart Disease**: Target variable indicating presence of heart disease (0: No, 1: Yes).

**Data Preprocessing**: Summary Statistics
Before diving into model training, it's crucial to understand the basic statistical properties of the dataset. The following summary statistics provide insights into the distribution and range of numerical features related to heart disease:

**Interpretation**: 
Cholesterol: The cholesterol levels range from a minimum of 150 mg/dL to a maximum of 349 mg/dL, with a mean of 249.94 mg/dL. The high standard deviation (57.91) indicates considerable variation in cholesterol levels among individuals.

**Blood Pressure**: Systolic blood pressure ranges from 90 mmHg to 179 mmHg. The mean value is 135.28 mmHg, with a standard deviation of 26.39, suggesting a broad range of blood pressure readings in the dataset.

**Heart Rate**: The heart rate varies between 60 and 99 beats per minute, with a mean of 79.20 bpm. The standard deviation is relatively low (11.49), indicating that heart rate values are more clustered around the mean.

**Exercise Hours**: The number of exercise hours per week ranges from 0 to 9 hours, with a mean of 4.53 hours. The relatively high standard deviation (2.93) shows variability in exercise habits among individuals.

**Stress Level**: Stress levels, on a scale from 1 to 10, have a mean of 5.65 and range from 1 to 10. The standard deviation (2.83) reflects some variability in perceived stress among the dataset's individuals.

**Blood Sugar**: Fasting blood sugar levels range from 70 mg/dL to 199 mg/dL. The mean blood sugar level is 134.94 mg/dL, with a standard deviation of 36.70, suggesting a wide range of values.

**Heart Disease**: The target variable indicating the presence of heart disease shows that 39.2% of individuals in the dataset have heart disease (1), while 60.8% do not (0). The mean of 0.39 and the standard deviation of 0.49 highlight the imbalance in class distribution.

**Missing Data Analysis**:Missing Data Analysis
Before proceeding with data analysis and model building, it's essential to check for missing values in the dataset, as they can impact the accuracy and reliability of the predictive models. The following code was used to identify the number of missing values in each column of the dataset:

**Interpretation**:
The Alcohol Intake column has 340 missing values, representing a significant portion of the dataset. Handling this missing data will be crucial, whether through imputation, removal, or other methods, to ensure the dataset remains robust for modeling.
All other features have no missing values, meaning they are complete and ready for further analysis.

**Exploratory Data Analysis (EDA) on Risk Factors for Heart Disease**:
Visualizing the Relationship Between Cholesterol Levels and Heart Disease by Smoking Status.
To explore the relationship between cholesterol levels and the likelihood of heart disease, while considering the influence of smoking status, I created a scatter plot with regression lines using Seaborn's lmplot. This visualization allows us to see how cholesterol levels correlate with heart disease across different smoking categories: Never, Former, and Current smokers.

Interpretation of the Plot:

**Former Smokers**: The regression line for former smokers indicates a moderate positive correlation between cholesterol levels and heart disease risk in this group. As cholesterol levels increase, the likelihood of heart disease also increases, though the slope suggests a less steep rise compared to other groups.

**Never Smokers**: The line for never smokers shows a steeper positive correlation between cholesterol levels and heart disease. This indicates that in individuals who have never smoked, higher cholesterol levels are more strongly associated with an increased risk of heart disease.

**Current Smokers**: The regression line for current smokers suggests a significant, but slightly less pronounced, relationship between cholesterol levels and heart disease compared to never smokers.

Distribution of Blood Pressure by Heart Disease Status:
To investigate how blood pressure levels are distributed among individuals with and without heart disease, I generated a histogram using Seaborn's histplot function. The histogram illustrates the distribution of blood pressure readings, with the bars colored to differentiate between individuals who have heart disease (1) and those who do not (0).

**Interpretation of the Plot**:

The histogram shows a range of blood pressure levels across the dataset, with clear differences between the two groups:

**Individuals without Heart Disease (Hue 0)**: The distribution is relatively uniform across various blood pressure ranges, with a noticeable number of individuals having lower blood pressure levels (around 100-120 mmHg).

**Individuals with Heart Disease (Hue 1)**: The distribution indicates a higher frequency of heart disease cases at elevated blood pressure levels, particularly around 140-160 mmHg.

This plot suggests a potential relationship between higher blood pressure levels and the prevalence of heart disease, as indicated by the increased number of heart disease cases in the higher blood pressure ranges.

Distribution of Smoking Status by Heart Disease:

To examine how smoking status correlates with the incidence of heart disease, I created a histogram using Seaborn's histplot function. The plot displays the count of individuals in each smoking category—Current, Never, and Former—divided by whether they have heart disease (1) or not (0).

**Interpretation of the Plot**:The histogram visualizes the distribution of smoking status among individuals with and without heart disease.
**Current Smokers**: The distribution shows a fairly even split between those with and without heart disease, with a slight dominance of non-heart disease cases.

**Never Smokers**: This group also exhibits a relatively balanced distribution, with similar counts of individuals with and without heart disease.

**Former Smokers**: The plot shows a higher overall count in this category, with a noticeable prevalence of individuals without heart disease compared to those with heart disease.

**Distribution of Blood Sugar Levels by Heart Disease**:
To explore the relationship between blood sugar levels and heart disease, I utilized Seaborn's histplot function to visualize the distribution of individuals based on their blood sugar levels, categorized by the presence of heart disease (1) or absence of it (0).

**Interpretation of the Plot**:
The histogram presents the distribution of blood sugar levels among individuals with and without heart disease:

**Blood Sugar Levels (80 - 200 mg/dL)**: The plot shows a relatively even distribution of individuals across various blood sugar levels. In each blood sugar category, there is a noticeable presence of both heart disease and non-heart disease cases.
**High Blood Sugar Levels (160 - 200 mg/dL)**: In the higher blood sugar categories, there is a slightly larger count of individuals without heart disease compared to those with heart disease.
**Distribution of Stress Levels by Heart Disease**:
To examine the relationship between stress levels and the presence of heart disease, I used Seaborn's histplot function to visualize the distribution of individuals according to their stress levels, with a distinction made between those with and without heart disease.

**Interpretation of the Plot**:
The histogram reveals that individuals with stress levels between 6 and 10 tend to have a higher incidence of heart disease compared to those with stress levels between 1 and 5. Notably, however, there is an exception within the lower stress levels: people with stress levels of 2 and 3 also show a higher prevalence of heart disease, contrasting with the general trend observed in the lower stress range.

**Distribution of Age by Heart Disease**:
To explore how age correlates with the presence of heart disease, I plotted a histogram using Seaborn’s histplot function, displaying the distribution of ages while distinguishing between individuals with and without heart disease.

**Interpretation of the Plot**:
The histogram reveals a clear age-related trend in the occurrence of heart disease. Individuals under the age of 50 predominantly do not have heart disease, as shown by the higher count in the "no heart disease" category. However, from the age of 50 onwards, the distribution shifts significantly, with a higher proportion of individuals suffering from heart disease. This trend becomes more pronounced as age increases, particularly in the 60 to 80 age range, where heart disease is more prevalent.

**Interpretation of MI Scores**:
The Mutual Information scores for a selection of features reveal the strength of their relationship with the presence of heart disease:
**Cholesterol**: With an MI score of 0.468090, cholesterol is the most significant feature related to heart disease among the selected features.
**Gender**: Although less impactful, gender still shows some dependency with an MI score of 0.016676.
**Smoking**: This feature has a lower MI score of 0.010393, indicating a weaker but still relevant relationship.
**Heart Rate**: The MI score for heart rate is quite low at 0.001186, suggesting minimal relevance.
**Diabetes**: Interestingly, diabetes has an MI score of 0.000000, indicating no measurable dependency with heart disease in this context.
This analysis helps to identify which features should be prioritized in predictive modeling efforts for heart disease, guiding the selection of variables for further investigation and model development.

**Data Splitting and Feature Selection**:
To prepare the data for machine learning, I split the dataset into training and validation subsets and selected relevant features for modeling.
Splitting the Data: The dataset is divided into training (80%) and validation (20%) subsets. The target variable y is 'Heart Disease,' while X consists of all other features. The random_state=0 parameter ensures that the results are reproducible.

**Feature Selection**:
Categorical Columns: I selected categorical features with low cardinality (fewer than 10 unique values). These are typically more manageable for machine learning models, and the selection criterion ensures that only categorical columns are included.

**Numerical Columns**: I also selected all numerical features, as they generally provide valuable information for predicting the target variable.
**Combining Features**: The selected categorical and numerical columns are then combined to form the final training (X_train) and validation (X_valid) datasets.
Finally, the code checks the data type of the target variable y, confirming that it is an integer (int64). This ensures that the target variable is correctly formatted for subsequent machine learning models.

Data Preprocessing and Model Building¶
To build an effective machine learning model, I employed preprocessing steps tailored to both numerical and categorical data, followed by modeling using the XGBoost classifier.

**Numerical Data**: For numerical features, I used a SimpleImputer with a mean strategy to fill in any missing values. This ensures that the model doesn't encounter any null values, which could otherwise cause errors.

**Categorical Data**: For categorical features, I created a pipeline that first imputes missing values using the most frequent value (SimpleImputer) and then applies one-hot encoding (OneHotEncoder). This transformation converts categorical variables into a format suitable for the model by creating binary columns for each category, while ignoring any unseen categories in new data.

**Combining Transformations**: I combined the numerical and categorical preprocessing steps using ColumnTransformer, ensuring that the right transformations are applied to the correct types of data.

**Model Building and Evaluation**:
Model Choice: I selected the XGBClassifier, a robust model well-suited for classification tasks, and fine-tuned it with 500 estimators and a learning rate of 0.05. These parameters aim to balance model performance and training time.

**Pipeline Creation**: I combined the preprocessing steps and the model into a single pipeline. This approach ensures that the same transformations are applied consistently to both the training and validation data, reducing the risk of data leakage.

**Model Training and Prediction**: The pipeline was fitted on the training data, and predictions were made on the validation set.

**Model Evaluation**: I evaluated the model's performance using accuracy, confusion matrix, and classification report.
This evaluation suggests that while the model has room for improvement, it demonstrates a reasonable balance between precision and recall, particularly given the complexity of predicting heart disease.
Interpretation:

**Accuracy**: The model achieved an accuracy of 60.5%, indicating that it correctly predicted the presence or absence of heart disease 60.5% of the time.

**Confusion Matrix**: The confusion matrix shows that out of 200 cases, 72 were true negatives (correctly identified as not having heart disease), and 49 were true positives (correctly identified as having heart disease). However, there were 40 false positives and 39 false negatives.

**Classification Report**: The classification report provides precision, recall, and F1-score for both classes. The model performs slightly better in predicting the absence of heart disease (class 0) than its presence (class 1).
Logistic Regression Model for Heart Disease Prediction¶
In addition to the XGBoost classifier, I also implemented a Logistic Regression model to predict heart disease. Logistic Regression is a linear model that is widely used for binary classification tasks.

**Data Preprocessing and Model Setup**:
**Standardization**: Since Logistic Regression benefits from standardized features, I included standard scaling in the preprocessing pipeline to normalize numerical features. However, this step is implicitly handled by the pipeline.

**Logistic Regression Model**: The Logistic Regression model was configured with the following parameters:
C=0.5: This parameter controls the regularization strength. A smaller value of C increases regularization, helping prevent overfitting.
penalty='l2': L2 regularization was applied to penalize large coefficients, thus stabilizing the model and improving generalization.
solver='lbfgs': The lbfgs solver is an efficient algorithm for logistic regression, especially with a large number of features.
max_iter=200: The maximum number of iterations was set to 200 to ensure convergence.
class_weight='balanced': This option was used to address class imbalance by adjusting the weights of the classes inversely proportional to their frequencies.

**Model Building and Evaluation**:
Pipeline Integration: The model was integrated into a pipeline along with the preprocessing steps. This ensures consistent application of preprocessing to both training and validation datasets.

**Model Training and Predictions**: The pipeline was fitted to the training data, and predictions were generated for the validation set.

**Model Evaluation**: The model was evaluated using accuracy, confusion matrix, and classification report.

**Accuracy**: The Logistic Regression model achieved an accuracy of 58%, meaning it correctly predicted the presence or absence of heart disease 58% of the time.

**Confusion Matrix**: The confusion matrix shows that out of 200 cases, 67 were true negatives (correctly identified as not having heart disease) and 49 were true positives (correctly identified as having heart disease). However, there were 45 false positives and 39 false negatives.

**Classification Report**:
**Class 0 (No Heart Disease)**: The precision, recall, and F1-score for class 0 were 0.63, 0.60, and 0.61, respectively. This indicates that the model performs slightly better in predicting the absence of heart disease.
**Class 1 (Heart Disease)**: The precision, recall, and F1-score for class 1 were lower, at 0.52, 0.56, and 0.54, respectively, indicating some challenges in predicting heart disease cases.
Macro Average and Weighted Average: Both averages hover around 0.58, reflecting the model's balanced performance across both classes.

**Insights**:
The Logistic Regression model shows moderate performance with an accuracy of 58%, slightly lower than the XGBoost model. While the model does well in handling class imbalance, as evidenced by balanced precision and recall, it still struggles to differentiate between heart disease and non-heart disease cases, particularly with precision for predicting heart disease (class 1).

This model could potentially be improved with additional feature engineering, hyperparameter tuning, or by exploring more complex models. Nonetheless, it provides a baseline for linear classification in this context.

**Conclusion on Model Performance for Heart Disease Prediction**:
In this notebook, we explored two different machine learning models—XGBoost and Logistic Regression—to predict heart disease based on a variety of patient features. The models were developed and evaluated with preprocessing steps included in a pipeline to ensure consistent data handling.

1. **Exploratory Data Analysis (EDA) Insights**:
Before delving into model building, we conducted an EDA to understand the underlying patterns and relationships within the dataset. Some key observations included:

**Cholesterol Levels**: Cholesterol was identified as a significant predictor of heart disease, exhibiting a relatively high mutual information (MI) score. Gender and Smoking: These categorical features, although less influential than cholesterol, also showed a correlation with heart disease presence. Other Features: Factors like heart rate and diabetes showed minimal predictive power, as indicated by their low MI scores. These insights guided us in selecting features and preprocessing strategies for the models.

2. **Model 1**: XGBoost Classifier
Performance: The XGBoost model achieved an accuracy of 60.5%. It demonstrated better performance than Logistic Regression, particularly in classifying the absence of heart disease (class 0).

Evaluation Metrics:

Confusion Matrix: The model correctly identified 72 out of 112 negative cases (no heart disease) and 49 out of 88 positive cases (heart disease).
Classification Report: The F1-scores were 0.65 for class 0 and 0.55 for class 1, showing a stronger ability to identify non-heart disease cases but with some struggles in detecting heart disease cases.
3. **Model 2**: Logistic Regression
Performance: The Logistic Regression model showed a slightly lower accuracy of 58%. It performed similarly to XGBoost in classifying class 0 but was less effective overall.

Evaluation Metrics:
**Confusion Matrix**: This model correctly predicted 67 out of 112 negative cases and 49 out of 88 positive cases, with a similar number of false positives and false negatives as XGBoost.
Classification Report: The F1-scores were 0.61 for class 0 and 0.54 for class 1, indicating a modest performance in predicting heart disease.
4. **Comparative Insights**:
Accuracy: Both models performed relatively close to each other, with XGBoost having a slight edge in accuracy.
Precision and Recall: XGBoost exhibited a balanced trade-off between precision and recall, especially for the more challenging task of predicting heart disease (class 1). Logistic Regression, while simpler, did not match the performance of XGBoost.
Handling Imbalance: Both models were designed to handle class imbalance, with Logistic Regression using the class_weight='balanced' parameter and XGBoost utilizing its inherent capability to deal with imbalance.
5. **Final Thoughts**:
While the XGBoost model outperformed Logistic Regression in this notebook, both models struggled to achieve high precision and recall for the heart disease class (class 1). This suggests that further improvements could be made, possibly through more sophisticated feature engineering, tuning of model hyperparameters, or exploring other advanced algorithms.

**In summary**:
XGBoost: Better overall accuracy and balance between precision and recall. Logistic Regression: A simpler model with decent performance, but less effective compared to XGBoost.

Given the medical context of this problem, where the cost of false negatives (failing to predict heart disease) could be significant, additional focus on improving recall for the heart disease class would be crucial in a real-world application. Future work might also include cross-validation and exploration of ensemble techniques to boost model performance.
