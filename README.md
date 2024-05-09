Welcome to the Airbnb Data Analysis and Insights project! This report provides a comprehensive overview of our endeavor to enhance the Airbnb experience through data analysis. We explore industry dynamics, the problem we address, our project's differentiation, data gathering methods, and the key components comprising our project.

**Industry/Domain Insights**

Introduction to the Short-Term Accommodation Rental Industry:
The short-term accommodation rental industry has experienced remarkable growth, with the global market projected to reach $124.28 billion by 2027. This sector encompasses a diverse range of accommodations, from apartments and houses to unique and unconventional lodging options.

**Problem and Opportunity**

Problem Statement:
Our focus is on enhancing the Airbnb experience for both hosts and guests, specifically addressing the challenges of inaccurate pricing and recommendations. Hosts struggle to set the right property price, and guests often face difficulties finding accommodations that precisely match their preferences. Lower user satisfaction and revenue losses due to inefficient pricing quantify this problem.

**Opportunity for Improvement:**
Solving these challenges presents a significant opportunity for both Airbnb and its users. Improving pricing and recommendations allows us to provide a more tailored experience for guests, increasing satisfaction and loyalty. For hosts, our solution translates to increased occupancy rates, better retention, and higher revenue, aligning with Airbnb's growth objectives.

**Data Gathering**
Datasets Used:
Our project relies on several datasets, including Airbnb listings, reviews, and booking history. These datasets are sourced from Airbnb's API and publicly available data, allowing us to create a comprehensive view of the platform. Data features include property type, location, price, reviews, and more.

## 🛠️ Built with

+ [Python 3](http://www.python.org/) - Main programming language used, done in Jupyter Notebook.
+ [Pandas](https://pandas.pydata.org/) - Main library used to manipulate the datasets.
+ [Scikit-learn](https://scikit-learn.org/stable/) - Main library used for machine learning.
+ [Matplotlib](https://matplotlib.org/) - Used for graph plots and visualizations.
+ [Python NLTK](https://www.nltk.org/) - Used during exploratory analysis to get further insights into the textual data.
Part 1: Individual Model Analysis
**Model 1: Linear Regression**
The linear regression model was employed to predict Airbnb listing prices based on features such as accommodates, bathrooms, bedrooms, beds, latitude, longitude, minimum nights, and availability throughout the year. After handling missing values and converting relevant columns to numeric types, the model achieved the following performance metrics on the test set:
Mean Squared Error: 158403.14
R-squared: 0.06
Additionally, a scatter plot was generated to visualize the predicted vs. actual prices.

**Model 2: Random Forest Regression**
A random forest regression model was implemented using the same features as in the linear regression model. After training and evaluation, the model's performance metrics were as follows:
Mean Squared Error: 155336.70
R-squared: 0.08
The visualization included a scatter plot with a red dashed line representing a perfect prediction, emphasizing the comparison between actual and predicted prices. Furthermore, a bar plot illustrated feature importance in predicting price.

**Model 3: XGBoost Regression**
XGBoost, a gradient boosting algorithm, was employed with similar features for predicting Airbnb prices. The model's evaluation metrics were as follows:
Mean Squared Error: 168425.48
R-squared: -0.00
Similar to the previous models, a scatter plot visualized the predicted vs. actual prices, and a bar plot showcased feature importance.

**Part 2: Model System Development**
Ensemble Model with Linear Regression and Random Forest
A two-step ensemble model was developed using linear regression and random forest regression:
1.	The linear regression model was trained on the original features and used to make predictions on the test set.
2.	The residuals (the difference between true prices and linear regression predictions) were then used as features to train a random forest model.
3.	The final prediction was obtained by combining predictions from both the linear regression and random forest models.

The performance of the ensemble model was evaluated, yielding the following metrics:
Mean Squared Error (Final Model): 42166.30
R-squared (Final Model): 0.75
The report concludes by emphasizing that ensemble models, leveraging the strengths of different algorithms, can often lead to improved performance, providing a robust approach for predicting Airbnb listing prices.

**Cross-Validation for Linear Regression**
Using 5-fold cross-validation, the mean squared error (MSE) was calculated for each fold. The overall cross-validated MSE for Linear Regression is as follows:
•	Cross-validated MSE for Linear Regression: 122879.69

**Cross-Validation for Random Forest Regression**
Similarly, 5-fold cross-validation was employed for the Random Forest Regression model. The calculated mean squared error is presented below:
•	Cross-validated MSE for Random Forest: 111124.65

**Part 4: Hyperparameter Tuning for Random Forest Regression**
To enhance the Random Forest Regression model's performance, a grid search was conducted to find the optimal hyperparameters. The following parameter grid was explored:
•	Number of trees in the forest (n_estimators): [50, 100, 150]
•	Maximum depth of the tree (max_depth): [None, 10, 20, 30]
•	Minimum number of samples required to split an internal node (min_samples_split): [2, 5, 10]
•	Minimum number of samples required to be at a leaf node (min_samples_leaf): [1, 2, 4]

After performing a grid search with 5-fold cross-validation, the best hyperparameters for the Random Forest model were determined. The optimal configuration is as follows:

•	Best Hyperparameters: { 'n_estimators': 100, 'max_depth': 20, 'min_samples_split': 2, 'min_samples_leaf': 1 }
The Random Forest model was then retrained using these hyperparameters, and its performance on the test set was evaluated:
Mean Squared Error (Best Random Forest Model): 84214.92
R-squared (Best Random Forest Model): 0.40
This hyperparameter tuning process aims to optimize the model's ability to generalize and make accurate predictions on new data. The performance metrics on the test set reflect the improved performance achieved with the tuned Random Forest model.


