# Predicitng Telecom Churn using Neural Networks

### Company Information:
A telecom company called ‘Firm X’ is a leading telecommunications provider in the country. The company earns most of its revenue by providing internet services. Based on the past and current customer information, the company has maintained a database containing personal/demographic information, the services availed by a customer and the expense information related to each customer.

### Problem Statement:
 ‘Firm X’ has a customer base set across the country. In a city ‘Y’, which is a significant revenue base for the company, due to heavy marketing and promotion schemes by other companies, this company is losing customers i.e. the customers are churning. Whether a customer will churn or not will depend on data from the following three buckets:-
* Demographic Information
* Services Availed by the customer
* Overall Expenses

##### The aim is to automate the process of predicting if a customer would churn or not and to find the factors affecting the churn.

### Goal
To develop multiple predictive models and find the best predictive model using the library "h2o". 

### Data cleaning Steps required to be performed:-
* Load the data file.
* Make bar charts displaying the relationship between the target variable and various other features.
* Perform de-duplication of data.
* Bring the data in the correct format
* Find the variables having missing values and impute them.  
* Perform outlier treatment if necessary

## Model Building and Validation

| Check Points                  | Aim                                                                                                                                                                                                                                                                                                         |
|-------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Model building without epochs | Experiment with hyperparameters, observe results and iterate them based on sound reasoning. Explanation of choice of hyperparameters based on the results obtained with each model.                                                                                                                         |
| Model building with epochs    | Experiment with hyperparameters (including the number of epochs), observe results and iterate them based on sound reasoning. Explanation of the choice of hyperparameters based on the results obtained with each model.  The choice of final model is based on sound reasoning based on these experiments. |
| Final Model                   | The final model performs reasonably well on the  test data.                                                                                                                                                                                                                                                 |
