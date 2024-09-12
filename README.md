# Flight Price Prediction Report

## **1. Introduction**
Air travel has become an essential mode of transportation in today's fast-paced world. The pricing of airline tickets, however, remains a dynamic and complex process. Ticket prices can fluctuate significantly based on factors such as the time of booking, seasonality, airline policies, and even the route taken. Predicting flight prices with accuracy can help consumers make more informed decisions about when to purchase tickets, potentially saving significant costs. 

In this project, we aim to develop a machine learning model that can predict the price of domestic flights in India using historical flight data. By analyzing various factors that influence ticket prices, our goal is to assist travelers in identifying the best time to book flights, helping them secure affordable airfares.

---

## **2. Problem Statement**
The challenge faced by most travelers is determining when to purchase flight tickets to get the best deal. Airlines use complex, often opaque, algorithms to set ticket prices, and these prices fluctuate due to a multitude of factors such as the number of stops, time of booking, and seasonality. Our project seeks to solve this problem by building a predictive model that accurately forecasts the price of a flight ticket for domestic travel in India, taking into account all relevant factors.

---

## **3. Objective**
The primary objective of this project is to develop a predictive model that:
- Identifies key factors influencing flight prices.
- Accurately predicts the price of a flight ticket based on various attributes such as departure time, duration, airline, and the number of stops.
- Provides travelers with insights on when to book tickets to get the best prices, enabling cost savings.

---

## **4. Data Description**
We used a publicly available dataset from Kaggle that includes approximately **10,000 records** of domestic flights in India. The dataset captures important flight attributes, including:
- **Date of Journey**: The day the flight is scheduled to depart.
- **Departure and Arrival Times**: The exact times of departure and arrival.
- **Duration**: The total duration of the flight, measured in hours and minutes.
- **Airline**: The carrier operating the flight.
- **Number of Stops**: The number of layovers before reaching the destination.
- **Price**: The price of the flight ticket (target variable).

### **4.1 Data Preprocessing**
Before building the model, several data preprocessing steps were undertaken:
1. **Handling Missing Data**: Some features in the dataset, such as `Route` and `Additional_Info`, contained missing values. These features were deemed unnecessary for prediction and were dropped.
2. **Date and Time Transformation**: Features such as `Date_of_Journey`, `Dep_Time`, and `Arrival_Time` were converted into datetime formats to extract useful information like the day of the week, month, and hour of departure/arrival.
3. **Flight Duration Transformation**: The `Duration` feature was converted into a single continuous variable representing the total duration of the flight in minutes.
4. **Categorization of Stops**: The number of stops in a flight was categorized into specific classes (e.g., `Non-stop`, `1 Stop`, `2+ Stops`), enabling the model to understand the effect of layovers on ticket pricing.
5. **Outlier Detection**: Outliers were identified and handled using the **Z-score method** to minimize their impact on the predictive performance of the model.
6. **Normalization**: Since some of the features were highly skewed, the **PowerTransformer** was applied to normalize the distributions, improving the effectiveness of the regression algorithms.

---

## **5. Exploratory Data Analysis (EDA)**
Exploratory Data Analysis was performed to understand the relationships and patterns within the data. Key findings from this analysis include:

- **Price Distribution**: Flight prices were found to be right-skewed, with most tickets priced on the lower end and a few expensive flights driving the skewness.
- **Airline Influence**: Low-cost carriers such as **IndiGo**, **Air Asia**, and **SpiceJet** offered the cheapest flights, while premium carriers such as **Jet Airways** and **Air India** had higher average prices.
- **Seasonal Patterns**: Prices were found to vary by season, with **January** being the most expensive month and **April** being the cheapest. This suggests that demand during holiday seasons may drive prices higher.
- **Day of the Week**: Flight prices were generally higher on **Thursdays** and lower on **Saturdays**, reflecting the impact of demand on pricing.
- **Stops vs. Price**: Flights with more stops were generally more expensive than direct flights, indicating that layovers tend to increase the total cost of the journey.

These insights provided important guidance for model building, as they revealed which features were likely to have the strongest impact on the price of a ticket.

---

## **6. Modeling Approach**
To predict flight prices, we experimented with several machine learning regression algorithms, each offering different strengths for this type of problem.

### **6.1 Regression Models**
The following regression models were trained on the dataset:
1. **Random Forest Regression**: An ensemble learning method that constructs multiple decision trees and averages their predictions to improve accuracy and reduce overfitting.
2. **Ridge Regression**: A regularized linear regression model that applies a penalty on large coefficients, helping to avoid overfitting.
3. **XGBoost Regression**: A powerful gradient boosting algorithm that builds an ensemble of weak learners (decision trees), with each tree correcting the errors of its predecessors.
4. **Support Vector Regression (SVR)**: A regression technique that finds a hyperplane that best fits the data, though it may struggle with complex datasets.
5. **Decision Tree Regression**: A simple tree-based model that splits the dataset based on features that maximize prediction accuracy.

### **6.2 Evaluation Metrics**
To evaluate the performance of each model, we used the following metrics:
- **R-Squared (R²)**: Measures the proportion of variance in the dependent variable explained by the model.
- **Mean Squared Error (MSE)**: The average of the squared differences between predicted and actual values.
- **Root Mean Squared Error (RMSE)**: The square root of MSE, which gives the error in the same units as the target variable (flight prices).

#### **Model Performance Results**:
| **Algorithm**               | **R-Squared** | **MSE**       | **RMSE**  |
|-----------------------------|---------------|---------------|-----------|
| Random Forest Regression     | 0.698         | 6499906.79    | 2549.49   |
| Ridge Regression             | 0.572         | 9233304.15    | 3038.63   |
| XGBoost Regression           | 0.724         | 5949177.43    | 2439.09   |
| Support Vector Regression    | 0.032         | 20892704.53   | 4570.85   |
| Decision Tree Regression     | 0.557         | 9552134.66    | 3090.65   |

Based on these metrics, **XGBoost Regression** was the best-performing model with the highest **R-Squared** value of **0.724** and the lowest **RMSE** of **2439.09**, making it the most accurate model for predicting flight prices.

---

## **7. Hyperparameter Tuning**
To further improve the model’s performance, hyperparameter tuning was applied, particularly for the **Random Forest** and **XGBoost** models. After tuning, **Random Forest** improved its accuracy to **74%**, providing a competitive alternative to XGBoost.

---

## **8. Conclusion**
This project successfully demonstrated the use of machine learning models to predict flight prices with high accuracy. By analyzing features such as the day of travel, number of stops, airline, and flight duration, we were able to build a model that helps users anticipate price fluctuations and make cost-effective booking decisions.

### **Key Findings:**
- **Seasonality**: January is the most expensive month for flights, while April is the cheapest.
- **Day of the Week**: Flights on Saturdays are typically cheaper than those on Thursdays.
- **Number of Stops**: Direct flights are generally less expensive than those with one or more stops.
- **Best Model**: The **XGBoost Regression** model provided the best performance with an **R-Squared of 0.724**, making it the most suitable model for flight price prediction.

---

## **9. Future Work**
1. **Incorporating External Factors**: Future models could incorporate external data such as holidays, special events, or weather conditions, which could further enhance the model's accuracy.
2. **Dynamic Pricing Models**: Implementing real-time dynamic pricing models that adapt to current demand and supply conditions.
3. **Competitor Analysis**: Including pricing data from other airlines for more comprehensive analysis and improving price predictions.

---

