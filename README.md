# Lloyds Banking Data Science Job Simulation: Customer Retention Enhancement through Predictive Analytics 

This project simulates a business-critical engagement with Lloyds Banking Group, aimed at reducing customer churn for its subsidiary, SmartBank. It leverages predictive analytics and supervised machine learning techniques to identify at-risk customers, enabling the delivery of strategic retention interventions. The case study demonstrates how data science can support financial institutions in protecting revenue, improving client satisfaction, and optimizing digital engagement channels.

## Executive Summary

Faced with a growing trend of customer attrition—particularly among young professionals and small business owners—SmartBank sought a data-driven solution to predict and mitigate churn. With competitors offering highly personalized digital solutions, retaining valuable clients has become a strategic imperative. This analysis integrates behavioral, demographic, and interaction-level data to uncover the drivers of churn and implement a robust prediction framework. The deliverable includes a machine learning pipeline, customer segmentation insights, and evidence-based business recommendations.

## Business Challenge

In 2022, SmartBank observed a **20.4% churn rate** across its customer base, posing a direct threat to its profitability and market share. A detailed diagnostic revealed that the churn is not tied to seasonal trends and is spread across digital engagement points such as mobile apps, websites, and online banking portals. The data indicates that churners are disproportionately represented among married individuals, low-income earners, and customers whose service complaints were unresolved. In addition, a noticeable lag in login recency and feedback resolution correlates with elevated attrition risks.

<p align="center">
  <img src="/assets/churn_rate_distribution.png" alt="Churn Rate Distribution">
</p>

<p align="center">
  <img src="/assets/digital_channel_usage.png" alt="Digital Channel Usage">
</p>

<p align="center">
  <img src="/assets/login_recency_timeseries.png" alt="Login Recency Time Series">
</p>

## Analytical Approach

The solution framework consists of exploratory analysis, feature engineering, and model development. Data was sourced from five structured datasets encompassing customer demographics, transaction history, service interaction logs, online activity, and churn status.

After preprocessing steps—such as one-hot encoding of categorical variables, normalization of numeric features, and balancing the dataset using SMOTE—a series of machine learning algorithms were evaluated, including Logistic Regression, Decision Trees, Random Forests, Neural Networks, and XGBoost. The XGBoost classifier was ultimately selected for its superior performance in capturing the complex interaction patterns within the dataset.

Model evaluation was based on multiple performance metrics including AUC-ROC, precision, recall, and F1-score. The best-performing model, **XGBoost**, achieved an **AUC of 0.57**—highlighting modest but informative predictive power. Feature importance analysis revealed that interaction frequency, service usage type, and resolution status were the most critical predictors of churn.

<p align = "center">
  <img src= "/assets/auc_comparison.png" alt= "AUC-ROC comparison">
</p>

## Strategic Insights

The predictive model uncovered several actionable insights:

- Customers who submitted unresolved complaints or feedback were significantly more likely to churn, emphasizing the need for service recovery protocols.

<p align="center">
  <img src="/assets/interaction_resolution.png" alt="Interaction Resolution Breakdown">
</p>

<p align="center">
  <img src="/assets/interaction_churn.png" alt="Resolution Status and Churn">
</p>


- Married individuals and middle-aged adults (30–39 years) exhibited higher churn rates at nearly 30% and 35.3%, respectively. This is likely linked to life-stage-driven financial stressors.

<p align="center">
  <img src="/assets/age_group_churn.png" alt="Age Group Churn Distribution">
</p>

<p align="center">
  <img src="/assets/marital_status_churn.png" alt="Marital Status Churn Distribution">
</p>

- Low-income segments experienced a churn rate exceeding 35%, indicating the need for affordability-focused financial products.

<p align="center">
  <img src="/assets/income_churn_analysis.png" alt="Income Level and Churn">
</p>

- Digital recency proved predictive, with churners having longer gaps since their last login, suggesting that inactivity can serve as an early warning signal.

<p align="center">
  <img src="/assets/recency_boxplot.png" alt="Recency by Churn Status">
</p>

- No single digital platform dominated customer engagement, offering an opportunity to improve user experience consistently across all channels.

## Model Diagnostics

Despite the relatively low AUC scores across all models, the XGBoost classifier demonstrated the highest discriminatory power. The confusion matrix showed high specificity but low recall—while the model was effective at identifying non-churners, it missed a majority of actual churners. This limitation presents an avenue for future improvement through model recalibration and ensemble strategies.

A visual summary of feature importance confirms that resolution status, number of interactions, and service access channel were the most influential variables. ROC curves further validate XGBoost as the best-performing model, albeit with room for optimization.

<p align="center">
  <img src="/assets/xgboost_confusion_matrix.png" alt="XGBoost Confusion Matrix">
</p>

<p align="center">
  <img src="/assets/xgboost_feature_importance.png" alt="XGBoost Feature Importance">
</p>

## Business Recommendations

To translate predictive insights into business value, several strategic initiatives are proposed:

1. **Real-time churn alert system:** Integrate predictive scores into SmartBank’s CRM to trigger early interventions via phone calls, in-app nudges, or tailored retention campaigns.
2. **Complaint-to-loyalty pipeline:** Establish a structured workflow to resolve complaints efficiently and follow up with goodwill gestures—such as waived fees or personalized incentives—to rebuild customer trust.
3. **Segment-specific engagement:** Design targeted offerings for freelancers, young professionals, and married customers. This may include startup loans, family-oriented benefits, and digital budgeting tools.
4. **Unified omnichannel experience:** Enhance consistency across mobile, web, and online banking to ensure frictionless and personalized customer interactions.
5. **Recency-based reactivation:** Launch re-engagement campaigns for users inactive for 90+ days, including limited-time offers or product feature highlights.
6. **Financial inclusion programs:** Develop fee-free, simplified accounts and personalized coaching tools for lower-income users to reduce financial stress and build loyalty.

Tracking KPIs such as churn reduction by segment, resolution times, and reactivation rates will be essential to validate the success of these initiatives.

## Future Work

The current model’s recall performance indicates the need for deeper exploration into ensemble modeling, temporal patterns, and advanced feature engineering. Additionally, the inclusion of qualitative customer feedback and third-party data sources could improve model granularity and accuracy. SHAP-based explainability tools may also be leveraged to enhance transparency and facilitate stakeholder buy-in.

You can click here to view the [report](Lloyds_Banking_Data_Science_Churn_Report.pdf). 



