# credit-risk-classification
Supervised ML - UNC Bootcamp Challenge 20
## Overview of the Analysis

The aim of this analysis was to deploy machine learning techniques to forecast the credit risk associated with loans. The dataset was composed of financial details from borrowers, and the primary objective was to identify which loans had a high probability of defaulting. This is essential for financial institutions to both limit the occurrence of bad loans and to ensure that creditworthy customers are granted loans.

The principal variable for prediction was a binary classification denoting `healthy` for low-risk loans and `high-risk` for those with a high likelihood of default. Initially, the data showed a significant imbalance between these two classes, which is typical in risk modeling.

The machine learning process involved several key steps: preprocessing the data, selecting and training models, and evaluating their performance. Logistic Regression model was implemented on the original data, and to address the imbalance in the dataset, Random OverSampling (ROS) was applied, and another Logistic regression model was implemented using the resampled data. Model performance was gauged using metrics such as Balanced Accuracy, Precision, and Recall.

---

## Results

The outcomes for each machine learning model are summarized below:

### Logistic Regression using original data:

- **Balanced Accuracy Score**: 0.944
A score of 0.944 is a strong score and implies that the model is performing well, but there is an imbalance in performance between the classes, likely skewed towards the majority class (healthy loans).

- **Confusion Matrix (Original Model)**

|                   | Predicted healthy | Predicted high-risk |
|-------------------|-------------------|---------------------|
| Actual healthy    | 18679             | 80                  |
| Actual high-risk  | 67                    | 558                |

    
    - True Positives (TP): The cases in which the model correctly predicted 'high-risk'. Here, TP is 558.
    - True Negatives (TN): The cases in which the model correctly predicted 'healthy'. Here, TN is 18679.
    - False Positives (FP): The cases in which the model incorrectly predicted 'high-risk' when they were actually 'healthy'. Here, FP is 80.
    - False Negatives (FN): The cases in which the model incorrectly predicted 'healthy' when they were actually 'high-risk'. Here, FN is 67.


- **Classification Report (Original Model)**

|                | precision | recall | f1-score | support |
|----------------|-----------|--------|----------|---------|
| healthy loan   | 1.00      | 1.00   | 1.00     | 18759   |
| high-risk loan | 0.87      | 0.89   | 0.88     | 625     |
| **accuracy**       |           |        | 0.99     | 19384   |
| **macro avg**      | 0.94      | 0.94   | 0.94     | 19384   |
| **weighted avg**   | 0.99      | 0.99   | 0.99     | 19384   |

    - Precision (healthy): Almost perfect at 1.00, meaning nearly all the predictions of 'healthy' are correct.
    - Precision (high-risk): At 0.87, this indicates that when the model predicts 'high-risk', it is correct 87% of the time.
    - Recall (healthy): Also almost perfect at 1.00, meaning the model is capturing almost all of the actual 'healthy' cases.
    - Recall (high-risk): At 0.89, this suggests that of all the actual 'high-risk' cases, the model correctly identifies 89% of them.
    - F1-Score: This is the harmonic mean of precision and recall. For 'healthy', it is 1.00, which is excellent. For 'high-risk', it is 0.88, which is good and indicates a balanced performance between precision and recall for this class.
    - Support: This column indicates the number of actual occurrences of each class in the dataset. There were 18759 'healthy' cases and 625 'high-risk' cases.
    - Accuracy: The overall accuracy of the model is 0.99, meaning it correctly predicted 99% of the total cases.

- **Conclusion (Original Model)**
The model performs exceptionally well at identifying 'healthy' cases, with almost perfect precision and recall. It performs well for 'high-risk' cases too, with good precision and recall, although not as high as for 'healthy' cases. This is expected because it is often more challenging to correctly identify the minority class in an imbalanced dataset (there are far more 'healthy' instances than 'high-risk'). The F1-score for 'high-risk' at 0.88 is quite good, indicating that the model has a good balance between precision and recall for this class. The model's overall accuracy is excellent. However, we may be particularly concerned with the model's performance on the 'high-risk' class due to the potential consequences of false negatives. The number of False Negatives (67) is concerning, as these are high-risk cases that were not identified.
In summary, the model is highly accurate and has strong performance metrics. However, the false negatives for the 'high-risk' category could be a concern, especially in contexts where missing a 'high-risk' classification has serious implications.
---
### Logistic Regression after ROS:

- **Balanced Accuracy Score**: 0.996
Balanced Accuracy Score is approximately 0.996, which is extremely high. This score suggests that the model is performing exceptionally well at correctly identifying both 'healthy loans' and 'high-risk loans', balancing out the performance across both classes.

- **Confusion Matrix (Resampled Model)**

|                   | Predicted healthy | Predicted high-risk |
|-------------------|-------------------|---------------------|
| Actual healthy    | 18668             | 91                  |
| Actual high-risk  | 2                 | 623                 |

    - True Positives (TP) for high-risk loans: Almost perfect at 623. This is an increase from 558 in the previous confusion matrix, indicating that more high-risk loans are being correctly identified after resampling.
    - True Negatives (TN) for healthy loans: Slightly decreased to 18668 from 18679, which is still a very high number of correct predictions for healthy loans.
    - False Positives (FP) for high-risk loans: Increased to 91 from 80, which means there are more cases where healthy loans are incorrectly labeled as high-risk.
    - False Negatives (FN) for high-risk loans: Dramatically reduced to 2 from 67, indicating that the model is now identifying almost all high-risk loans correctly.
- **Classification Report (Resampled Model)**

|                | precision | recall | f1-score | support |
|----------------|-----------|--------|----------|---------|
| healthy loan   | 1.00      | 1.00   | 1.00     | 18759   |
| high-risk loan | 0.87      | 1.00   | 0.93     | 625     |
| **accuracy**       |           |        | 1.00     | 19384   |
| **macro avg**      | 0.94      | 1.00   | 0.96     | 19384   |
| **weighted avg**   | 1.00      | 1.00   | 1.00     | 19384   |

    - Precision for high-risk loans: Remained the same at 0.87, meaning when the model predicts a loan as high-risk, it is correct 87% of the time.
    - Recall for high-risk loans: Increased to 1.00, meaning the model now identifies all high-risk loans.
    - F1-Score for high-risk loans: Improved to 0.93 from 0.88, showing a better balance between precision and recall after resampling. It's closer to the perfect score of 1, which indicates a high precision and a high recall.
    - Accuracy: Remains very high, at practically 1.00.

- **Conclusion (Resampled Model)**
The resampling technique has been extremely effective in improving the recall for the high-risk loan predictions without sacrificing precision. While the number of False Positives increased slightly, this is often an acceptable trade-off for such a significant decrease in False Negatives in cases where the cost of missing a high-risk loan is high. The F1-score's increase for high-risk loans indicates that the balance between precision and recall has become more favorable. The overall accuracy of the model has been maintained, which is an excellent outcome, as the increase in accuracy for the minority class did not come at the expense of the majority class accuracy.
In conclusion, the model now performs much better at identifying high-risk loans, which are often the most critical to detect in financial risk modeling. The slight increase in false positives may be an acceptable compromise, especially if the cost associated with failing to detect a high-risk loan is significantly greater than the cost of a false alarm.

---
## Summary

The analysis demonstrates that the application of Random OverSampling significantly enhanced the performance of the Logistic Regression model. The Balanced Accuracy Score saw a considerable rise from 0.944 to 0.996 after addressing the class imbalance. Most notably, the recall for high-risk loans achieved a perfect score of 1.00 following ROS, which is particularly significant for financial institutions that aim to minimize losses associated with high-risk loans.

While precision for high-risk loans remained constant across both models, the improved recall suggests that ROS was effective in overcoming the challenge posed by the imbalanced dataset. Consequently, the use of Logistic Regression post-ROS is recommended for the prediction of loan risk. It provides a robust approach to identifying high-risk loans, which is a critical factor for financial stability.

## Conclusion

In the task of predicting loan risk, the performance priorities can vary depending on the financial institution's goals. If reducing default risk is paramount, a high recall for high-risk loans is critical to capture as many potential defaults as possible. Conversely, if the institution aims to grow its loan portfolio, precision in identifying healthy loans is essential to avoid turning away good business.

In our analysis, the second model (post-ROS) demonstrated superior performance with a Balanced Accuracy Score of 0.996 and a perfect recall of 1.00 for high-risk loans, without sacrificing precision. This suggests that the model effectively identifies high-risk loans while maintaining reliability in classifying healthy loans.

Given the comprehensive improvement, particularly in recall for high-risk loans, I recommend the usage of the second model. This recommendation is based on its ability to better identify all high-risk loans, a crucial aspect for minimizing risk in loan portfolios.

## Sources
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Imbalanced-learn Documentation](https://imbalanced-learn.org/stable/)

## References
Data for this dataset was generated by edX Boot Camps LLC, and is intended for educational purposes only.

