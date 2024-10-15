# Fraud Detection with Sparkov

__This project developed an effective fraud detection model by utilizing advanced machine learning techniques. The model was designed to swiftly identify__
__fraudulent transactions while minimizing disruption to legitimate users. Multiple models were evaluated, with XGBoost ultimately selected for its balance__
__between precision and recall, making it the most effective solution for this task.__

# Key Steps
* __Data Preparation:__ Addressed class imbalance using the SMOTE technique and ensured consistent scaling and transformation of both training and validation datasets.
* __Feature Engineering:__ Created new features, such as transaction risk scores, to enhance the model's predictive capabilities.
* __Model Selection:__ Three models were evaluated—Logistic Regression, XGBoost, and Neural Networks—with XGBoost emerging as the best performer.
* __Hyperparameter Tuning:__ Utilized RandomizedSearchCV to optimize model parameters for improved performance.
* __Model Evaluation:__ Key metrics such as accuracy, precision, recall, F1 score, and AUC-ROC were used to compare model effectiveness. XGBoost achieved the highest precision and recall on unseen data.

Note: An HTML rendering of the project notebook can be viewed [here](https://nbviewer.org/github/databymir/fraud_sparkov/blob/main/sparkov.ipynb).

## Results
The XGBoost model was selected for its superior performance compared to Logistic Regression and Neural Networks. It achieved a precision of 0.81, recall of 0.84, an F1 score of 0.82, 
and an AUC-ROC of 0.92 on the test set. These metrics reflect the model’s ability to accurately detect fraudulent transactions while maintaining a balance between minimizing false 
positives and false negatives.

## Data
### Source Data
This dataset was downloaded from Kartik Shenoy's Kaggle, and is described as "a simulated credit card transaction dataset containing legitimate and fraud transactions from the duration 
1st Jan 2019 - 31st Dec 2020. It covers credit cards of 1000 customers doing transactions with a pool of 800 merchants" (Shenoy, 2020). Shenoy notes that he utilized the 
"Sparkov Data Generation | Github tool created by Brandon Harris" to run the simulation and convert the files to a standard format.

### Data Acquisition
The transaction data was acquired by downloading the two .csv files from Kaggle at https://www.kaggle.com/datasets/kartik2112/fraud-detection. 
Please note that the data files are not included in this repository due to size constraints but can be downloaded from Kaggle by following those steps.

Additionally, there is a stored_variables.pkl file referenced in the Jupyter notebook. The file contains saved variables such as metrics, SMOTE’d training data, and predictions. 
It was too large to include in this repository, but if you'd like to reproduce it, the relevant code is included in the notebook, currently commented out for efficiency. 
Simply uncomment the code, run the necessary sections, and the file will be recreated.

## Analysis Questions
1. What is the precision and recall for each model?
2. How does class imbalance affect model performance, and what techniques are most effective in addressing this?
3. What features have the highest impact on predicting fraudulent transactions?
4. How can hyperparameter tuning improve the balance between precision and recall?

## Data Analysis & Visualizations
Visualizations were created to illustrate the model’s performance, feature importance, and the distribution of fraudulent versus non-fraudulent transactions. 
These visualizations help provide insights into how the models work and how they can be optimized for future deployment.

## Installation
### Codes and Resources Used
* Python Version: 3.12.3
* Jupyter Notebook Version: 7.1.3

### Python Packages Used
#### Data Manipulation
* NumPy Version: 1.26.2
* Pandas Version: 2.1.4
* SciPy Version: 1.14.1

#### Data Visualization
* Matplotlib Version: 3.8.2
* Seaborn Version: 0.13.2
* IPython Version: 8.18.1

#### Machine Learning & Metrics
* Pickle (as part of standard Python)
* Joblib Version: 1.4.2
* Scikit-Learn Version: 1.5.2
* Imbalanced-Learn Version: 
* XGBoost Version: 2.1.1
* TensorFlow Version: 2.17.0
* SciKeras Wrappers Version: 0.13.0

## Code
├── sparkov.ipynb

├── models

│   └── best_log_clf.pkl

│   └── best_nn_clf.pkl

│   └── best_xgb_clf.pkl

├── README.md

├── LICENSE

└── .gitignore

## Authors 
[@databymir](https://github.com/databymir)

## References
Chawla, N. V., De Santo, S., & Davis, D. (2004). SMOTE: synthetic minority over-sampling technique. Journal of Artificial Intelligence Research, 16, 321-357. https://doi.org/10.1613/jair.1410

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785-794). ACM. https://doi.org/10.1145/2939672.2939785

Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. MIT Press. http://www.deeplearningbook.org/

Hosmer, D. W., & Lemeshow, S. (2000). Applied logistic regression (2nd ed.). John Wiley & Sons.

Robusto, C. C. (1957). The cosine-haversine formula. The American Mathematical Monthly, 64(1), 38-40.

Shenoy, Kartik. (2020, August 5). *Credit Card Transactions Fraud Detection Dataset.* Kaggle. https://www.kaggle.com/datasets/kartik2112/fraud-detection

Stevens, W. R. (1998). *Unix network programming, volume 1: The sockets networking API (2nd ed.)*. Prentice Hall.

## License
For this GitHub repository, the License used is [MIT License](https://opensource.org/license/mit/).