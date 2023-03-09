# **Titanic Survivors Challenge: a look into the payoff of different strategies**
In this notebook, strategies to improve the accuracy of the *submission* predictions for the [Titanic challenge](https://www.kaggle.com/competitions/titanic/overview) are discussed, with focus on how much an action improved upon the previous score, and which efforts are most beneficial, considering the added development and computational time required.

## **Aspects explored:**
1. Effects of **model selection**
2. Effects of **hyperparameters tuning**
3. Effects of **features engineering**

## **Workflow and results:**
## 1. **Model screening** 

**Minimal preprocessing** and **default model parameters** are considered for a rapid and inexpensive preliminary output, which will be the benchmark for comparison with later refinements. Preprocessing is done using a **DataProcessing class** that can be found in the utility scripts. 
Preprocessing choices:
* Empty values in **"Age"** and **"Fare"** are filled with the median. 
* Uniform labels are given for the missing values in **"Cabin"** and **"Embarked"**.
* **"Sex"** and **"Embarked"** are encoded with value between 0 and n_classes-1.
* At this stage, **"Name"**, **"Ticket"**, **"Cabin"**, **"PassengerId"** are dropped.

Please check out **[Titanic. 40+ models](https://www.kaggle.com/code/grigol1/titanic-40-models)** for a detailed description on how to use the **all_estimators** method to query the scikit-learn library to list all available estimators of various families (here classificators). 

A quick prediction with default hyperparameters is computed with the available classificators in the scikit-learn library. For the **15 best scoring** ones, the accuracy obtained on the unseen **submission set** was obtained and reported below. On average, the accuracy was 7% worse on the submission set than on the modelling set.

Best **result of model screening** obtained with **out-of-the box** configurations and minimal preprocessing of the dataset: **<span style="color:green">0.79186</span>**  with **StackingClassifier**, which is an ensemble model combining predictions from various estimators. Here RandomForestClassifier and LogisticRegression were included in the StackingClassifier. 

Training time: **38 s** for 40 models.

|Model |Cross Validation|Accuracy|Acc. on Submission |
|------|----------------|--------|-------------------|
|HistGradientBoostingClassifier|0.833910|0.852020|0.779900|
|GradientBoostingClassifier|0.821570|0.847530|0.677940|
|AdaBoostClassifier|0.811500|0.843050|0.765550|
|MLPClassifier|0.789030|0.843050|0.760760|
|BaggingClassifier |0.805920|0.843050|0.746410|
|CalibratedClassifierCV|0.793500|0.838570|0.746410|
|**StackingClassifier**|0.817070|0.829600|**0.79186**|
|RandomForestClassifier|0.812600|0.829600|0.765550|
|OutputCodeClassifier |0.764280|0.829600 |0.765550|
|OneVsRestClassifier|0.812600|0.829600|0.765550|
|MultiOutputClassifier |0.812600|0.829600|0.765550|
|QuadraticDiscriminantAnalysis|0.795730 |0.825110 |0.765550|
|LogisticRegressionCV |0.786760|0.820630|0.765550|
|LogisticRegression|0.790130|0.820630 |0.763150|
|ClassifierChain|0.790130|0.820630 |0.772720|

## 2. **Hyperparameters optimization**

For **4 selected models** (*LogisticRegressionCV*, *MLPClassifier*, *RandomForestClassifier*, *HistGradientBoostingClassifier*) hyperparmeters (hp) tuning was performed. The **[Scikit-Optimize](https://scikit-optimize.github.io/stable/auto_examples/sklearn-gridsearchcv-replacement.html)** library was used, which allows to make the search in the hp space more efficient and converge faster to an improved solution with respect to a grid search method. A maximum of **50 hyperparameter optimization cycles** per estimator were considered, for a total of 200 optimization runs.

For selected cases, hyperparmeters (hp) tuning was performed using **[Scikit-Optimize](https://scikit-optimize.github.io/stable/auto_examples/sklearn-gridsearchcv-replacement.html)**. This allows to optimize the search in the hp space and converge faster to an improved solution with respect to a grid search method. 
The **get_params()** method can be used to get a list of all the parameters of a model, to choose which ones to optimize. The choice of hyperparameters to optimize an the value ranges tested was guided by the Documentation of each Estimator. The number of hyperparameter **optimization attempts was limited to 50 per model**, making 200 cross validation cycles overall. 

Results of **[hyperparameter tuning](#section_id4)** test:

| Model         | Init. acc. val. set  | Init. acc. subm. set|  Opt. acc. val. set| Opt. acc. subm. set | Time (min) | %diff. subm. acc.|
|---------------|----------------------|-------------------|---------------------|--------------------|------------|---------|
| MLPClassifier                   | 0.84305    | 0.86547 | 0.76076| **0.77990** | 7 | <span style="color:green">+3%</span>  |
| LogisticRegressionCV            | 0.82063    | 0.83857| 0.76315 | 0.77033 | 12 | <span style="color:green">+1%</span>  |
| RandomForestClassifier          | 0.82960    | 0.85202| 0.76555 | 0.77033 | 4 | <span style="color:green">+1%</span>  |
| HistGradientBoostingClassifier  | **0.85202** | **0.89238**| **0.77990**  | 0.75119 | **2** | <span style="color:red">-4%</span>  |

The out-of-the box performance is only slightly improved by hyperparameter tuning, with one instance where the performance on the submission set actually worsens, which points to **overfitting issues**. Hyperparameters tuning comes at a **significantly higher computational cost**, without yielding a proportional benefit in terms of increased accuracy, therefore it did not prove to be an efficient strategy.

## 3. **Feature engineering**
Only 2 values are missing for the *Embarked* feature in the modelling set and 1 for *Fare* in the submission set, therefore the choice of how to handle the missing cases will most likely not have an appreciable effect on the final results, and therefore it will not be investigated. 

In the case of the **Age** though, we are inferring 177 out of 891 data points (approx. **20%**!), therefore this feature merits more attention. From the correlation of Age with the other features (Pearson correlation coefficient - [jump to code](#section_id4)), the relationships evidenced are:
* an **inverse relationship with Pclass, SibSp, and Parch**: the younger the person, the higher the class (younger less wealthy people travelling more likely in 3rd or 2nd class), and the higher number of family members travelling together.
* a **positive correlation with Sex and Fare**: the older the person, the higher the Fare (older, more wealthy people, travelling in 1st class more likely) and the more likely they are to be a man (Male is encoded to 1 and Female is encoded to 0).

### 3.1 **Train a regressor for Age**

A series of regressors were traiend to predict Age based on a series of independent variables and compared, repeating the test three times considering a different number of features:
1. Test dropping: ["Name", "Ticket", "Cabin", "PassengerId", "Embarked"]
2. Test dropping: **["Name", "Ticket", "Cabin", "PassengerId", "Embarked", "Fare", "Sex"]**
3. Test dropping: ["Name", "Ticket", "Cabin", "PassengerId", "Embarked", "Fare", "Sex", "Parch"]

Test #2 yielded the best results in terms of test set accuracy, and a **GradientBoostingRegressor()** was trained to predict and fill in the missing Age values. 

The Survivors predictions with this new feature **did not yield an improvement**, as can be seen in the following table:

| Model         | Old. acc. val. set  | New. acc. val. set| Old. acc. subm. set | New. acc. subm. set | %diff. subm. acc.|
|---------------|----------------------|-------------------|---------------------|--------------------|---------|
| MLPClassifier                   | 0.84305     | 0.84305      | 0.76076 | 0.76076 | same  |
| LogisticRegressionCV            | 0.82063     | 0.82960      | 0.76315 | 0.76076 |  <span style="color:red">-0.3%</span>  |
| RandomForestClassifier          | 0.82960     | 0.83408      | 0.76555 | 0.76794 | <span style="color:green">+0.3%</span>  |
| HistGradientBoostingClassifier  | 0.85202 | **0.85202**  | 0.77990 | 0.76315 |  <span style="color:red">-2%</span>  |
| StackinClassifier               |**0.829600**|0.83857| **0.79186** | 0.76315 |  <span style="color:red">-4%</span>  |


### 3.2 **Feature manipulation**

Various strategies were tested to improve the relevance of the features considered. As a guiding criterion, the **Pearson correlation coefficient** between the various features and the survival rate was evaluated. 
* Group **age by ranges**: this significantly increased (by a factor 2) the correlation of the age with the survival rate, and it was **sensitive to the interval limits** chosen.
* Create a feature for **single travellers** (**"Alone"**) and for the total family size ("**FamilySize**") as the sum of "SibSp" and "Parch". The Family size did not prove to be useful, because actually the correlation between "SibSp" and "Parch" with Survival is of the opposite sign. Alone, however, displayed a significant correlation with Survival
* The salutation **title** of the passengers was also analyzed but it did not display a significant correlation with Survival
* An attempt was made also to create **FareRanges**, but it was did not translate into an increase in the correlation beyond the one already achieved with Fare.

Finally, two tests were performed:
* Test 1 : "More features" -  **dropping ["Name", "Ticket", "Cabin", "PassengerId"]** 
* Test 2 : "Fewer features" -  **dropping *additionally* ["CombinedFamily", "Age", "Parch", "SibSp", "Title"]** 

The **MLPClassifier**, **HistGradientBoostingClassifier** and **StackinClassifier** were tested, using for their hyperpatemers the **values** resulted **from the hyperparameter optimization** performed previously.
The following results were obtained. Interestingly, sometimes "**less is more**": with the exclusions of some features which had a lower correlation with Survival, the submission accuracy increased for 2 out of 3 models.

| Model         | Acc. more features  | Subm. acc. more features| Acc. fewer features | Subm. acc. fewer features | %diff. subm. acc.|
|---------------|----------------------|-------------------|---------------------|--------------------|---------|
| MLPClassifier                   | 0.83408     |   0.75358      | 0.82511 |  0.76076 | <span style="color:green">+1%</span>  |
| HistGradientBoostingClassifier  | 0.82960     |   0.77033      | 0.82063 |  <span style="color:green">**0.78229** </span>|  <span style="color:green">+2%</span>  |
| StackingClassifier               |**0.84753**  | **0.77751** | **0.81166** | 0.77751 |  same  |

## 4. **Conclusions**

Testing out different angles to attempt to improve the prediciton accuracy for the Titanic dataset was an interesting and fruitful learning opportunity, to experiment with **model selection**, **hp tuning**, and **feature engineering** and familiarize with how these concepts are applied within the scikit-learn library. I hope that this Notebook can be helpful to other learners, as many others have been for me! 

Happy modelling! :D 
