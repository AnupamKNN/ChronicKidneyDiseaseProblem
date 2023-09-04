###  Chronic Kidney Disease Problem

### Software & Tools Requiremntts

1. [Github Accounts](https://github.com)
2. [GitCLI](https://git-scm.com)
3. [Anaconda](https://www.anaconda.com)
4. [VS Code IDE](https://code.visualstudio.com)


Create a new environment
```
conda create -p venv python -y
```

Assign User Name
```
git config --global user.name "Your name"
```

Assign Email used to create Github Account
```
git config --global user.email "You@example.com"
```

Install ipykernel within the environment in the terminal before running jupyter notebook.
```
pip install ipykernel
```


### Introduction (Problem Statement):
In this project, we are required to build & analyze some of the best predictive models to predict whether the patient is suffering from chronic kidney disease or not.

### Dataset Description:
The dataset for this project was acquired from Kaggle. The dataset was uploaded using 'pd.read_csv' & a data frame 'chronic_df' was created.

- age		-	age	
- bp		-	blood pressure
- sg		-	specific gravity
- al		-   	albumin
- su		-	sugar
- rbc		-	red blood cells
- pc		-	pus cell
- pcc		-	pus cell clumps
- ba		-	bacteria
- bgr		-	blood glucose random
-	bu		-	blood urea
-	sc		-	serum creatinine
-	sod		-	sodium
-	pot		-	potassium
-	hemo		-	hemoglobin
-	pcv		-	packed cell volume
-	wc		-	white blood cell count
-	rc		-	red blood cell count
-	htn		-	hypertension
-	dm		-	diabetes mellitus
-	cad		-	coronary artery disease
-	appet		-	appetite
-	pe		-	pedal edema
-	ane		-	anemia
-	class		-	class	
- ##################################################
%
- Number of Attributes: 24 + class = 25 ( 11  numeric ,14  nominal) 
%
<h3>Attribute Information with their measuring units:</h3>
- 1.Age(numerical)
  	age in years

- 2.Blood Pressure(numerical)
	  bp in mm/Hg
-	3.Specific Gravity(nominal)
		sg - (1.005,1.010,1.015,1.020,1.025)

- 4.Albumin(nominal)
		al - (0,1,2,3,4,5)

-	5.Sugar(nominal)
		su - (0,1,2,3,4,5)

- 6.Red Blood Cells(nominal)
		rbc - (normal,abnormal)

- 7.Pus Cell (nominal)
		pc - (normal,abnormal)

- 8.Pus Cell clumps(nominal)
		pcc - (present,notpresent)

- 9.Bacteria(nominal)
		ba  - (present,notpresent)

- 10.Blood Glucose Random(numerical)		
		bgr in mgs/dl

- 11.Blood Urea(numerical)	
		bu in mgs/dl

- 12.Serum Creatinine(numerical)	
		sc in mgs/dl

- 13.Sodium(numerical)
		sod in mEq/L

- 14.Potassium(numerical)	
		pot in mEq/L

- 15.Hemoglobin(numerical)
		hemo in gms

- 16.Packed  Cell Volume(numerical)

- 17.White Blood Cell Count(numerical)
		wc in cells/cumm

- 18.Red Blood Cell Count(numerical)	
		rc in millions/cmm

- 19.Hypertension(nominal)	
		htn - (yes,no)

- 20.Diabetes Mellitus(nominal)	
		dm - (yes,no)

- 21.Coronary Artery Disease(nominal)
		cad - (yes,no)

- 22.Appetite(nominal)	
	 appet - (good,poor)

- 23.Pedal Edema(nominal)
		pe - (yes,no)	

- 24.Anemia(nominal)
		ane - (yes,no)
		  
- 25.Class (nominal)		
		class - (ckd,notckd)

### EDA on the dataset.

1. 'chronic_df.info()' was used to analyze the information regarding the feature data types.
2. 'chronic_df.isnull().sum()' was used to return the count of all the missing values in all the features. It was found that several features had very few missing values while some features had many values missing.

### Data Cleaning

1. 'id' column was dropped from 'chronic_df' as it was irrelevant.
2. Column names in the data frame were renamed for better understanding.
3. Columns (packed_cell_volume, white_blood_cell_count and red_blood_cell_count) was required to be converted to numerical since it was object right then.
4 Columns (specific_gravty, albumin and sugar) should be nominal, so it was converted from float to object.
5. Categorical columns were extracted from the data frame & were stored in cat_cols.
6. Similarly, numerical columns were extracted from the data frame & were stored in num_cols
7. A for loop was used to capture unique values of every categorical columns in cat_cols.
8. Certain incorrect categorical values like \tno', '\tyes', ' yes', '\tno', 'ckd\t' were replaced
9. Target column values were changed to {'ckd': 0, 'notckd': 1} & was transformed to numeric to analyze the correlation.

### Exploratory Data Analysis (EDA)

1. A for loop was used to iterate overe the num_cols & distribution plots were plotted to check the distribution of values of all the features.

![Plot 1](https://github.com/AnupamKNN/ChronicKidneyDiseaseProblem/blob/main/Plots/age.png)

![Plot 2](https://github.com/AnupamKNN/ChronicKidneyDiseaseProblem/blob/main/Plots/blood_glucose_random.png)

![Plot 3](https://github.com/AnupamKNN/ChronicKidneyDiseaseProblem/blob/main/Plots/blood_pressure.png)

![Plot 4](https://github.com/AnupamKNN/ChronicKidneyDiseaseProblem/blob/main/Plots/blood_urea.png)

![Plot 5](https://github.com/AnupamKNN/ChronicKidneyDiseaseProblem/blob/main/Plots/haemoglobin.png)

![Plot 6](https://github.com/AnupamKNN/ChronicKidneyDiseaseProblem/blob/main/Plots/packed_cell_volume.png)

![Plot 7](https://github.com/AnupamKNN/ChronicKidneyDiseaseProblem/blob/main/Plots/potassium.png)

![Plot 8](https://github.com/AnupamKNN/ChronicKidneyDiseaseProblem/blob/main/Plots/red_blood_cell_count.png)

![Plot 9](https://github.com/AnupamKNN/ChronicKidneyDiseaseProblem/blob/main/Plots/serum_creatinine.png)

![Plot 10](https://github.com/AnupamKNN/ChronicKidneyDiseaseProblem/blob/main/Plots/sodium.png)

![Plot 10](https://github.com/AnupamKNN/ChronicKidneyDiseaseProblem/blob/main/Plots/white_blood_cell_count.png)


2. Count plots were plotted to analyze the categorical column distribution.

![plot 11](https://github.com/AnupamKNN/ChronicKidneyDiseaseProblem/blob/main/Plots%202/aanemia.png)


3. Correlation matrix was plotted using heatmap to analyze the correlation of all the features with the target column.

![plot 12](https://github.com/AnupamKNN/ChronicKidneyDiseaseProblem/blob/main/Plots/class.png)

4. Null values in both cat_cols & numm_cols were checked & sorted in descending order to analyze the features having count of null values from highest to lowest.

### Missing Value Treatment

1. For filling null values, two methods were used:-
     - Random Sampling for higher count of null values
     - Mean/Mode sampling for lower count of null values
2. Filled num_cols null values using random sampling method.
3. Filled "red_blood_cells" and "pus_cell" using random sampling method and rest of categorical columns using mode imputation.
4. Check the count of unique values in each categorical column by looping over cat_cols.
5. Used labelencoder and applied on categorical column.
6. Split data into features and target variables (X and y)
7. Split data into training and test set using (train_test_split) from sklearn.model_selection with test_size = 30%.

### Model Building
1. Various classification type machine learning models were imported & were evaluated using accuracy_score, confusion_matrix & classification_report.
2. All the models were trained & tested for their training & testing accuracy.
3. The training accuracy of each model is as bellow:
  - K-Neighbours Classifier:- Training Accuracy: 76.42%, Test Accuracy: 62.5%
  - Decision Tree Classifier (before hyperparameter tuning):- Training Accuracy: 100%, Test Accuracy: 99.17%
  - Decision Tree Classifier (after hyperparemeter tuning):- Training Accuracy: 97.14%, Test Accuracy: 95.84%
  - Random Forest Classifier:- Training Accuracy: 100%, Test Accuracy: 99.17%
  - Ada Boost Classifier:- Training Accuracy: 100%, Test Accuracy: 100%
  - Gradient Boosting Classifier:- Training Accuracy: 100%, Test Accuracy: 99.17%
  - Stochastic Gradient Boosting Classifier:- Training Accuracy: 100%, Test Accuracy: 99.17
  - XGB Classifier:- Training Accuracy: 100%%, Test Accuracy: 99.17%
  - Cat Boost Classifier:- Training Accuracy: 100%, Test Accuracy: 99.17
  - Extra Trees Classifier:- Training Accuracy: 100%, Test Accuracy: 100%
  - LGBM Classifier:- Training Accuracy: 100%, Test Accuracy: 99.17%

After analyzing the train & test accuracy scores of the above listed models, one can select almost any model with accuracies within the range (95%-100%).
