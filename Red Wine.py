# import libraries

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.metrics import f1_score,recall_score,precision_score
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')


# import data

data= pd.read_csv(r"C:\Users\tawfi\Downloads\data\winequality-red.csv")


# clean data

def deal_with_nulls_and_duplicates( df , drop_na = False ):
    
    # handle with missing data 
    if drop_na is True:
        df.dropna(inplace=True)
    
    else:
        for col in df.columns:
        
            # drop columns that are more than half nan-values.
            if df[col].isna().sum()/len(df) > 0.5 :
                df.drop(columns=col , inplace = True)
        
            # fill nulls .
            elif df[col].dtype =="float":
                df[col] = df[col].fillna(df[col].mean())
            
            elif df[col].dtype =="int":
                df[col] = df[col].fillna(df[col].median())
            
            elif df[col].dtype =="object":
                df[col] = df[col].fillna(df[col].mode())
            
    # remove duplicated rows.
    df.drop_duplicates(inplace=True)
    
    return df

data = deal_with_nulls_and_duplicates(data ,drop_na= False)


def deal_with_outliers(df , columns = None):
    
    if columns is None:
        columns = df.select_dtypes(include=['number']).drop(columns= "quality").columns

    for col in columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        
        # replace the outliers with the lower/upper bound with the median or mean of the column
        if df[col].dtype =="int":
            
            df.loc[(df[col] < lower_bound) | (df[col] > upper_bound), col] = np.nan
            df[col].fillna(df[col].median(), inplace=True)
            
        elif df[col].dtype =="float":
            
            df.loc[(df[col] < lower_bound) | (df[col] > upper_bound), col] = np.nan
            df[col].fillna(df[col].mean(), inplace=True)
            
    return df.head()

deal_with_outliers(data,columns=['residual sugar','chlorides','volatile acidity'])

for col in ['fixed acidity','total sulfur dioxide','free sulfur dioxide','sulphates','alcohol']:
    data[col]=np.log(data[col])
    
    
    
# Explore target column

data["quality"].value_counts()

plt.figure(dpi=120)
ax = sns.countplot(data=data, x='quality', palette='CMRmap_r')
ax.bar_label(ax.containers[0], fmt='%.1f');

data["quality"] = (data["quality"] > 6.5).astype(int)
data["quality"].value_counts()


# Build model

# split data

target = "quality"
X = data.drop(columns=target)
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(
    X , y ,
    test_size=0.2,
    random_state=42
)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

#iteration

RF_classifer = RandomForestClassifier()
RF_classifer.fit(X_train,y_train)

RF_y_pred = RF_classifer.predict(X_test)

print('Model accuracy score with criterion entropy: {:0.2%}'. format(accuracy_score(y_test, RF_y_pred)))

# Hyperparameter Tuning
cv_acc_scores = cross_val_score(RF_classifer ,X_train ,y_train ,cv = 5 ,n_jobs=-1)
print(cv_acc_scores)

params = {
    "n_estimators":range(25,100,25),
    "max_depth":range(10,50,10)  
}
model = GridSearchCV(
    RF_classifer,
    param_grid=params,
    cv=5,
    n_jobs=-1,
    verbose=1
)

print(model.best_params_)
print("-------------")
print(model.best_score_)
print("-------------")
print(model.best_estimator_)


# evaluation

#Calculating the training and test accuracy scores for model.
acc_train = model.score(X_train,y_train)
acc_test = model.score(X_test,y_test)

print("Training Accuracy:", round(acc_train, 4))
print("Test Accuracy:", round(acc_test, 4))

y_test.value_counts()

# Create a confusion matrix
cm = confusion_matrix(y_test,model.predict(X_test))

# Plot confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# Calculate accuracy
accuracy = accuracy_score(y_test,model.predict(X_test))

# Calculate precision, recall, and F1-score
precision = precision_score(y_test, model.predict(X_test), average='weighted')
recall = recall_score(y_test, model.predict(X_test), average='weighted')
f1 = f1_score(y_test, model.predict(X_test), average='weighted')

# Create a classification report
report = classification_report(y_test, model.predict(X_test))

# Print evaluation metrics
print("Accuracy: {:.4f}".format(accuracy))
print("Precision: {:.4f}".format(precision))
print("Recall: {:.4f}".format(recall))
print("F1-score: {:.4f}".format(f1))

# Print classification report
print("Classification Report:\n", report)


#communication

# Get feature names from training data
features = X_train.columns

# Extract importances from model
importances = model.best_estimator_.feature_importances_

# Create a series with feature names and importances
feat_imp = pd.Series(importances,index=features).sort_values()

# Plot 10 most important features
feat_imp.tail(10).plot(kind="barh")
plt.xlabel("Gini Importance")
plt.ylabel("Feature")
plt.title("Feature Importance");