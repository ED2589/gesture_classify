# import packages
import pandas as pd
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, f1_score, recall_score

# cumulative importance plot for 1 participant

#load data
df= pd.read_csv('FtTblN_F01.csv')  # CAN CHANGE to a different feature table
y = df.iloc[:,0]  # subset truelabel column (response)
x = df[list(df.iloc[:, 1: df.shape[1]])]

# list of feature names
feat_list = list(x.columns)

# split F01 feature table into train-test data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3) # can change TEST SET SIZE - have more data means can afford larger test sizes

# fit random forest classifier (hyperparameters from previous grid search)
clf_RF = RandomForestClassifier(bootstrap=True, n_estimators=500, n_jobs=4, oob_score=True, class_weight='balanced',
                                    criterion='gini', max_depth=60, max_features='auto', max_leaf_nodes=None,
                                    min_samples_leaf=2, min_samples_split=2, min_weight_fraction_leaf=0.0,
                                    random_state=0, verbose=1, warm_start=False)
clf_RF.fit(X_train, y_train) # random forest training

# look into feature importances
importances = list(clf_RF.feature_importances_) # get list of importance scores for all features of F01

feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feat_list, importances)] # tuple of feature names + importance scores, rounded to 2 dec places

feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True) # sort features from most to least important

# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

############################################################################
##  barplot of feature importances (in order of most to least important)  ##
############################################################################

# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
sorted_importances = [importance[1] for importance in feature_importances] # importance scores in descending order
plt.bar(x_values,sorted_importances,orientation = 'vertical', color = 'r', edgecolor = 'k', linewidth = 1.2)
# Tick labels for x axis
sorted_features = [importance[0] for importance in feature_importances] # features in descending importance order
plt.xticks(x_values, sorted_features, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances')

###################################
##  cumulative importance graph  ##
###################################
cumulative_importances = np.cumsum(sorted_importances)
# Make a line graph
plt.plot(x_values, cumulative_importances, 'g-')
# Draw line at 95% of importance retained
plt.hlines(y = 0.90, xmin=0, xmax=len(sorted_importances), color = 'r', linestyles = 'dashed')
# Format x ticks and labels
plt.xticks(x_values, sorted_features, rotation = 'vertical')
# Axis labels and title
plt.xlabel('Variable'); plt.ylabel('Cumulative Importance'); plt.title('Cumulative Importances')

# find exact # of features needed to reach 90% cumulative importance
# Add 1 because Python is zero-indexed
print('Number of features for 90% importance:', np.where(cumulative_importances > 0.90)[0][0] + 1)

##
# from above graph, learned 64 out of 83 features required for 95% cumulative importance scores
##
# extract these features, then apply SVM
##
# Extract the names of the most important features
important_feature_names = [feature[0] for feature in feature_importances[0:72]]
# Find the columns of the most important features
important_indices = [feat_list.index(feature) for feature in important_feature_names]
# Create training and testing sets with only the important features
X_train_new = X_train.iloc[:, important_indices]
X_test_new = X_test.iloc[:, important_indices]
# Sanity check on operations
print('Important train features shape:', X_train_new.shape)
print('Important test features shape:', X_test_new.shape)

# apply SVM using new X_train and X_test sets (with selected features only)
clf_svm = svm.SVC(kernel='rbf',gamma= 0.1, C = 10 , decision_function_shape='ova')
clf_svm.fit(X_train_new, y_train)

y_pred_svm = clf_svm.predict(X_test_new)
f1_score_svm = f1_score(y_test, y_pred_svm, average='weighted')
recall_score_svm = recall_score(y_test, y_pred_svm, average='weighted')
print(f1_score_svm)
print(recall_score_svm)