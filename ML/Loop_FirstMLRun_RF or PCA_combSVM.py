# created June 5 2019
# template see 'F01_FirstMLRun_RF or PCA_combSVM.py'
# Loop above algorithm over ALL 19 feature tables (note: # of features in each ftTbl will be different - Alex email June 5)

# params used:
# test size = .3 ; RF features selected using mean decrease in impurity (instead of permutation importance); metric of feat select is top 20 features
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support,confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
import seaborn as sn
import matplotlib.pyplot as plt

csvfile_list = ['FtTblN_F01.csv','FtTblN_F02.csv', 'FtTblN_F03.csv' , 'FtTblN_F04.csv' , 'FtTblN_F05.csv' , 'FtTblN_F06.csv', 'FtTblN_F07.csv',
           'FtTblN_F08.csv', 'FtTblN_F09.csv', 'FtTblN_F10.csv','FtTblN_P12.csv','FtTblN_P43.csv', 'FtTblN_P52.csv' , 'FtTblN_P63.csv' ,
                'FtTblN_P81.csv' , 'FtTblN_P84.csv', 'FtTblN_P96.csv','FtTblN_P97.csv', 'FtTblN_P98.csv']
df_ftbl_list = []
for file in csvfile_list:
    # read in feature table
    df_ftbl_list.append(pd.read_csv(file))

def count_tl_freq(df):
    # counts truelabel frequencies (counts for each label classes 1,2,3)
    return (df.iloc[:,0].value_counts())
tl_freq_list = []
for df in df_ftbl_list:
    tl_freq_list.append(count_tl_freq(df))

###############
## Alt 1: PCA + SVM ##
###############
def SplitTrainTest_GetFeat_PCA(df,test_size, percent_variance):
    '''
    splits FtTbl into test and training features/tls and performs Principal Component analysis to features to reduce dimension of features so that
    only features that explain 'percent_variance' amount of the model is included in transformed feature df
    :param df: ftbl as df
    :param test_size: test data size
    :param percent_variance: % variance of mode explained by features selected
    :return: a list of outputs related to pca: [pca-transformed X_train df , pca_transformed X_test df ,
     original y_train, original y_test, % variance explained by each pc], the first 4 elements of the list will be fed into SVM
    '''

    y = df.iloc[:, 0]  # subset truelabel column (response)
    x = df[list(df.iloc[:, 1: df.shape[1]])]
    X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(x, y, test_size=test_size)
    pca = PCA(percent_variance)
    #pca = PCA(n_components=percent_variance)
    pca.fit(X_train_pca)
    X_train_pca = pca.transform(X_train_pca)
    X_test_pca = pca.transform(X_test_pca)  # apply SAME mapping (transform) to test set as the training set
    return [X_train_pca, X_test_pca, y_train_pca, y_test_pca, pca.explained_variance_ratio_]

a=[]
for df in df_ftbl_list:
    a.append(SplitTrainTest_GetFeat_PCA(df,0.3,0.95))
#for i in range(len(a)):
#    print (a[i][0].shape[1] == a[i][1].shape[1])
pca_df_X_train_list = []
for i in range(len(a)):
    pca_df_X_train_list.append(a[i][0])
pca_df_X_test_list = []
for i in range(len(a)):
    pca_df_X_test_list.append(a[i][1])

#for df1,df2 in zip(pca_df_X_train_list,pca_df_X_test_list):
#    print (df1.shape[1] == df2.shape[1])

orig_df_y_train_list = []
for i in range(len(a)):
    orig_df_y_train_list.append(a[i][2])
orig_df_y_test_list = []
for i in range(len(a)):
    orig_df_y_test_list.append(a[i][3])
percVariance_pca_list = []
for i in range(len(a)):
    percVariance_pca_list.append(a[i][4])

 # want LOW standard deviation because that means less risk of overfitting aka uniform prediction accuracy across ANY test set given

def svm_classify_afterPCA(X_train_pca,X_test_pca,y_train_pca,y_test_pca,kernel, gamma, Cparam, decision_function_shape, degree = 3 ):
    y_pred_svm = pd.DataFrame()
    crossval = np.empty([1, 10])
    if kernel =='rbf':

        clf_svm = svm.SVC(kernel=kernel, gamma=gamma, C=Cparam, decision_function_shape=decision_function_shape)
        clf_svm.fit(X_train_pca, y_train_pca)
        crossval= cross_val_score(estimator=clf_svm, X=X_train_pca, y=y_train_pca, cv=10)
        y_pred_svm = clf_svm.predict(X_test_pca)

    if kernel == 'poly':
        clf_svm = svm.SVC(kernel=kernel, gamma=gamma, C=Cparam, decision_function_shape=decision_function_shape, degree = degree)
        clf_svm.fit(X_train_pca, y_train_pca)
        crossval= cross_val_score(estimator=clf_svm, X=X_train_pca, y=y_train_pca, cv=10)
        y_pred_svm = clf_svm.predict(X_test_pca)
    classify_report = classification_report(y_test_pca,y_pred_svm, output_dict=True)
    clf_svm_matrix = pd.crosstab(y_test_pca, y_pred_svm, rownames=['Actual Result'], colnames=['Predicted Result'])
    # clf_svm_norm_matrix = clf_svm_matrix / clf_svm_matrix.astype(np.float).sum(axis = 1)
    f1_score_svm = f1_score(y_test_pca, y_pred_svm, average='weighted') # 'weighted' OR 'micro'
    return [clf_svm_matrix, f1_score_svm, classify_report]

def crossval_svm_afterPCA(X_train_pca,y_train_pca,gamma, Cparam, decision_function_shape, folds):
    clf_svm = svm.SVC(kernel='rbf', gamma=gamma, C=Cparam, decision_function_shape=decision_function_shape)
    clf_svm.fit(X_train_pca, y_train_pca)
    crossval = cross_val_score(estimator=clf_svm, X=X_train_pca, y=y_train_pca, cv=folds)
    return [crossval.mean(),crossval.std()]
#
svm_afterPCA_confMatrix_list = []
svm_afterPCA_f1Score_list  = []
svm_afterPCA_classreport_list = []
#
svm_afterPCA_crossval_mean = []
svm_afterPCA_crossval_std = []
#
for df1,df2,df3,df4 in zip(pca_df_X_train_list,pca_df_X_test_list,orig_df_y_train_list,orig_df_y_test_list):
    svm_afterPCA_confMatrix_list.append(svm_classify_afterPCA(df1,df2,df3,df4,'rbf',0.02,4,'ovo')[0])

    svm_afterPCA_f1Score_list.append(svm_classify_afterPCA(df1,df2,df3,df4,'rbf',0.02,4,'ovo')[1])

    svm_afterPCA_classreport_list.append(svm_classify_afterPCA(df1,df2,df3,df4,'rbf',0.02,4,'ovo')[2])
for df1,df2 in zip(pca_df_X_train_list,orig_df_y_train_list):
    svm_afterPCA_crossval_mean.append(crossval_svm_afterPCA(df1,df2, 0.02,4,'ovo',10)[0]) # list of crossval means
    svm_afterPCA_crossval_std.append(crossval_svm_afterPCA(df1, df2, 0.02, 4, 'ovo', 10)[1])  # list of crossval stds # lower means less risk of overfitting aka
                                                                                            # uniform classififier performance across ANY test sets
np.mean(svm_afterPCA_f1Score_list)  # 82 % for 'weighted' f1
print(svm_afterPCA_crossval_mean) # 10-fold cross validation mean scores for each of 19 Ftbls
print(svm_afterPCA_crossval_std)  # 10-fold cross validation std dev for each of 19 Ftbls

lst11 = [] # recall for class 1
lst22 = [] # ' '    for class 2
lst33 = [] # ' '    for class 3
lstAA = [] # f1-score for class 1
lstBB = [] # ' '       for class 2
lstCC = [] # '  '      for class 3
lstaa= []
lstbb = []
lstcc = []
for i in range(len(svm_afterPCA_classreport_list)):
    lst11.append(svm_afterPCA_classreport_list[i]['1']['recall'])
    lst22.append(svm_afterPCA_classreport_list[i]['2']['recall'])
    lst33.append(svm_afterPCA_classreport_list[i]['3']['recall'])
    lstAA.append(svm_afterPCA_classreport_list[i]['1']['f1-score'])
    lstBB.append(svm_afterPCA_classreport_list[i]['2']['f1-score'])
    lstCC.append(svm_afterPCA_classreport_list[i]['3']['f1-score'])
    lstaa.append(svm_afterPCA_classreport_list[i]['1']['support'])
    lstbb.append(svm_afterPCA_classreport_list[i]['2']['support'])
    lstcc.append(svm_afterPCA_classreport_list[i]['3']['support'])


results_table11 = pd.DataFrame(
    {'Gesture 1 accuracy (%)': lst11,
     'Gesture 2 accuracy (%)': lst22,
     'Gesture 3 accuracy (%)': lst33})
results_table22 = pd.DataFrame(
    {'Gesture 1 f1-score (%)': lstAA,
     'Gesture 2 f1-score (%)': lstBB,
     'Gesture 3 f1-score (%)': lstCC})

list_particips = ['F01', 'F02', 'F03' , 'F04' , 'F05' , 'F06', 'F07',
           'F08', 'F09', 'F10','P12','P43', 'P52' , 'P63' ,
                'P81' , 'P84', 'P96','P97', 'P98']
results_table_index = pd.DataFrame({
    'Participant':list_particips})
tbl_accuracy2 = pd.concat([results_table_index,results_table11],axis=1)
tbl_f12 = pd.concat([results_table_index,results_table22], axis=1)

tbl_accuracy2['Mean accuracy by classifier (%)'] = tbl_accuracy2.mean(axis=1)
tbl_f12['Mean f1-score by classifier'] = tbl_f12.mean(axis=1)
tbl_accuracy2['Std dev by classifier (%)'] = tbl_accuracy2.std(axis=1)
tbl_f12['Std dev by classifier (%)'] = tbl_f12.std(axis=1)
tbl_accuracy2.set_index('Participant', inplace = True)
tbl_f12.set_index("Participant", inplace = True)
tbl_accuracy2.loc['Mean accuracy per class'] = tbl_accuracy2.mean()
tbl_f12.loc['Mean f1 per class'] = tbl_f12.mean()
tbl_accuracy2.loc['std dev per class'] = tbl_accuracy2.std(axis=0)
tbl_f12.loc['std dev per class'] = tbl_f12.std(axis=0)

tbl_accuracy2 = tbl_accuracy2.round(decimals=3) * 100
tbl_f12 = tbl_f12.round(decimals=3) * 100
#
export_csv3 = tbl_accuracy2.to_csv ('RF_PCA_accuracyForAllFtbls.csv', index = True, header=True)
export_csv4 = tbl_f12.to_csv ('RF_PCA_f1ForAllFtbls.csv', index = True, header=True)
#
b=svm_classify_afterPCA(pca_df_X_train_list[18],pca_df_X_test_list[18],orig_df_y_train_list[18],orig_df_y_test_list[18],'rbf',0.07,6,'ovo')
precision_recall_fscore_support(orig_df_y_test_list[18], b[2], average='weighted')
b[3]['1']['precision']
######################
## Alt 2: RF + SVM ##
####################

from timeit import default_timer as timer
def SplitTrainTest_GetFeat_RF(df,test_size, method_ImpFeature_select):
    '''
    Splits data into X's used for training and testing, and Y's used for the same. Then uses Random Forest classifier to sort important features from most to
    least important
    :param df: feature table data frame
    :param test_size: number between 0 and 1; % of entire data used as testing set
    :param method_ImpFeature_select: method of calculating feature importance scores;  string: can be 'mean_decrease_in_impurity' or 'permutation_importance'
    :return: a pandas series with index as names of features ordered from most to least important, and 1 column of their relative importance scores
    '''
    df_FeatImp = pd.Series()
    y = df.iloc[:,0]  # subset truelabel column (response)
    x = df[list(df.iloc[:, 1: df.shape[1]])]
    X_train_RF, X_test_RF, y_train_RF, y_test_RF = train_test_split(x, y, test_size=test_size)
    start = timer()
    clf_RF = RandomForestClassifier(bootstrap=True, n_estimators=500, n_jobs=2, oob_score=True, class_weight='balanced',
                                    criterion='gini', max_depth=60, max_features='auto', max_leaf_nodes=None,
                                    min_samples_leaf=2, min_samples_split=2, min_weight_fraction_leaf=0.0,
                                    random_state=0, verbose=1, warm_start=False)
    clf_RF.fit(X_train_RF, y_train_RF) # fit RF classifier
    end = timer()
    if method_ImpFeature_select == 'mean_decrease_in_impurity':
        df_FeatImp = pd.Series(clf_RF.feature_importances_, index=X_train_RF.columns)
        df_FeatImp = df_FeatImp.sort_values(ascending=False)
    return [df_FeatImp,end-start]

ImpFeat_list = []
train_time_RF_list = []
for df in df_ftbl_list:
    ImpFeat_list.append(SplitTrainTest_GetFeat_RF(df, test_size=0.30, method_ImpFeature_select='mean_decrease_in_impurity' )[0])
for df in df_ftbl_list:
    train_time_RF_list.append(SplitTrainTest_GetFeat_RF(df, test_size=0.30,method_ImpFeature_select='mean_decrease_in_impurity' )[1])
# bar plot
featimpRF_bar= ImpFeat_list[0].nlargest(10).plot(kind='barh') # CAN ALTER argument - set fixed # of features to select
featimpRF_bar.invert_yaxis() # so the most important feature is at top instead of bottom of barplot - more intuitive
plt.title('Feature Importances')
plt.xlabel('Relative Importance')

list_particips = ['F01', 'F02', 'F03' , 'F04' , 'F05' , 'F06', 'F07',
           'F08', 'F09', 'F10','P12','P43', 'P52' , 'P63' ,
                'P81' , 'P84', 'P96','P97', 'P98']
Top5ImpFeat_list = []
for i in ImpFeat_list:
    # first make index (which are feature names) in EACH series in ImpFeat_list a COLUMN of a df
    # make new column in 'ImpFeat_list' for each df, called 'UniqueID' and append corresponding name in 'list_particips' above defined variable,
    #   then concatenate this list of dfs (now with new column of uniqueIDs) into a dataframe - save it!
    # then extract top 5 values from 'values' column from the dfs (Note: NOT series anymore)
    Top5ImpFeat_list.append(ImpFeat_list[0].nlargest(5))
    # then get frequency 'y.bin.values count from numpy'??? of each of the feature names occurrence
    # do pie chart!!

def get_top_features(df_ftbl,df, num_feat):
    # get top 'num_feat' features from sorted pandas series of features from most to least important
    # returns data frame of only those feature columns and the truelabel column (subsetted from imported feature table df)
    df_impfeat_top = df_ftbl.loc[:,df.nlargest(num_feat).index]
    df_forSVM = pd.concat([df_ftbl.iloc[:,0],df_impfeat_top],axis = 1)
    return df_forSVM

FeatForSVM_df_list = []
for df_ftbl , df in zip(df_ftbl_list,ImpFeat_list):
    FeatForSVM_df_list.append(get_top_features(df_ftbl, df, num_feat = 20))

def svm_classify_afterRF(df, test_size, kernel, gamma, Cparam, decision_function_shape, degree = 3 ):
    y_pred_svm = pd.DataFrame()
    crossval = np.empty([1,10])
    x = df.loc[:,df.columns != 'tl']
    y = df.iloc[:,0]
    X_train_SVM, X_test_SVM, y_train_SVM, y_test_SVM = train_test_split(x, y, test_size=test_size,
                                                                        random_state=1)  # can CHANGE test size
    if kernel =='rbf':
        start = timer()
        clf_svm = svm.SVC(kernel=kernel, gamma=gamma, C=Cparam, decision_function_shape=decision_function_shape)
        clf_svm.fit(X_train_SVM, y_train_SVM)
        end=timer()
        crossval = cross_val_score(estimator=clf_svm, X=X_train_SVM, y=y_train_SVM, cv=10)
        start2=timer()
        y_pred_svm = clf_svm.predict(X_test_SVM)
        end2=timer()

    if kernel == 'poly':
        clf_svm = svm.SVC(kernel=kernel, gamma=gamma, C=Cparam, decision_function_shape=decision_function_shape, degree = degree)
        clf_svm.fit(X_train_SVM, y_train_SVM)
        crossval = cross_val_score(estimator=clf_svm, X=X_train_SVM, y=y_train_SVM, cv=10)
        y_pred_svm = clf_svm.predict(X_test_SVM)
    class_report = classification_report(y_test_SVM,y_pred_svm,output_dict=True)
    #clf_svm_matrix = pd.crosstab(y_test_SVM, y_pred_svm, rownames=['Actual Result'], colnames=['Predicted Result'])
    # clf_svm_norm_matrix = clf_svm_matrix / clf_svm_matrix.astype(np.float).sum(axis = 1)
    f1_score_svm = f1_score(y_test_SVM, y_pred_svm, average='micro') # adjust 'AVERAGE" parameter for calculating f1 score, 'macro' penalizes the minority class misclassfication more
    return [f1_score_svm,class_report, y_test_SVM, y_pred_svm, crossval.mean(), crossval.std(), end-start, end2-start2]

   # return classification_report(y_test_SVM,y_pred_svm)

#svm_afterRF_confMatrix_list = []
svm_afterRF_f1Score_list  = []
svm_afterRF_classrep_list = []
#
train_time_svm_list = []
test_time_svm_list=[]
#
y_test_svm_list = []
y_pred_svm_list = []
classes_list = []
#
svm_afterRF_crossval_mean = []
svm_afterRF_crossval_std = []
#
for df in FeatForSVM_df_list:
    svm_afterRF_f1Score_list.append(svm_classify_afterRF(df,0.3,'rbf',gamma=0.07,Cparam=6,decision_function_shape='ovo')[0])
    svm_afterRF_classrep_list.append(svm_classify_afterRF(df,0.3,'rbf',gamma=0.07,Cparam=6,decision_function_shape='ovo')[1])

    y_test_svm_list.append(svm_classify_afterRF(df,0.3,'rbf',gamma=0.07,Cparam=6,decision_function_shape='ovo')[2])
    y_pred_svm_list.append(svm_classify_afterRF(df,0.3,'rbf',gamma=0.07,Cparam=6,decision_function_shape='ovo')[3])
    #classes_list.append(svm_classify_afterRF(df,0.3,'rbf',gamma=0.07,Cparam=6,decision_function_shape='ovo')[4])
    svm_afterRF_crossval_mean.append(svm_classify_afterRF(df, 0.3, 'rbf', gamma=0.07, Cparam=6, decision_function_shape='ovo')[4])
    svm_afterRF_crossval_std.append(svm_classify_afterRF(df, 0.3, 'rbf', gamma=0.07, Cparam=6, decision_function_shape='ovo')[5])
for df in FeatForSVM_df_list:
    train_time_svm_list.append(svm_classify_afterRF(df, 0.3, 'rbf', gamma=0.07, Cparam=6, decision_function_shape='ovo')[6])
    test_time_svm_list.append(svm_classify_afterRF(df, 0.3, 'rbf', gamma=0.07, Cparam=6, decision_function_shape='ovo')[7])

np.mean(svm_afterRF_f1Score_list) # 81% using 'weighted' f1 score  # 82 % using 'micro' f1 score
print(svm_afterRF_f1Score_list)
print(svm_afterRF_crossval_mean) # compare 2 lists outputs and know NOT overfitting bc very similar prediction accracies of test sets VS 10-fold CV
print(svm_afterRF_crossval_std)

lst1 = []
lst2 = []
lst3 = []
lstA = []
lstB = []
lstC = []
lsta= []
lstb = []
lstc = []
for i in range(len(svm_afterRF_classrep_list)):
    lst1.append(svm_afterRF_classrep_list[i]['1']['recall'])
    lst2.append(svm_afterRF_classrep_list[i]['2']['recall'])
    lst3.append(svm_afterRF_classrep_list[i]['3']['recall'])
    lstA.append(svm_afterRF_classrep_list[i]['1']['f1-score'])
    lstB.append(svm_afterRF_classrep_list[i]['2']['f1-score'])
    lstC.append(svm_afterRF_classrep_list[i]['3']['f1-score'])
    lsta.append(svm_afterRF_classrep_list[i]['1']['support'])
    lstb.append(svm_afterRF_classrep_list[i]['2']['support'])
    lstc.append(svm_afterRF_classrep_list[i]['3']['support'])


results_table1 = pd.DataFrame(
    {'Gesture 1 accuracy (%)': lst1,
     'Gesture 2 accuracy (%)': lst2,
     'Gesture 3 accuracy (%)': lst3})
results_table2 = pd.DataFrame(
    {'Gesture 1 f1-score (%)': lstA,
     'Gesture 2 f1-score (%)': lstB,
     'Gesture 3 f1-score (%)': lstC})

list_particips = ['F01', 'F02', 'F03' , 'F04' , 'F05' , 'F06', 'F07',
           'F08', 'F09', 'F10','P12','P43', 'P52' , 'P63' ,
                'P81' , 'P84', 'P96','P97', 'P98']
results_table_index = pd.DataFrame({
    'Participant':list_particips})
tbl_accuracy = pd.concat([results_table_index,results_table1],axis=1)
tbl_f1 = pd.concat([results_table_index,results_table2], axis=1)

tbl_accuracy['Mean accuracy by classifier (%)'] = tbl_accuracy.mean(axis=1)
tbl_f1['Mean f1-score by classifier'] = tbl_f1.mean(axis=1)
tbl_accuracy['Std dev by classifier (%)'] = tbl_accuracy.std(axis=1)
tbl_f1['Std dev by classifier (%)'] = tbl_f1.std(axis=1)
tbl_accuracy.set_index('Participant', inplace = True)
tbl_f1.set_index("Participant", inplace = True)
tbl_accuracy.loc['Mean accuracy per class'] = tbl_accuracy.mean()
tbl_f1.loc['Mean f1 per class'] = tbl_f1.mean()
tbl_accuracy.loc['std dev per class'] = tbl_accuracy.std(axis=0)
tbl_f1.loc['std dev per class'] = tbl_f1.std(axis=0)

tbl_accuracy = tbl_accuracy.round(decimals=3) * 100
tbl_f1 = tbl_f1.round(decimals=3) * 100
#
export_csv = tbl_accuracy.to_csv ('RF_SVM_accuracyForAllFtbls.csv', index = True, header=True)
export_csv2 = tbl_f1.to_csv ('RF_SVM_f1ForAllFtbls.csv', index = True, header=True)
#



## PLOTTING CONFUSION MATRIX HEAT MAP
# VERSION 1 (without total columns and rows)
def cm_analysis(y_true, y_pred, labels, ymap=None, figsize=(6,6)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args:
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      #filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    #if ymap is not None:
    #    y_pred = [ymap[yi] for yi in y_pred]
    #    y_true = [ymap[yi] for yi in y_true]
    #    labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = '0'
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual gesture label'
    cm.columns.name = 'Predicted gesture label'
    fig, ax = plt.subplots(figsize=figsize)
    hm=sn.heatmap(cm, annot=annot,cmap = 'Blues',linewidths=.5, fmt='', ax=ax, cbar=False, annot_kws={"size": 12})
    hm.xaxis.set_ticks_position('top')
    hm.xaxis.set_label_position('top')
    sn.set(font_scale=1.5)
    #plt.savefig(filename)
    plt.show()
cm_analysis(y_test_svm_list[8],y_pred_svm_list[8],labels=classes_list[0],ymap = None, figsize = (6,6))

# should I do a GridSear xch-CV for EACH feature table? aka different svm params for EACH ftbl?????
# Bayesian optimization to tune hyperparameters? https://thuijskens.github.io/2016/12/29/bayesian-optimisation/
# CONFUSION MATRIX HEAT MAP
# hyperparameter tune PCA - even without gridsearch (aka using same params for pca-SVM as I did for RF-SVM, was able to get 80% accuracy), SO PCA may actually be BETTER than RF20-SVM
# AUC/ROC curve plotting  

# version 2 (with ''; similar to MatLab heatmap)
from pandas import DataFrame
import matplotlib.font_manager as fm
from matplotlib.collections import QuadMesh

def get_new_fig(fn, figsize=[9,9]):
    """ Init graphics """
    fig1 = plt.figure(fn, figsize)
    ax1 = fig1.gca()   #Get Current Axis
    ax1.cla() # clear existing plot
    return fig1, ax1
#

def configcell_text_and_colors(array_df, lin, col, oText, facecolors, posi, fz, fmt, show_null_values=0):
    """
      config cell text and colors
      and return text elements to add and to dell
      @TODO: use fmt
    """
    text_add = []
    text_del = []
    cell_val = array_df[lin][col]
    tot_all = array_df[-1][-1]
    per = (float(cell_val) / tot_all) * 100
    curr_column = array_df[:,col]
    ccl = len(curr_column)

    #last line  and/or last column
    if(col == (ccl - 1)) or (lin == (ccl - 1)):
        #tots and percents
        if(cell_val != 0):
            if(col == ccl - 1) and (lin == ccl - 1):
                tot_rig = 0
                for i in range(array_df.shape[0] - 1):
                    tot_rig += array_df[i][i]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif(col == ccl - 1):
                tot_rig = array_df[lin][lin]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif(lin == ccl - 1):
                tot_rig = array_df[col][col]
                per_ok = (float(tot_rig) / cell_val) * 100
            per_err = 100 - per_ok
        else:
            per_ok = per_err = 0

        per_ok_s = ['%.2f%%'%(per_ok), '100%'] [per_ok == 100]

        #text to DEL
        text_del.append(oText)

        #text to ADD
        font_prop = fm.FontProperties(weight='bold', size=fz)
        text_kwargs = dict(color='w', ha="center", va="center", gid='sum', fontproperties=font_prop)
        lis_txt = ['%d'%(cell_val), per_ok_s, '%.2f%%'%(per_err)]
        lis_kwa = [text_kwargs]
        dic = text_kwargs.copy(); dic['color'] = 'g'; lis_kwa.append(dic)
        dic = text_kwargs.copy(); dic['color'] = 'r'; lis_kwa.append(dic)
        lis_pos = [(oText._x, oText._y-0.3), (oText._x, oText._y), (oText._x, oText._y+0.3)]
        for i in range(len(lis_txt)):
            newText = dict(x=lis_pos[i][0], y=lis_pos[i][1], text=lis_txt[i], kw=lis_kwa[i])
            #print 'lin: %s, col: %s, newText: %s' %(lin, col, newText)
            text_add.append(newText)
        #print '\n'

        #set background color for sum cells (last line and last column)
        carr = [0.27, 0.30, 0.27, 1.0]
        if(col == ccl - 1) and (lin == ccl - 1):
            carr = [0.17, 0.20, 0.17, 1.0]
        facecolors[posi] = carr

    else:
        if(per > 0):
            txt = '%s\n%.2f%%' %(cell_val, per)
        else:
            if(show_null_values == 0):
                txt = ''
            elif(show_null_values == 1):
                txt = '0'
            else:
                txt = '0\n0.0%'
        oText.set_text(txt)

        #main diagonal
        if(col == lin):
            #set color of the textin the diagonal to white
            oText.set_color('w')
            # set background color in the diagonal to blue
            facecolors[posi] = [0.35, 0.8, 0.55, 1.0]
        else:
            oText.set_color('r')

    return text_add, text_del
#

def insert_totals(df_cm):
    """ insert total column and line (the last ones) """
    sum_col = []
    for c in df_cm.columns:
        sum_col.append( df_cm[c].sum() )
    sum_lin = []
    for item_line in df_cm.iterrows():
        sum_lin.append( item_line[1].sum() )
    df_cm['sum_lin'] = sum_lin
    sum_col.append(np.sum(sum_lin))
    df_cm.loc['sum_col'] = sum_col
    #print ('\ndf_cm:\n', df_cm, '\n\b\n')
#

def pretty_plot_confusion_matrix(df_cm, annot=True, cmap="Blues", fmt='.2f', fz=11,
      lw=0.5, cbar=False, figsize=[8,8], show_null_values=0, pred_val_axis='y'):
    """
      print conf matrix with default layout (like matlab)
      params:
        df_cm          dataframe (pandas) without totals
        annot          print text in each cell
        cmap           Oranges,Oranges_r,YlGnBu,Blues,RdBu, ... see:
        fz             fontsize
        lw             linewidth
        pred_val_axis  where to show the prediction values (x or y axis)
                        'col' or 'x': show predicted values in columns (x axis) instead lines
                        'lin' or 'y': show predicted values in lines   (y axis)
    """
    if(pred_val_axis in ('col', 'x')):
        xlbl = 'Predicted'
        ylbl = 'Actual'
    else:
        xlbl = 'Actual'
        ylbl = 'Predicted'
        df_cm = df_cm.T

    # create "Total" column
    insert_totals(df_cm)

    #this is for print allways in the same window
    fig, ax1 = get_new_fig('Conf matrix default', figsize)

    #thanks for seaborn
    ax = sn.heatmap(df_cm, annot=annot, annot_kws={"size": fz}, linewidths=lw, ax=ax1,
                    cbar=cbar, cmap=cmap, linecolor='w', fmt=fmt)

    #set ticklabels rotation
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, fontsize = 10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 25, fontsize = 10)

    # Turn off all the ticks
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    #face colors list
    quadmesh = ax.findobj(QuadMesh)[0]
    facecolors = quadmesh.get_facecolors()

    #iter in text elements
    array_df = np.array( df_cm.to_records(index=False).tolist() )
    text_add = []
    text_del = []
    posi = -1 #from left to right, bottom to top.
    for t in ax.collections[0].axes.texts: #ax.texts:
        pos = np.array( t.get_position()) - [0.5,0.5]
        lin = int(pos[1]); col = int(pos[0])
        posi += 1
        #print ('>>> pos: %s, posi: %s, val: %s, txt: %s' %(pos, posi, array_df[lin][col], t.get_text()))

        #set text
        txt_res = configcell_text_and_colors(array_df, lin, col, t, facecolors, posi, fz, fmt, show_null_values)

        text_add.extend(txt_res[0])
        text_del.extend(txt_res[1])

    #remove the old ones
    for item in text_del:
        item.remove()
    #append the new ones
    for item in text_add:
        ax.text(item['x'], item['y'], item['text'], **item['kw'])

    #titles and legends
    ax.set_title('Confusion matrix')
    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    plt.tight_layout()  #set layout slim
    plt.show()
#

def plot_confusion_matrix_from_data(y_test, predictions, columns=None, annot=True, cmap="Oranges",
      fmt='.2f', fz=11, lw=0.5, cbar=False, figsize=[8,8], show_null_values=0, pred_val_axis='lin'):
    """
        plot confusion matrix function with y_test (actual values) and predictions (predic),
        whitout a confusion matrix yet
    """
    from sklearn.metrics import confusion_matrix
    from pandas import DataFrame

    #data
    if(not columns):
        #labels axis integer:
        ##columns = range(1, len(np.unique(y_test))+1)
        #labels axis string:
        from string import ascii_uppercase
        columns = ['gesture label %s' % (i) for i in range(1,len(np.unique(y_test))+1) ] # CHANGE NAMES FOR CLASS LABELS
        #columns = ['gesture label %s' %(i) for i in list(ascii_uppercase)[0:len(np.unique(y_test))]] # CHANGE NAMES FOR CLASS LABELS

    confm = confusion_matrix(y_test, predictions)
    cmap = 'Blues'
    fz = 11
    figsize=[9,9]
    show_null_values = 2
    df_cm = DataFrame(confm, index=columns, columns=columns)
    pretty_plot_confusion_matrix(df_cm, fz=fz, cmap=cmap, figsize=figsize, show_null_values=show_null_values, pred_val_axis=pred_val_axis)
#
def _test_data_class(y_test_svm,y_pred_svm):
    columns = []
    annot = True
    cmap = 'Blues'
    fmt = '.2f'
    lw = 0.5
    cbar = False
    show_null_values = 2
    pred_val_axis = 'y'
    #size::
    fz = 12
    figsize = [9,9]
    if(len(y_test_svm) > 10):
        fz=9; figsize=[14,14]
    plot_confusion_matrix_from_data(y_test_svm, y_pred_svm, columns,
      annot, cmap, fmt, fz, lw, cbar, figsize, show_null_values, pred_val_axis)
#
_test_data_class(y_test_svm_list[8],y_pred_svm_list[8]) # PLOT FOR F09