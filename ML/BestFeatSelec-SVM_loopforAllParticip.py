import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import feature_selection
from sklearn import svm
from sklearn.metrics import classification_report, f1_score, recall_score
from sklearn.model_selection import cross_val_score
from timeit import default_timer as timer

csvfile_list = ['FtTblN_F01.csv','FtTblN_F02.csv', 'FtTblN_F03.csv' , 'FtTblN_F04.csv' , 'FtTblN_F05.csv' , 'FtTblN_F06.csv', 'FtTblN_F07.csv',
           'FtTblN_F08.csv', 'FtTblN_F09.csv', 'FtTblN_F10.csv','FtTblN_P12.csv','FtTblN_P43.csv', 'FtTblN_P52.csv' , 'FtTblN_P63.csv' ,
                'FtTblN_P81.csv' , 'FtTblN_P84.csv', 'FtTblN_P96.csv','FtTblN_P97.csv', 'FtTblN_P98.csv']
df_ftbl_list = []
for file in csvfile_list:
    # read in feature table
    df_ftbl_list.append(pd.read_csv(file))

# Random Forest used to feature-select, then selection criteria (threshold of selection = features that exceed mean importance score of all features)
def SplitTrainTest_GetFeat_RF(df, method_ImpFeature_select):
    '''
    Splits data into X's used for training and testing, and Y's used for the same. Then uses Random Forest classifier to sort important features from most to
    least important, using default metric in RF (i.e. mean decre. in impurity of feature)
    :param df: feature table data frame
    :param test_size: number between 0 and 1; % of entire data used as testing set
    :param method_ImpFeature_select: method of calculating feature importance scores;  string: can be 'mean_decrease_in_impurity' or 'permutation_importance'
    :return: a pandas series with index as names of features ordered from most to least important, and 1 column of their relative importance scores
    '''
    y = df.iloc[:,0]  # subset truelabel column (response)
    x = df[list(df.iloc[:, 1: df.shape[1]])]
    feat_imp = []

    start = timer()
    clf_RF = RandomForestClassifier(bootstrap=True, n_estimators=500, n_jobs=2, oob_score=True, class_weight='balanced',
                                    criterion='gini', max_depth=60, max_features='auto', max_leaf_nodes=None,
                                    min_samples_leaf=2, min_samples_split=2, min_weight_fraction_leaf=0.0,
                                    random_state=0, verbose=1, warm_start=False)
    sfm = feature_selection.SelectFromModel(clf_RF, threshold='mean') # threshold of feature picking is the MEAN importance score # CAN CHANGE
    sfm.fit(x,y) # fit RF classifier
    x_sfm = sfm.transform(x) # pulling the important features only
    end = timer()

    # put together true label column + relevant features (from using SelectFromModel above)
    df_x_sfm = pd.DataFrame(x_sfm, columns=x.columns[sfm.get_support()])
    n_features = x_sfm.shape[1]
    df_forSVM = pd.concat([y, df_x_sfm], axis=1)
    for index in sfm.get_support(indices=True):
        feat_imp.append(x.columns[index])

    return [df_forSVM,n_features,end-start, feat_imp]
# implements above UDF for all 19
df_forSVM_list = []
num_feat_selected_list = []
train_time_RF_list = []
feat_imp_list = []
for df in df_ftbl_list:
    df_forSVM_list.append(SplitTrainTest_GetFeat_RF(df, method_ImpFeature_select='mean_decrease_in_impurity' )[0])
    num_feat_selected_list.append(SplitTrainTest_GetFeat_RF(df,method_ImpFeature_select='mean_decrease_in_impurity')[1])
    train_time_RF_list.append(SplitTrainTest_GetFeat_RF(df, method_ImpFeature_select='mean_decrease_in_impurity' )[2])
    feat_imp_list.append(SplitTrainTest_GetFeat_RF(df, method_ImpFeature_select='mean_decrease_in_impurity' )[3])


# apply SVM
def SVM_afterRF(df_afterFeatSelect,test_size, kernel, gamma, Cparam, decision_function_shape, degree = 3 ):
    y_pred_svm = pd.DataFrame()
    x_SVM = df_afterFeatSelect.loc[:, df_afterFeatSelect.columns != 'tl']
    y_SVM = df_afterFeatSelect.iloc[:, 0]
    X_train_SVM, X_test_SVM, y_train_SVM, y_test_SVM = train_test_split(x_SVM, y_SVM, test_size=test_size,
                                                                                random_state=1)
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

    class_report = classification_report(y_test_SVM, y_pred_svm, output_dict=True)
    f1_score_svm = f1_score(y_test_SVM, y_pred_svm, average='weighted')
    recall_score_svm = recall_score(y_test_SVM, y_pred_svm, average='weighted')
    return [class_report,f1_score_svm,recall_score_svm, crossval.mean(), crossval.std(), end-start, end2-start2]


svm_afterRF_classrep_list = []
svm_afterRF_f1Score_list  = []
svm_afterRF_recallScore_list = []

svm_afterRF_crossval_mean = []
svm_afterRF_crossval_sd = []

svm_traintime_list = []
svm_predtime_list = []

for df in df_forSVM_list:
    svm_afterRF_classrep_list.append(SVM_afterRF(df,0.3,'rbf',gamma=0.07,Cparam=6,decision_function_shape='ova')[0])
    svm_afterRF_f1Score_list.append(SVM_afterRF(df,0.3,'rbf',gamma=0.07,Cparam=6,decision_function_shape='ova')[1])
    svm_afterRF_recallScore_list.append(SVM_afterRF(df,0.3,'rbf',gamma=0.07,Cparam=6,decision_function_shape='ova')[2])

    svm_afterRF_crossval_mean.append(SVM_afterRF(df,0.3,'rbf',gamma=0.07,Cparam=6,decision_function_shape='ova')[3])
    svm_afterRF_crossval_sd.append(SVM_afterRF(df,0.3,'rbf',gamma=0.07,Cparam=6,decision_function_shape='ova')[4])

    svm_traintime_list.append(SVM_afterRF(df,0.3,'rbf',gamma=0.07,Cparam=6,decision_function_shape='ova')[5])
    svm_predtime_list.append(SVM_afterRF(df,0.3,'rbf',gamma=0.07,Cparam=6,decision_function_shape='ova')[6])

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
export_csv = tbl_accuracy.to_csv ('BestFeatRF_SVM_accuracyForAllFtbls.csv', index = True, header=True)
export_csv2 = tbl_f1.to_csv ('BestFeatRF_SVM_f1ForAllFtbls.csv', index = True, header=True)
#
