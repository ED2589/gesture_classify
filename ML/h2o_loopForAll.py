import h2o
from h2o.automl import H2OAutoML
h2o.init()

csvfile_list = ['FtTblN_F01.csv','FtTblN_F02.csv', 'FtTblN_F03.csv' , 'FtTblN_F04.csv' , 'FtTblN_F05.csv' , 'FtTblN_F06.csv', 'FtTblN_F07.csv',
           'FtTblN_F08.csv', 'FtTblN_F09.csv', 'FtTblN_F10.csv','FtTblN_P12.csv','FtTblN_P43.csv', 'FtTblN_P52.csv' , 'FtTblN_P63.csv' ,
                'FtTblN_P81.csv' , 'FtTblN_P84.csv', 'FtTblN_P96.csv','FtTblN_P97.csv', 'FtTblN_P98.csv']
df_h2o_list = []
for file in csvfile_list:
    # read in feature table using h2o import function
    df_h2o_list.append(h2o.import_file(file))
len(df_h2o_list)

def split_test_train_h2o(df):
    # convert response column to a factor
    df['tl'] = df['tl'].asfactor()
    # split into train and validation sets
    train, test = df.split_frame(ratios=[.7], seed=1234)  # 30% test size  # seed value is for reproducibility
    # set the predictor names and the response column name
    return [train,test]

train_list=[] # list of training data frames
test_list=[] # list of test data frames
for df in df_h2o_list:
    train_list.append(split_test_train_h2o(df)[0]) # length of 19 train dfs
    test_list.append(split_test_train_h2o(df)[1]) # length of 19 test dfs

from timeit import default_timer as timer
def autoML(df,train_df,num_models):
    x = df.columns
    y = "tl"
    x.remove(y)
    start=timer()
    aml = H2OAutoML(max_models=num_models, seed=1)  # num_models number of base models + 2 stacked ensemble models trained
    aml.train(x=x, y=y, training_frame=train_df)
    end=timer()
    return [aml.leaderboard, end-start] # return models ranked from best to worst-performing

leaderboard_list = []
train_time_h2o_10models = []
for df,train_df in zip(df_h2o_list,train_list):
    leaderboard_list.append(autoML(df,train_df,10)[0]) # here we specify 10 different models to be trained, for each feature table
    train_time_h2o_10models.append(autoML(df,train_df,10)[1])

# extract the top 3 models into 3 separate lists
topmodel_list=[]
second_rank_model_list = []
third_rank_model_list = []
for i in leaderboard_list:
    topmodel_list.append(h2o.get_model(i[0,'model_id'])) # get top model
    second_rank_model_list.append(h2o.get_model(i[1,'model_id'])) # get 2nd-ranked model
    third_rank_model_list.append(h2o.get_model(i[2,'model_id'])) # get 3rd-ranked model

# test set predictions
def topmodel_test_metrics(topmodel,test_df):
    start=timer()
    topmodel.predict(test_df)  # test set prediction for top model
    end=timer()
    leader_perform = topmodel.model_performance(test_df)  # model performance: hit ratio, confusion matrix on train + cross-validation sets, feature importance measures
    #metrics_all = topmodel.show()  # BETTER way to show model performance: hit ratio, confusion matrix, variable importance, etc
    confusion_mat = topmodel.model_performance(test_df).confusion_matrix()  # confusion matrix on test set ONLY (not cross valid sets here)
    hit_ratio = topmodel.model_performance(test_df).hit_ratio_table()
    return [leader_perform,confusion_mat,hit_ratio,end-start]

metrics_topmodel_list = []  # test set prediction for top model
metrics_2ndmodel_list = []  #  test set prediction for 2nd-ranked model
metrics_3rdmodel_list = []  # test set prediction for 3rd-ranked model
confusion_mat_list=[] # confusion marix of test set for top model
hit_ratio_list = []  # hit ratio table (for 3 classes) from test set for top model

#
topmodel_test_time_list = []
secondmodel_test_time_list = []
thirdmodel_test_time_list = []
#
for model,df in zip(topmodel_list,test_list):
    metrics_topmodel_list.append(topmodel_test_metrics(model,df)[0]) # top ranked model test set metrics
    confusion_mat_list.append(topmodel_test_metrics(model, df)[1])
    hit_ratio_list.append(topmodel_test_metrics(model, df)[2])
for model, df in zip(topmodel_list, test_list):
    topmodel_test_time_list.append(topmodel_test_metrics(model, df)[3])
for second_ranked_model,df in zip(second_rank_model_list,test_list):
    metrics_2ndmodel_list.append(topmodel_test_metrics(second_ranked_model,df)[0]) # 3rd ranked model test set metrics
for second_ranked_model, df in zip(second_rank_model_list, test_list):
    secondmodel_test_time_list.append(topmodel_test_metrics(second_ranked_model,df)[3])
for third_ranked_model,df in zip(third_rank_model_list,test_list):
    metrics_3rdmodel_list.append(topmodel_test_metrics(third_ranked_model,df)[0]) # 3rd ranked model test set metrics
for third_ranked_model, df in zip(third_rank_model_list, test_list):
    thirdmodel_test_time_list.append(topmodel_test_metrics(third_ranked_model,df)[3])

# getting hyperparameters for top models
topmodel_list[1].params.keys()  # get names of parameters of model
    # GBM hyperparameter used by h2o autoML: histogram_type , ntrees, max_depth, min_rows, learn_rate, sample_rate, col_sample_rate, col_sample_rate_per_tree, min_split_improvement
topmodel_list[0].params['ntrees'] # check out the default and actual values of the hyperparameter #GBM model 0 # 50,51
topmodel_list[0].params['max_depth'] # {'default': 5, 'actual': 6}
second_rank_model_list[0].params['ntrees'] # #GBM model 2 # {'default': 50, 'actual': 56}
second_rank_model_list[1].params['ntrees']

# consolidate results for all classifiers into Excel file (top models, confusion matrix, cross-validation accuracies, test set accuracies, variable importances)
# METHOD of giving best predictions is key - rather than the indivdidual model iteself
# look into: a) hyperparameter meanings for GBM model b) micro or macro averages
# try it for the other more conventional models (KNN, Naive Bayes, etc.) - see if there's an AutoML for these types of models
# figure out how to PLOT confusion matrix

# BUG: could not figure how to find base learner importance to ensemble(i.e. stacked) model
# Get model ids for all models in the AutoML Leaderboard
    #model_ids = list(leaderboard_list[1]['model_id'].as_data_frame().iloc[:,0])
# Get the "All Models" Stacked Ensemble model
    #se = h2o.get_model([mid for mid in model_ids if "StackedEnsemble_AllModels" in mid][0])
# Get the Stacked Ensemble metalearner model
    #metalearner = h2o.get_model(se.metalearner()['name'])
    #metalearner.coef_norm()

# game plan: 1) extracting ensemble models (below code) 2) comparing excel table results to find patterns 3) trying other methods (KNN etc.)

# getting base models involved in stacked ensemble model
model_ids = list(leaderboard_list[18]['model_id'].as_data_frame().iloc[:,0])
m = h2o.get_model(model_ids[0])
m.params['base_models']
urllist = []
for model in m.params['base_models']['actual']:
    urllist.append(model['URL'])

print(urllist) # the specific models involved in stacked ensemble, from most to least important