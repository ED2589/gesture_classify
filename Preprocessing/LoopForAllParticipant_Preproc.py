# Created May 24,2019 #
# Overview:
        # takes each of the 19 EMG files + each of the 19 IMU files,
        # and returns ,mat files of binned data (in Python it's 3D array)
##########
# Step 1 #
##########
# import raw data (.csv) into Python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import itertools
import scipy.io

def import_data(csv_file):
    """
    :param csv_file: a string i.e. name of .csv file
        csv_file is the raw data (uniqueID, date, pods columns) - for 1 participant
    :return: pandas data frame of imported data
    """
    return pd.read_csv(csv_file)

emg_csv_list = ['F01_emg.csv','F02_emg.csv','F03_emg.csv', 'F04_emg.csv','F05_emg.csv','F06_emg.csv','F07_emg.csv', 'F08_emg.csv',
                'F09_emg.csv','F10_emg.csv','P12_emg.csv','P43_emg.csv','P52_emg.csv','P63_emg.csv','P81_emg.csv','P84_emg.csv',
                'P96_emg.csv','P97_emg.csv','P98_emg.csv']
imu_csv_list = ['F01_imu.csv','F02_imu.csv','F03_imu.csv', 'F04_imu.csv','F05_imu.csv','F06_imu.csv','F07_imu.csv', 'F08_imu.csv',
                'F09_imu.csv','F10_imu.csv','P12_imu.csv','P43_imu.csv','P52_imu.csv','P63_imu.csv','P81_imu.csv','P84_imu.csv',
                'P96_imu.csv','P97_imu.csv','P98_imu.csv']
emg_rawDF_list = []
imu_rawDF_list = []
# creates a list of 19 dataframes, by reading in the 19 emg raw files
for i in emg_csv_list: # import all 19 EMG datasets
    emg_rawDF_list.append(import_data(i))
# creates a list of 19 dataframes, by reading in the 19 imu raw files
for i in imu_csv_list: # import all 19 IMU datasets
    imu_rawDF_list.append(import_data(i))

# extract each df from list - IS THIS NECESSARY?
#F01EmgRaw = emg_rawDF_list[0]
#F02EmgRaw = emg_rawDF_list[1]
#F03EmgRaw = emg_rawDF_list[2]
#F04EmgRaw = emg_rawDF_list[3]
# DO THIS FOR ALL 19 PARTICIPANTS
#F01ImuRaw = imu_rawDF_list[0]
# DO THIS FOR ALL 19 PARTICIPANTS

##########
# Step 2 #
##########

# Normalize each channel column in EMG to range (0,1)

def normalize_emg(df):
    '''
    :param df: Raw emg df
    :return: Emg df with each of pod0 columns normalized to range(0,1)
    '''
    ncol = df.shape[1]
    df1 = df.iloc[ :, 7: ncol-1] # all pod columns
    df2 = df.iloc[:,0]  # uniqueID column
    df1 = abs(df1)
    absEMG = pd.concat([df2,df1],axis=1)
    groupEMG = absEMG.groupby(['uniqueID'])

    def norm_func(g):
        return ((g - g.min()) / (g.max() - g.min()))

    dfnorm = groupEMG.apply(norm_func)  # normalizing step!

    subs_1 = df[['uniqueID', 'participant', 'phase', 'trial', 'date', 'structName','Timestr']]
    subs_2 = df[['TrueLabel']]
    return pd.concat([subs_1, dfnorm, subs_2], axis=1)


emg_normDF_list = [] # create empty list to store normalized EMG dfs
for df in emg_rawDF_list:
    emg_normDF_list.append(normalize_emg(df))

##########
# Step 3 #
##########
# Low - pass filter
def lowpass_filt(df , sf, order,cutoff):
    '''
    :param df:  normalized Emg df OR Raw Imu df
    :param sf: sampling freq (200 Hz for Emg ; 40 Hz for Imu)
    :param order: order of low pass filter
    :param cutoff: cutoff freq in Hz
    :return: lowpass filtered df
    '''
    ncol = df.shape[1]
    df_a = df.iloc[:, 7: ncol - 1]  # all numeric columns excluding 'TrueLabel' column
    df_b = df.iloc[:, 0]  # uniqueID column
    df2 = pd.concat([df_b, df_a], axis=1)
    df2_gp = df2.groupby(['uniqueID'])  # do new filter for each different trial
    df3_gp = [x for _, x in df2_gp] # list of data frames, each consisting of a different trial

    for group in df3_gp:
        group.drop('uniqueID', axis=1, inplace=True)
    norm_param = cutoff / (sf/2)
    d, c = signal.butter(order,norm_param,'low')
    for group in df3_gp:
        ncol = group.shape[1]
        for j in range(ncol):
            group.iloc[:, j] = signal.filtfilt(d, c, group.iloc[:, j], axis=0) # butterworth for each dataframe in the df3_gp (which is a list)
    df3_conc = pd.concat(df3_gp)
    # put the data back together, after the numerical pod columns are all low pass filtered
    subs_1 = df[['uniqueID', 'participant', 'phase', 'trial', 'date', 'structName', 'Timestr']]
    subs_2 = df[['TrueLabel']]
    df4 = pd.concat([subs_1, df3_conc, subs_2], axis=1)  # this is the LOW-PASS FILTERED EMG Data
    return df4

emg_LowFiltNormDF_list = [] # create empty list to store lowpass filtered, normalized EMG dfs
# low pass for all 19
for df_emg in emg_normDF_list:
    emg_LowFiltNormDF_list.append(lowpass_filt(df_emg , 200, 4,10))
imu_LowFiltDF_list = []  # create empty list to store lowpass filtered IMU dfs
for df_imu in imu_rawDF_list:
    imu_LowFiltDF_list.append(lowpass_filt(df_imu , 50, 4,10))

#######
## pod4 plot check for F02 Trial 1
#FakeT_Emg_F02T1_list = FakeTimeArray(emg_normDF_list[1],0.005)[0].tolist()
#EmgF02T1pod4_Nofilt_list = list(emg_normDF_list[1].pod4[0:RowNum(emg_normDF_list[1])[0]])
#EmgF02T1pod4_filt_list = list(emg_LowFiltNormDF_list[1].pod4[0:RowNum(emg_LowFiltNormDF_list[1])[0]])
#testdf_F02T1_pod4_fakeT = pd.DataFrame({'pod4_norm':EmgF02T1pod4_Nofilt_list,'pod4_norm_lowfilt': EmgF02T1pod4_filt_list},
                                       #index=FakeT_Emg_F02T1_list)
#LinePlot_Emg_F02T1_pod4 = testdf_F02T1_pod4_fakeT.plot.line()
#LinePlot_Emg_F02T1_pod4.set(xlabel='Time elapsed(s)', ylabel = 'Pod 4 Channel (V)')
#SepLinePlot_Emg_F02T1_pod4 = testdf_F02T1_pod4_fakeT.plot.line(subplots=True)
##########
# Step 4 #
##########
# REMOVE all rows where TrueLabel!=0 for both EMG and IMU
def RemoveZeroTrueLabel(df_filt):
    '''
    :param df_filt: filtered EMG or IMU df
    :return: same input df but with 'TrueLabel == 0 ' rows removed
    '''
    return df_filt[df_filt.TrueLabel != 0]

## EMG ##
emg_Subs_NoFiltDF_list =[]  # create empty list to store subsetted, NON-filtered, normalized EMG dfs
for df_emg in emg_normDF_list:
    emg_Subs_NoFiltDF_list.append(RemoveZeroTrueLabel(df_emg))

emg_Subs_LowFiltNormDF_list = [] # create empty list to store subsetted, lowpass filt, normalized EMG dfs
for df_filt_emg in emg_LowFiltNormDF_list:
    emg_Subs_LowFiltNormDF_list.append(RemoveZeroTrueLabel(df_filt_emg))

## IMU ##
imu_Subs_NoFiltDF_list =[]   # create empty list to store subsetted, NON-filtered filt IMU dfs
for df_imu in imu_rawDF_list:
    imu_Subs_NoFiltDF_list.append(RemoveZeroTrueLabel(df_imu))

imu_Subs_LowFiltNormDF_list = [] # create empty list to store subsetted, lowpass filt IMU dfs
for df_filt_imu in imu_LowFiltDF_list:
    imu_Subs_LowFiltNormDF_list.append(RemoveZeroTrueLabel(df_filt_imu))

##########
# Step 5 #
##########
#### binning after subsetting ######

# Step 1: delete rows from EMG and IMU to ensure SAME starting time for EACH uniqueID group
#       For detailed algorithm of function 'start_time_same' -> see 'PreprocStep1to4_F01.py' Line 375
def start_time_same(dfEMG,dfIMU):
    dfEMG["Timestr"] = pd.to_datetime(dfEMG["Timestr"])
    dfIMU["Timestr"] = pd.to_datetime(dfIMU["Timestr"])
    df_g_EMG = dfEMG.groupby('uniqueID')
    df_gp_EMG = [x for _, x in df_g_EMG]
    df_g_IMU = dfIMU.groupby('uniqueID')
    df_gp_IMU = [x for _, x in df_g_IMU]

    list_EMG = []
    list_IMU = []

    for i in range(len(df_gp_EMG)):
        EMG_1st_time = df_gp_EMG[i]['Timestr'].iloc[0]
        IMU_1st_time = df_gp_IMU[i]['Timestr'].iloc[0]

        # Case 1: if EMG earlier
        if EMG_1st_time < IMU_1st_time:
            # if any of the first 5 EMG time elements MATCHES IMU[0] original:
            if len( df_gp_EMG[i]['Timestr'].head(5) [ df_gp_EMG[i]['Timestr'].head(5) == IMU_1st_time ] ) != 0:
            # then remove all Emg[Emg < Imu[0]] i.e. keep ALL later times in EMG + last earlier element
                df_gp_EMG[i] = df_gp_EMG[i] [ df_gp_EMG[i]['Timestr'] >= IMU_1st_time ]
                list_EMG.append(df_gp_EMG[i])
                list_IMU.append(df_gp_IMU[i])
            # if not, remove all EMG < IMU[0] except for last earlier EMG Time (ensures output EMG time[0] still earlier)
            elif len( df_gp_EMG[i]['Timestr'].head(5) [ df_gp_EMG[i]['Timestr'].head(5) == IMU_1st_time ] ) == 0:
                last_earlier_EMG = df_gp_EMG[i]['Timestr'][df_gp_EMG[i]['Timestr'] < IMU_1st_time].max()  # extract the last EARLIER element from EMG time
                df_gp_EMG[i] = df_gp_EMG[i][ df_gp_EMG[i]['Timestr'] >= last_earlier_EMG]  # keep last EARLIER EMG time + all other times greater than IMU[0]
                list_EMG.append(df_gp_EMG[i])
                list_IMU.append(df_gp_IMU[i])
        # Case 2: if IMU earlier -> remove from IMU all earlier times (so 1st time in output IMU STILL later than 1st EMG time orig)
        elif EMG_1st_time > IMU_1st_time:
            df_gp_IMU[i] = df_gp_IMU[i] [ df_gp_IMU[i]['Timestr'] > EMG_1st_time ]
            list_EMG.append(df_gp_EMG[i])
            list_IMU.append(df_gp_IMU[i])
        else:
            list_EMG.append(df_gp_EMG[i])
            list_IMU.append(df_gp_IMU[i])

    return [pd.concat(list_EMG), pd.concat(list_IMU)]

emg_SameT_NoFiltDF_list =[]  # create empty list to store SameTime, no tl==0 rows,NON-filtered, normalized EMG dfs
imu_SameT_NoFiltDF_list =[]  # create empty list to store SameTime,no tl==0 rows,NON-filtered IMU dfs
#start_time_same(emg_Subs_NoFiltDF_list[1],imu_Subs_NoFiltDF_list[1])[0]
for (df_emg, df_imu) in itertools.zip_longest(emg_Subs_NoFiltDF_list,imu_Subs_NoFiltDF_list):
    emg_SameT_NoFiltDF_list.append(start_time_same(df_emg, df_imu)[0])
    imu_SameT_NoFiltDF_list.append(start_time_same(df_emg, df_imu)[1])

emg_SameT_LowFiltNormDF_list = [] # create empty list to store SameTime, subsetted,lowpass filt, normalized EMG dfs
imu_SameT_LowFiltNormDF_list = [] # create empty list to store SameTime, subsetted, lowpass filt, IMU dfs
#start_time_same(emg_Subs_LowFiltNormDF_list[1],imu_Subs_LowFiltNormDF_list[1])[1]
for (df_filt_emg,df_filt_imu) in itertools.zip_longest(emg_Subs_LowFiltNormDF_list,imu_Subs_LowFiltNormDF_list):
    emg_SameT_LowFiltNormDF_list.append(start_time_same(df_filt_emg,df_filt_imu)[0])
    imu_SameT_LowFiltNormDF_list.append(start_time_same(df_filt_emg,df_filt_imu)[1])

## check ##
#emg_a = emg_Subs_LowFiltNormDF_list[1].groupby('uniqueID')
#emg_b = [x for _, x in emg_a]
#imu_a = imu_Subs_LowFiltNormDF_list[1].groupby('uniqueID')
#imu_b = [x for _, x in imu_a]
#emg_b[0].Timestr.head(15)
#imu_b[0].Timestr.head(20)
#emg_a_t = emg_SameT_LowFiltNormDF_list[1].groupby('uniqueID')
#emg_b_t = [x for _, x in emg_a_t]
#imu_a_t = imu_SameT_LowFiltNormDF_list[1].groupby('uniqueID')
#imu_b_t = [x for _, x in imu_a_t]
#emg_b_t[0].Timestr.head(10)
#imu_b_t[0].Timestr.head(10)
#####################################
# Step 2: Binning
def binning(dfEmg, dfImu, winEmg , winImu):
    '''
    :param dfEmg , dfImu: EMG or IMU data (whether filtered or unfiltered)
    :param winEmg, winImu: # of rows of data frame per bin : EMG -> 40 ; IMU -> 10
    :return: 2 3D data frames (one of EMG , one of IMU)  with 3rd dimension being # of bins
    '''
    df.drop('Timestr' , axis = 1 , inplace= True)    # CONFIRM W ALEX - drop 'Timestr' column ? Yes, won't be needing for feature table.
    Emg_Array_List = []
    Imu_Array_List = []
    df_Emg = dfEmg.groupby('uniqueID')
    df_gp_Emg = [x for _ , x in df_Emg]
    df_Imu = dfImu.groupby('uniqueID')
    df_gp_Imu = [x for _, x in df_Imu]
    for group in df_gp_Emg:
        remEmg = group.shape[0] % winEmg   # number of rows to remove (mod operation)
        group.drop(group.tail(remEmg).index, inplace=True)
        group = group.values.reshape(group.size // winEmg // group.shape[1], winEmg, group.shape[1]) # (number of bins, num row aka winbin, num col)
        Emg_Array_List.append(group)
    for group in df_gp_Imu:
        remImu = group.shape[0] % winEmg   # number of rows to remove (mod operation)
        group.drop(group.tail(remImu).index, inplace=True)
        group = group.values.reshape(group.size // winImu // group.shape[1], winImu, group.shape[1]) # (number of bins, num row aka winbin, num col)
        Imu_Array_List.append(group)
    for i in range(len(Emg_Array_List)):
        diff_bins = Emg_Array_List[i].shape[0] - Imu_Array_List[i].shape[0]
        if diff_bins > 0:
            Emg_Array_List[i] = np.delete(Emg_Array_List[i] ,np.s_[-abs(diff_bins):] , axis=0)
        elif diff_bins < 0:
            Imu_Array_List[i] = np.delete(Imu_Array_List[i], np.s_[-abs(diff_bins):], axis=0)
        else:
            Emg_Array_List[i] = Emg_Array_List[i]
            Imu_Array_List[i] = Imu_Array_List[i]

    Emg_df = np.concatenate(Emg_Array_List , axis = 0 )
    Imu_df = np.concatenate(Imu_Array_List , axis = 0)
    return [Emg_df , Imu_df]
    # return [ pd.Panel(Emg_df) , pd.Panel(Imu_df)]

emg_bin_NoFiltDF_list =[]  # create empty list to store binned, SameTime, subsetted,NON-filtered, normalized EMG dfs
imu_bin_NoFiltDF_list =[]  # create empty list to store binned, SameTime,subsetted,NON-filtered IMU dfs

for (df_emg,df_imu) in itertools.zip_longest(emg_SameT_NoFiltDF_list,imu_SameT_NoFiltDF_list):
    emg_bin_NoFiltDF_list.append(binning(df_emg,df_imu,40,10)[0])
    imu_bin_NoFiltDF_list.append(binning(df_emg,df_imu,40,10)[1])

emg_bin_LowFiltNormDF_list = [] # create empty list to store binned,SameTime, subsetted,lowpass filt, normalized EMG dfs
imu_bin_LowFiltNormDF_list = [] # create empty list to store binned,SameTime, subsetted, lowpass filt, IMU dfs
for (df_filt_emg,df_filt_imu) in itertools.zip_longest(emg_SameT_LowFiltNormDF_list,imu_SameT_LowFiltNormDF_list):
    emg_bin_LowFiltNormDF_list.append(binning(df_filt_emg,df_filt_imu,40,10)[0])
    imu_bin_LowFiltNormDF_list.append(binning(df_filt_emg,df_filt_imu,40,10)[1])

## check ##
# emg_a_t = emg_SameT_LowFiltNormDF_list[1].groupby('uniqueID')
# emg_b_t = [x for _, x in emg_a_t]
# imu_a_t = imu_SameT_LowFiltNormDF_list[1].groupby('uniqueID')
# imu_b_t = [x for _, x in imu_a_t]
# emg_b_t[5].shape[0] // 40
# imu_b_t[5].shape[0] // 10
# # versus
# emg_bin_NoFiltDF_list[1].shape
# imu_bin_NoFiltDF_list[1].shape
# emg_bin_LowFiltNormDF_list[1].shape
# imu_bin_LowFiltNormDF_list[1].shape


###########
## Step 6 #
###########

# export to Matlab (cell array format)
## P12 ##
P12_Emg_NoFilt_Fin =emg_bin_NoFiltDF_list[0]
numcol_emg = P12_Emg_NoFilt_Fin.shape[2]
P12_Emg_NoFilt_Fin2 = np.array(P12_Emg_NoFilt_Fin[:,:,7:numcol_emg],dtype = np.float64)

P12_Emg_LowFilt_Fin =emg_bin_LowFiltNormDF_list[0]
P12_Emg_LowFilt_Fin2 = np.array(P12_Emg_LowFilt_Fin[:,:,7:numcol_emg],dtype = np.float64)

P12_Imu_NoFilt_Fin = imu_bin_NoFiltDF_list[0]
numcol_imu = P12_Imu_NoFilt_Fin.shape[2]
P12_Imu_NoFilt_Fin2 = np.array(P12_Imu_NoFilt_Fin[:,:,7:numcol_imu],dtype = np.float64)

P12_Imu_LowFilt_Fin =imu_bin_LowFiltNormDF_list[0]
P12_Imu_LowFilt_Fin2 = np.array(P12_Imu_LowFilt_Fin[:,:,7:numcol_imu],dtype = np.float64)

scipy.io.savemat('P97.mat', dict(P97Emg_NoFilt = P12_Emg_NoFilt_Fin2 , P97Imu_NoFilt = P12_Imu_NoFilt_Fin2,
                                 P97Emg_LowFilt = P12_Emg_LowFilt_Fin2, P97Imu_LowFilt = P12_Imu_LowFilt_Fin2))

## P63 ##
P43_Emg_NoFilt_Fin =emg_bin_NoFiltDF_list[1]
numcol_emg2 = P43_Emg_NoFilt_Fin.shape[2]
P43_Emg_NoFilt_Fin2 = np.array(P43_Emg_NoFilt_Fin[:,:,7:numcol_emg2],dtype = np.float64)

P43_Emg_LowFilt_Fin =emg_bin_LowFiltNormDF_list[1]
P43_Emg_LowFilt_Fin2 = np.array(P43_Emg_LowFilt_Fin[:,:,7:numcol_emg2],dtype = np.float64)

P43_Imu_NoFilt_Fin = imu_bin_NoFiltDF_list[1]
numcol_imu2 = P43_Imu_NoFilt_Fin.shape[2]
P43_Imu_NoFilt_Fin2 = np.array(P43_Imu_NoFilt_Fin[:,:,7:numcol_imu2],dtype = np.float64)

P43_Imu_LowFilt_Fin =imu_bin_LowFiltNormDF_list[1]
P43_Imu_LowFilt_Fin2 = np.array(P43_Imu_LowFilt_Fin[:,:,7:numcol_imu2],dtype = np.float64)

scipy.io.savemat('P98.mat', dict(P98Emg_NoFilt = P43_Emg_NoFilt_Fin2 , P98Imu_NoFilt = P43_Imu_NoFilt_Fin2,
                                 P98Emg_LowFilt = P43_Emg_LowFilt_Fin2, P98Imu_LowFilt = P43_Imu_LowFilt_Fin2))
## P52 ##
P52_Emg_NoFilt_Fin =emg_bin_NoFiltDF_list[2]
numcol_emg3 = P52_Emg_NoFilt_Fin.shape[2]
P52_Emg_NoFilt_Fin2 = np.array(P52_Emg_NoFilt_Fin[:,:,7:numcol_emg3],dtype = np.float64)
P52_Emg_LowFilt_Fin =emg_bin_LowFiltNormDF_list[2]
P52_Emg_LowFilt_Fin2 = np.array(P52_Emg_LowFilt_Fin[:,:,7:numcol_emg3],dtype = np.float64)
P52_Imu_NoFilt_Fin = imu_bin_NoFiltDF_list[2]
P52_Imu_NoFilt_Fin2 = np.array(P52_Imu_NoFilt_Fin[:,:,7:19],dtype = np.float64)
P52_Imu_LowFilt_Fin =imu_bin_LowFiltNormDF_list[2]
P52_Imu_LowFilt_Fin2 = np.array(P52_Imu_LowFilt_Fin[:,:,7:19],dtype = np.float64)

scipy.io.savemat('P96.mat', dict(P96Emg_NoFilt = P52_Emg_NoFilt_Fin2 , P96Imu_NoFilt = P52_Imu_NoFilt_Fin2,
                                 P96Emg_LowFilt = P52_Emg_LowFilt_Fin2, P96Imu_LowFilt = P52_Imu_LowFilt_Fin2))

## P63 ##
F10_Emg_NoFilt_Fin =emg_bin_NoFiltDF_list[3]
numcol_emg4 = F10_Emg_NoFilt_Fin.shape[2]
P63_Emg_NoFilt_Fin2 = np.array(F10_Emg_NoFilt_Fin[:,:,7:numcol_emg4],dtype = np.float64)
F10_Emg_LowFilt_Fin =emg_bin_LowFiltNormDF_list[3]
P63_Emg_LowFilt_Fin2 = np.array(F10_Emg_LowFilt_Fin[:,:,7:numcol_emg4],dtype = np.float64)
F10_Imu_NoFilt_Fin = imu_bin_NoFiltDF_list[3]
P63_Imu_NoFilt_Fin2 = np.array(F10_Imu_NoFilt_Fin[:,:,7:19],dtype = np.float64)
F10_Imu_LowFilt_Fin =imu_bin_LowFiltNormDF_list[3]
P63_Imu_LowFilt_Fin2 = np.array(F10_Imu_LowFilt_Fin[:,:,7:19],dtype = np.float64)

scipy.io.savemat('F09.mat', dict(F09Emg_NoFilt = P63_Emg_NoFilt_Fin2 , F09Imu_NoFilt = P63_Imu_NoFilt_Fin2,
                                 F09Emg_LowFilt = P63_Emg_LowFilt_Fin2, F09Imu_LowFilt = P63_Imu_LowFilt_Fin2))

## P63 ##
F10_Emg_NoFilt_Fin =emg_bin_NoFiltDF_list[4]
numcol_emg4 = F10_Emg_NoFilt_Fin.shape[2]
P63_Emg_NoFilt_Fin2 = np.array(F10_Emg_NoFilt_Fin[:,:,7:numcol_emg4],dtype = np.float64)
F10_Emg_LowFilt_Fin =emg_bin_LowFiltNormDF_list[4]
P63_Emg_LowFilt_Fin2 = np.array(F10_Emg_LowFilt_Fin[:,:,7:numcol_emg4],dtype = np.float64)
F10_Imu_NoFilt_Fin = imu_bin_NoFiltDF_list[4]
P63_Imu_NoFilt_Fin2 = np.array(F10_Imu_NoFilt_Fin[:,:,7:19],dtype = np.float64)
F10_Imu_LowFilt_Fin =imu_bin_LowFiltNormDF_list[4]
P63_Imu_LowFilt_Fin2 = np.array(F10_Imu_LowFilt_Fin[:,:,7:19],dtype = np.float64)

scipy.io.savemat('F10.mat', dict(F10Emg_NoFilt = P63_Emg_NoFilt_Fin2 , F10Imu_NoFilt = P63_Imu_NoFilt_Fin2,
                                 F10Emg_LowFilt = P63_Emg_LowFilt_Fin2, F10Imu_LowFilt = P63_Imu_LowFilt_Fin2))

# check
# P43_Emg_NoFilt_Fin2[0,:,:]
# P63_Emg_LowFilt_Fin2[0,:,:]
# P63_Imu_NoFilt_Fin2[0,:,:]
# P63_Imu_LowFilt_Fin2[0,:,:]

# Line 128 from 'classifycalib_Rndfrst - gived parametrs of Random Forest