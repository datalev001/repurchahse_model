
 trans_data = pd.read_csv('trans_data.csv')
 trans_data['transaction_date'] = pd.to_datetime(trans_data['transaction_date'] , format='%Y-%m-%d')

date1 = datetime.strptime('2011-01-01', '%Y-%m-%d')
date2 = datetime.strptime('2012-05-01', '%Y-%m-%d')
date3 = datetime.strptime('2012-11-30', '%Y-%m-%d')

filter1 = (trans_data['transaction_date'] >= date1)
filter2 = (trans_data['transaction_date'] <= date2)
filter_train = (filter1 & filter2)
# get transaction data for creating train data
trans_data_train = trans_data[filter_train]
filter3 = (trans_data['transaction_date'] > date2)
filter4 = (trans_data['transaction_date'] <= date3)
filter_test = (filter3 & filter4)
# get transaction data for creating testing data
trans_data_test = trans_data[filter_test]

# Get features from the transaction data for the training dataset

# Calculate Recency
train_Recency = trans_data_train.groupby('customer_id')\
                ['transaction_date'].max().reset_index()
train_Recency['Recency'] = (date2 - train_Recency['transaction_date']).dt.days
train_Recency = train_Recency[['customer_id', 'Recency']]

# Calculate early_days
train_early_days = trans_data_train.groupby('customer_id')\
           ['transaction_date'].min().reset_index()
train_early_days['early_days'] = (date2 - train_early_days['transaction_date']).dt.days
train_early_days = train_early_days[['customer_id', 'early_days']]

# Calculate Frequency
trans_data_train['Freq'] = 1
train_freq = trans_data_train.groupby('customer_id')\
    ['Freq'].sum().reset_index()
train_freq = train_freq[['customer_id', 'Freq']]

# Calculate quantity for different time periods (e.g., 10, 30, 60, 90 days)
c = ((date2 - trans_data_train['transaction_date']).dt.days <= 10)
train_10_days = trans_data_train[c].groupby('customer_id')\
    ['quantity'].sum().reset_index()
train_10_days.columns = ['customer_id', 'quantity_10days']

c = ((date2 - trans_data_train['transaction_date']).dt.days <= 30)
train_30_days = trans_data_train[c].groupby('customer_id')\
    ['quantity'].sum().reset_index()
train_30_days.columns = ['customer_id', 'quantity_30days']

c = ((date2 - trans_data_train['transaction_date']).dt.days <= 60)
train_60_days = trans_data_train[c].groupby('customer_id')\
    ['quantity'].sum().reset_index()
train_60_days.columns = ['customer_id', 'quantity_60days']

c = ((date2 - trans_data_train['transaction_date']).dt.days <= 90)
train_90_days = trans_data_train[c].groupby('customer_id')\
    ['quantity'].sum().reset_index()
train_90_days.columns = ['customer_id', 'quantity_90days']

##get targets in test data using the transaction data

# Initialize an empty list to store the resulting data frames
test_dfs = []

# Define the start date
start_date = '2012-05-02'

test_dfs = []
start_date = '2012-05-02'
for it in range(10, 100, 10):
    c = ((trans_data_test['transaction_date'] - pd.to_datetime(start_date)) \
         .dt.days <= it)
    test_df = trans_data_test[c].groupby('customer_id')['quantity'] \
             .mean().reset_index()
    col = 'target_' + str(it) + 'days'
    test_df[col] = (test_df['quantity'] > 10).astype(int)
    print (test_df[col].mean())
    test_df = test_df[['customer_id', col]]
    test_dfs.append(test_df)


LST = [train_Recency, train_early_days, train_freq,
       train_10_days, train_30_days, train_60_days, train_90_days]
       
LST.extend(test_dfs)       

len(LST)

# here is the function to merge the all 
# data frames in a python list based on a key column
# This function merges data frames within a Python list based on a specified key column
def custom_merge(dflist, keys, merge_style):
    # Define a function to check if all keys are present in a list
    def contain_keys(keys, lst):
        x = True
        for item in keys:
            if (item in lst) == False:
                x = False
        return x     

    k = 0    
    result_df = pd.DataFrame([])
    for temp_df in dflist:
        if k == 0:
            result_df = temp_df.copy()
        else: 
            keys_in_result = contain_keys(keys, list(result_df.columns))
            keys_in_temp = contain_keys(keys, list(temp_df.columns))
            
            if (
                len(temp_df) > 0 and 
                len(result_df) > 0 and 
                keys_in_temp and 
                keys_in_result
            ): 
                result_df = pd.merge(
                    temp_df, 
                    result_df, 
                    on=keys, 
                    how=merge_style
                )
        k = k + 1
    return result_df

train_test = cosemerge(LST, ['customer_id'], 'outer')
train_test = train_test.fillna(0)
print (list(train_test.columns))


# columns names in the data
print (list(train_test.columns)) 
['customer_id', 'target_90days', 'target_80days', 
 'target_70days', 'target_60days', 'target_50days',
  'target_40days', 'target_30days', 'target_20days',
  'target_10days', 'quantity_90days', 'quantity_60days',
 'quantity_30days', 'quantity_10days', 'Freq', 
 'early_days', 'Recency']

# columns row number in the data
print (train_test.shape)
(118444, 17)

# total number of missing observations in the data
print (train_test.isnull().sum().sum())
0 # no missing value in the data

# total number of unique customers the data
print (len(set(train_test.customer_id)))
118226  # row number = numberof unique customer

########modeling#####

cus_prom = pd.read_csv(r'C:\lsg\专利2023\cus_prom.csv')

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp

train_test_prom = pd.merge(train_test, cus_prom , \
                  on = ['customer_id'], how = 'inner')

features_new = ['Recency', 'Freq', 'early_days', 'quantity_10days',
                'quantity_30days', 'quantity_60days', 
                'quantity_90days', 'promotion']

targets = ['target_90days', 'target_80days', 'target_70days',
           'target_60days', 'target_50days', 'target_40days', 
           'target_30days', 'target_20days',  'target_10days']

train_test_prom[['target_90days', 'target_80days', 'target_70days']].head(100)

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
}

Y = train_test_prom['target_30days']
X_train, X_test, y_train, y_test = train_test_split\
    (train_test_prom, Y, test_size=0.4, random_state=11)     
    
performance_lst = []
score_df_lst = []
scores = []
for target in targets:
    y_train, y_test = X_train[target], X_test[target]
    train_data = lgb.Dataset(X_train[features_new], \
                             label=X_train[target])
    model = lgb.train(params, train_data)
    X_test['score_'+ target] = model.predict(X_test[features_new])
    sc_df = X_test[['customer_id', 'promotion', 'score_'+ target]]
    auc = roc_auc_score(X_test[target].astype(int), \
                        X_test['score_'+ target])  
    ks = ks_stat(X_test[target].astype(int), \
                 X_test['score_'+ target])
    performance = ['ev_auc', auc, ks]
    performance_lst.append(performance)
    score_df_lst.append(sc_df)
    print ('auc and ks for target:' + target , [auc, ks])
    scores.append('score_'+ target)
    
result_df = cosemerge(score_df_lst, \
                     ['customer_id', 'promotion'], 'outer')
result_df = result_df.fillna(0)

c = list(result_df.columns)
c.remove('customer_id')
c.remove('promotion')

'''
result_df[c] = (result_df[c] - result_df[c].min())/\
                (result_df[c].max() - result_df[c].min())
'''

for c1 in c:
    result_df[c1] = result_df[c1].round(4)
    

result_df[['score_target_10days',
 'score_target_20days',
 'score_target_30days',
 'score_target_40days']].head(10)


DD = result_df.groupby('promotion')[c].mean().reset_index()    

import matplotlib.pyplot as plt
import pandas as pd

# Sample data

df = DD.copy()

# Extract the values for 'Group A' and 'Group B'
group_a_values = df[df['promotion'] == 1][c].values[0]
group_b_values = df[df['promotion'] == 0][c].values[0]

# X-axis values
x = list(range(1, len(c) + 1))

# Create the plot
plt.plot(x, group_a_values, marker='o', color='blue', label='promotion')
plt.plot(x, group_b_values, marker='o', color='red', label='no promotion')

# Add labels and legend
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()

# Show the plot
plt.show()

################

c = list(result_df.columns)
c.remove('customer_id')
c.remove('promotion')

result_df['score_avg'] = result_df[scores].mean(axis = 1)
result_df[result_df.promotion == 1]['score_avg'].mean()
result_df[result_df.promotion == 0]['score_avg'].mean()
result_df.score_avg.quantile([j*0.05 for j in range(20)])

result_df['max_reach'] = result_df[c].idxmax(axis=1)
result_df['max_reach_days'] = \
    result_df['max_reach'].str.slice(13,15).astype(int)
result_df[result_df.promotion == 1]['max_reach_days'].mean()
result_df[result_df.promotion == 0]['max_reach_days'].mean()

result_df_q = result_df[result_df.score_avg>0.09]
result_df_q['max_reach'] = result_df_q[c].idxmax(axis=1)
result_df_q['max_reach_days'] = \
    result_df_q['max_reach'].str.slice(13,15).astype(int)
result_df_q[result_df_q.promotion == 1]['max_reach_days'].mean()
result_df_q[result_df_q.promotion == 0]['max_reach_days'].mean()

result_df_nq = result_df[result_df.score_avg<=0.01]
result_df_nq['max_reach'] = result_df_nq[c].idxmax(axis=1)
result_df_nq['max_reach_days'] = \
    result_df_nq['max_reach'].str.slice(13,15).astype(int)
result_df_nq[result_df_nq.promotion == 1]['max_reach_days'].mean()
result_df_nq[result_df_nq.promotion == 0]['max_reach_days'].mean()


S = [['All' , 60, 60], ['Low Score', 65, 61], ['High Score', 56, 56]]
S_DF = pd.DataFrame(S, columns = ['Score', 'promotion', 'non_promotion'])

'''
  Score       promotion  non_promotion
   All         60             60
   Low Score   65             61
   High Score  56             56
'''

import matplotlib.pyplot as plt
import numpy as np

# Data
categories = ['All', 'Low Score', 'High Score']
promotion_values = [60, 65, 56]
non_promotion_values = [60, 61, 56]

# Set the figure size
plt.figure(figsize=(10, 6))

# Create an array for the x-axis positions
x = np.arange(len(categories))

# Set the width of the bars
bar_width = 0.35

# Create the bar chart for promotion values
plt.bar(x - bar_width/2, promotion_values, bar_width, label='Promotion')
# Create the bar chart for non_promotion values
plt.bar(x + bar_width/2, non_promotion_values, bar_width, label='Non-Promotion')

# Set the x-axis labels
plt.xticks(x, categories)

# Add a legend
plt.legend()

# Set the chart title and adjust the font size
plt.title('AVG Time Customer Reach Peak Repurchase Probability', fontsize=16)

# Show the plot
plt.show()
