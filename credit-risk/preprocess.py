import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

norm = False
debug = False

def debug_log(*s):
    if debug:
        print(*s)

def normalize(df, col, df_train_cnt):
    if norm:
        df_train = df.iloc[:df_train_cnt]
        # std normalize
        mu = df_train[col].mean()
        sigma = df_train[col].std()
        df[col] = (df[col] - mu) / sigma
        # fill na with -9
        df[col] = df[col].fillna(-9)

def catagory_encode(df, col):
    # treat na as a special catagory
    df[col] = df[col].fillna('na_')
    # one hot or label encoder
    if len(df[col].unique()) < 16:
        new_cols = [col+': '+s for s in df[col].unique()]
        new_vals = LabelBinarizer().fit_transform(df[col])
        df[new_cols] = pd.DataFrame(new_vals, index=df.index)
        df.drop(col, axis=1, inplace=True)
        if debug:
            assert((df[new_cols].sum(1) == 1).all())
    else:
        # OCCUPATION_TYPE(19), ORGANIZATION_TYPE(35)
        df[col] = LabelEncoder().fit_transform(df[col])

for i in range(1, len(sys.argv)):
    if sys.argv[i] == 'norm':
        norm = True
    elif sys.argv[i] == 'debug':
        debug = True
    else:
        raise ValueError('Only supports "norm" or "debug"')

df_train = pd.read_csv('dataset/application_train.csv')
df_test = pd.read_csv('dataset/application_test.csv')

df_train_cnt = df_train.shape[0]
df_train_target = df_train['TARGET']
df_train = df_train.drop(['TARGET'], axis=1)

df_all = pd.concat([df_train, df_test], ignore_index=True, copy=False)
if debug:
    col = 'AMT_CREDIT'
    assert (df_train[col].values == df_all.iloc[:df_train_cnt][col]).all()
    assert (df_test[col].values == df_all.iloc[df_train_cnt:][col]).all()

del df_train
del df_test

#######################################################################

# Manually fill features
df_all['AMT_ANNUITY'].fillna(df_all['AMT_CREDIT']/21, inplace=True)

# Catagorical features
col = 'NAME_CONTRACT_TYPE'
df_all[col] = (df_all[col] == 'Cash loans')
debug_log(col, df_all[col].unique())

col = 'CODE_GENDER'
df_all[col] = (df_all[col] == 'F')
debug_log(col, df_all[col].unique())

cols = ('FLAG_OWN_CAR', 'FLAG_OWN_REALTY')
for col in cols:
    df_all[col] = (df_all[col] == 'Y')
    debug_log(col, df_all[col].unique())

col = 'NAME_TYPE_SUITE'
df_all[col] = (df_all[col] == 'Family').fillna(0)
debug_log(col, df_all[col].unique(), df_all[col].isnull().sum())

col = 'NAME_INCOME_TYPE'
df_all.loc[df_all[col].isin(['Student','Maternity leave']), col] = 'Unemployed'
df_all.loc[df_all[col] == 'Businessman', col] = 'Commercial associate'
debug_log(col, df_all[col].unique())
catagory_encode(df_all, col)

col = 'NAME_FAMILY_STATUS'
df_all.loc[df_all[col] == 'Unknown', col] = 'Single / not married'
debug_log(col, df_all[col].unique())
catagory_encode(df_all, col)

col = 'REGION_RATING_CLIENT_W_CITY'
df_all.loc[df_all[col] == -1, col] = 2
debug_log(col, df_all[col].unique())

col = 'WEEKDAY_APPR_PROCESS_START'
df_all[col] = df_all[col].apply(
    lambda x: 1 if x in ('SUNDAY','SATURDAY') else 0)
debug_log(col, df_all[col].unique())

col = 'HOUR_APPR_PROCESS_START'
df_all[col] = df_all[col].apply(lambda x: 0 if x <= 12 else 1)

col = 'ORGANIZATION_TYPE'
vals = ('Industry', 'Trade', 'Business', 'Transport')
for v in vals:
    df_all[col] = df_all[col].apply(lambda x: v if x.startswith(v) else x)
debug_log(col, len(df_all[col].unique()))
catagory_encode(df_all, col)

cols= ('NAME_EDUCATION_TYPE', 'NAME_HOUSING_TYPE', 'FONDKAPREMONT_MODE',
       'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE',
       'OCCUPATION_TYPE')
for col in cols:
    catagory_encode(df_all, col)

# Already processed
cols = ('FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE',
        'FLAG_PHONE', 'FLAG_EMAIL', 'REGION_RATING_CLIENT',
        'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
        'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',
        'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY')

# New features
col = 'CREDIT_DIV_INCOME'
df_all[col] = df_all['AMT_CREDIT'] / df_all['AMT_INCOME_TOTAL']
df_all[col] = df_all[col]
normalize(df_all, col, df_train_cnt)

col = 'ANNUITY_DIV_CREDIT'
df_all[col] = (df_all['AMT_ANNUITY'] / df_all['AMT_CREDIT'])
normalize(df_all, col, df_train_cnt)

# Numerical features
df_all['CNT_CHILDREN'] = df_all['CNT_CHILDREN'].apply(lambda x: min(x, 5))

df_all['AMT_INCOME_TOTAL'] = \
    df_all['AMT_INCOME_TOTAL'].apply(lambda x: min(x, 1e6))

df_all['DAYS_EMPLOYED'] = df_all['DAYS_EMPLOYED'].apply(lambda x: min(x, 0))

df_all.loc[df_all['OWN_CAR_AGE'] >= 60, 'OWN_CAR_AGE'] = np.nan

df_all['CNT_FAM_MEMBERS'] = df_all['CNT_FAM_MEMBERS'].fillna(2)

df_all['DAYS_LAST_PHONE_CHANGE'] = df_all['DAYS_LAST_PHONE_CHANGE'].fillna(0)

cols = (
    'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'DAYS_EMPLOYED', 'OWN_CAR_AGE',
    'CNT_FAM_MEMBERS', 'DAYS_LAST_PHONE_CHANGE', 'AMT_CREDIT', 'AMT_ANNUITY',
    'AMT_GOODS_PRICE', 'DAYS_BIRTH', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH',
    'REGION_POPULATION_RELATIVE', 'EXT_SOURCE_1', 'EXT_SOURCE_2',
    'EXT_SOURCE_3', 'APARTMENTS_AVG', 'BASEMENTAREA_AVG',
    'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG', 'COMMONAREA_AVG',
    'ELEVATORS_AVG', 'ENTRANCES_AVG', 'FLOORSMAX_AVG', 'FLOORSMIN_AVG',
    'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG',
    'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG', 'APARTMENTS_MODE',
    'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BUILD_MODE',
    'COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE', 'FLOORSMAX_MODE',
    'FLOORSMIN_MODE', 'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE',
    'LIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE',
    'APARTMENTS_MEDI', 'BASEMENTAREA_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI',
    'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI',
    'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI', 'LANDAREA_MEDI',
    'LIVINGAPARTMENTS_MEDI', 'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI',
    'NONLIVINGAREA_MEDI', 'TOTALAREA_MODE',
    'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE',
    'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE',
    'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
    'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
    'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR',
)
for col in cols:
    normalize(df_all, col, df_train_cnt)

#######################################################################

# Merge additional tables to df_all
def merge_tbl(df_all, df, cols):
    vals = df.groupby('SK_ID_CURR')[cols].sum()
    df_all = pd.merge(left=df_all, right=vals, left_on='SK_ID_CURR',
                      right_index=True, how='left')
    df_all[cols] = df_all[cols].fillna(0)
    for col in cols:
        normalize(df_all, col, df_train_cnt)
    del df
    return df_all

# bureau
df_bureau = pd.read_csv('dataset/bureau.csv')
df_bureau['CREDIT_ACTIVE'] = df_bureau['CREDIT_ACTIVE'].apply(
    lambda x: 1 if x == 'Active' else 0)
df_bureau['AMT_CREDIT_SUM'] = df_bureau['AMT_CREDIT_SUM'].apply(
    lambda x: min(x, 1e6)).fillna(0)
cols = ['CREDIT_ACTIVE', 'CREDIT_DAY_OVERDUE', 'AMT_CREDIT_SUM']
df_all = merge_tbl(df_all, df_bureau, cols)

# POS_CASH_balance
df_pos = pd.read_csv('dataset/POS_CASH_balance.csv')
cols = ['SK_DPD', 'SK_DPD_DEF']
df_all = merge_tbl(df_all, df_pos, cols)

# credit_card_balance
df_credit = pd.read_csv('dataset/credit_card_balance.csv')
df_credit.rename(columns={'SK_DPD': 'SK_DPD2', 'SK_DPD_DEF': 'SK_DPD_DEF2'},
                 inplace=True)
cols = ['AMT_DRAWINGS_CURRENT', 'CNT_DRAWINGS_CURRENT',
        'SK_DPD2', 'SK_DPD_DEF2']
df_all = merge_tbl(df_all, df_credit, cols)

# previous_application
df_prev = pd.read_csv('dataset/previous_application.csv')
df_prev['AMT_CREDIT'] = df_prev['AMT_CREDIT'].fillna(0)
df_prev.rename(columns={'AMT_CREDIT': 'AMT_CREDIT2'}, inplace=True)
cols = ['AMT_CREDIT2']
df_all = merge_tbl(df_all, df_prev, cols)

# installments_payments
df_install = pd.read_csv('dataset/installments_payments.csv')
df_install['AMT_INSTALL_BALANCE'] = \
    (df_install['AMT_INSTALMENT'] - df_install['AMT_PAYMENT']).fillna(0)
cols = ['AMT_INSTALL_BALANCE']
df_all = merge_tbl(df_all, df_install, cols)

#######################################################################

if debug:
    test_sk_id = pd.read_csv('dataset/sample_submission.csv')['SK_ID_CURR']
    assert((test_sk_id.values==df_all.iloc[df_train_cnt:]['SK_ID_CURR']).all())

df_all = df_all.drop('SK_ID_CURR', axis=1)
df_all = df_all.astype('float32')

if norm:
    assert(df_all.isnull().sum().sum() == 0)
    v = df_all.iloc[:df_train_cnt]['CNT_CHILDREN']
    assert((np.isclose([v.mean(), v.std()], [0, 1], atol=1e-3)).all())

df_train = df_all.iloc[:df_train_cnt].reset_index(drop=True)
df_test = df_all.iloc[df_train_cnt:].reset_index(drop=True)
df_train['TARGET'] = df_train_target.values

if debug:
    print(len(df_train.columns))
    print(list(df_train.columns))

if norm:
    with open('dataset/df_train_norm.pkl', 'wb') as f:
        pickle.dump(df_train, f)
    with open('dataset/df_test_norm.pkl', 'wb') as f:
        pickle.dump(df_test, f)
else:
    with open('dataset/df_train.pkl', 'wb') as f:
        pickle.dump(df_train, f)
    with open('dataset/df_test.pkl', 'wb') as f:
        pickle.dump(df_test, f)
