#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv('application_data.csv')
pd.set_option('display.max_rows',None)
pd.set_option('display.Max_columns',None)


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.dtypes


# # find missing values

# In[7]:


df_null=df.isnull().sum()*100/len(df)
df_null


# In[8]:


len(df_null[df_null>0])


# drop the null values greater than 40%

# In[9]:


df.drop(df_null[df_null.values>40].index,inplace=True,axis=1)
df.isnull().sum()


# In[10]:


df_null


# In[11]:


df.shape


# In[12]:


len(df_null[df_null>0])


# In[13]:


df_null=list(df.isnull().sum()[df.isnull().sum()>600].index)
df_null


# In[14]:


df_null.remove('NAME_TYPE_SUITE')
df_null.remove('OCCUPATION_TYPE')
df_null.append('DAYS_LAST_PHONE_CHANGE')


# In[15]:


df.drop(df_null,axis=1,inplace=True)
df.isnull().sum()[df.isnull().sum()>0]


# In[16]:


df_col = df.filter(regex='^FLAG',axis=1).columns.tolist()
df_col


# In[17]:


df.drop(df_col,inplace=True,axis=1)


# In[18]:


df.isnull().sum()


# In[19]:


df['AMT_ANNUITY']=df['AMT_ANNUITY'].fillna(df['AMT_ANNUITY'].mean())


# In[20]:


df.isnull().sum()


# In[21]:


df['AMT_GOODS_PRICE'].describe()


# In[22]:


df['AMT_GOODS_PRICE']=df['AMT_GOODS_PRICE'].fillna(df['AMT_GOODS_PRICE'].median())
df.isnull().sum()


# In[23]:


df['NAME_TYPE_SUITE'].value_counts()


# In[24]:


df['NAME_TYPE_SUITE']=df['NAME_TYPE_SUITE'].fillna('Unaccompanied')
df.isnull().sum()


# In[25]:


df['OCCUPATION_TYPE'].value_counts()


# In[26]:


df['OCCUPATION_TYPE']=df['OCCUPATION_TYPE'].fillna('no job')
df.isnull().sum()


# In[27]:


df['CNT_FAM_MEMBERS']=df['CNT_FAM_MEMBERS'].fillna(df['CNT_FAM_MEMBERS'].mean())


# In[28]:


df.isnull().sum()


# # find any errors in data such as typos,-ve or +ve sign changes

# therefore there is no null values in the applications dataset

# In[29]:


df.head()


# In[30]:


df.shape


# In[31]:


df['CNT_FAM_MEMBERS'].nunique()


# In[32]:


df['CNT_FAM_MEMBERS'].dtypes


# In[142]:


df['CNT_FAM_MEMBERS']=df['CNT_FAM_MEMBERS'].astype('int64')
df['CNT_FAM_MEMBERS'].dtypes


# In[34]:


import numpy as np


# In[35]:


np.unique(df['CNT_FAM_MEMBERS'])


# In[36]:


df['DAYS_BIRTH']/=-365


# In[37]:


df.head()


# In[38]:


import math


# In[39]:


df['DAYS_BIRTH']=df['DAYS_BIRTH'].apply(lambda x:math.ceil(x))
df['DAYS_BIRTH'].head()


# In[40]:


num=df.select_dtypes(np.number)
num.head()


# In[41]:


cat=df.select_dtypes(exclude=np.number)
cat.head()


# In[42]:


num.shape


# In[43]:


cat.shape


# # UNIVARIATE ANALYSIS

# In[44]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[45]:


plt.hist(num['DAYS_BIRTH'],bins=15)


# # FIND OUTLIERS USING BOXPLOTS

# In[46]:


import seaborn as sns


# In[47]:


plt.figure(figsize=(18,6))
for i in range(len(num.columns)):
    sns.boxplot(data=num,y=num.columns[i])
    plt.show()


# In[48]:


print(num.CNT_CHILDREN.quantile(0.99))
len(num[num.CNT_CHILDREN>3])


# In[49]:


num[num['CNT_CHILDREN']>3]=4


# In[50]:


sns.boxplot(data=num,y='CNT_CHILDREN')


# In[51]:


num['CNT_CHILDREN'].describe()


# # IMPUTING USING QUARTILE

# In[52]:


iqr=num['AMT_INCOME_TOTAL'].quantile(0.75)-num['AMT_INCOME_TOTAL'].quantile(0.25)
out=num['AMT_INCOME_TOTAL'].quantile(0.75)+iqr*(1.5)
out


# In[53]:


num['AMT_INCOME_TOTAL'].quantile(0.99)


# In[54]:


num[num['AMT_INCOME_TOTAL']>337500.0]=337500.0+10000.0


# In[55]:


sns.boxplot(data=num,y='AMT_INCOME_TOTAL')


# In[56]:


iqr=num['AMT_CREDIT'].quantile(0.75)-num['AMT_CREDIT'].quantile(0.25)
out=num['AMT_CREDIT'].quantile(0.75)+iqr*(1.5)
out


# In[57]:


num[num['AMT_CREDIT']>1616625.0]=1616625.0+10000.0


# In[58]:


sns.boxplot(data=num,y='AMT_CREDIT')


# In[59]:


iqr=num['AMT_ANNUITY'].quantile(0.75)-num['AMT_ANNUITY'].quantile(0.25)
out=num['AMT_ANNUITY'].quantile(0.75)+iqr*(1.5)
out


# In[60]:


num[num['AMT_ANNUITY']>62145.0]=62145.0+1000.0


# In[61]:


sns.boxplot(data=num,y='AMT_ANNUITY')


# In[62]:


iqr=num['AMT_GOODS_PRICE'].quantile(0.75)-num['AMT_GOODS_PRICE'].quantile(0.25)
out=num['AMT_GOODS_PRICE'].quantile(0.75)+iqr*(1.5)
out


# In[63]:


num[num['AMT_GOODS_PRICE']>1350000.0]=1350000.0+10000.0


# In[64]:


sns.boxplot(data=num,y='AMT_GOODS_PRICE')


# In[65]:


df_T0 = df[df.TARGET == 0]
df_T1 = df[df.TARGET == 1]
df.head()


# # UNIVARIATE ANALYSIS

# In[66]:


plt.style.use('ggplot')
for column in cat:
    plt.figure(figsize=(20,4))
    plt.subplot(121)
    df_T0[column].value_counts().plot(kind='bar')
    plt.title(column)


# In[67]:


for column in num.columns:
    plt.figure(figsize=(20,4))
    plt.subplot(121)
    sns.displot(df_T0[column])
    plt.title(column)


# In[68]:


for column in num.columns:
    plt.figure(figsize=(20,4))
    plt.subplot(121)
    sns.displot(df_T1[column])
    plt.title(column)


# In[69]:


for column in cat:
    plt.figure(figsize=(30,6))
    sns.countplot(x=df[column],hue=df['TARGET'],data=df)
    plt.title(column)    
    plt.xticks(rotation=90)


# # CORRELATION 

# In[70]:


s=num.corr()
s


# In[71]:


plt.figure(figsize=(30,10))
sns.heatmap(round(s,3),annot=True)


# #  PREVIOUS DATA

# In[72]:


df1=pd.read_csv('previous_application (2).csv')
df1.head()


# In[73]:


df1.info()


# In[74]:


df1.describe(include='all')


# In[75]:


df1.shape


# In[76]:


df1_n=df1.isnull().sum()*100/len(df1)


# In[77]:


df1_n=df1_n[df1_n.values>40]
len(df1_n)


# In[78]:


df1_n


# In[79]:


df1.drop(df1[['AMT_DOWN_PAYMENT','RATE_DOWN_PAYMENT','RATE_INTEREST_PRIVILEGED','RATE_INTEREST_PRIMARY','NAME_TYPE_SUITE','DAYS_FIRST_DRAWING','DAYS_FIRST_DUE','DAYS_LAST_DUE_1ST_VERSION','DAYS_LAST_DUE','DAYS_TERMINATION','NFLAG_INSURED_ON_APPROVAL']],inplace=True,axis='columns')


# In[80]:


df1.shape


# In[81]:


mer_df=df.merge(df1,on='SK_ID_CURR',how = 'inner')
mer_df.head()


# In[82]:


mer_df.shape


# In[83]:


mer_df['NAME_CONTRACT_TYPE_x'].nunique()


# In[84]:


mer_df.info()


# In[85]:


mer_df.drop(['SK_ID_PREV','NAME_CONTRACT_TYPE_y','WEEKDAY_APPR_PROCESS_START_y','HOUR_APPR_PROCESS_START_y','FLAG_LAST_APPL_PER_CONTRACT','NFLAG_LAST_APPL_IN_DAY','DAYS_DECISION','NAME_PAYMENT_TYPE','NAME_CLIENT_TYPE','NAME_GOODS_CATEGORY','NAME_PORTFOLIO','CHANNEL_TYPE','SELLERPLACE_AREA','NAME_SELLER_INDUSTRY','CNT_PAYMENT','NAME_YIELD_GROUP','PRODUCT_COMBINATION','REGION_RATING_CLIENT','REGION_RATING_CLIENT_W_CITY','WEEKDAY_APPR_PROCESS_START_x','HOUR_APPR_PROCESS_START_x','REG_REGION_NOT_LIVE_REGION','REG_REGION_NOT_WORK_REGION','LIVE_REGION_NOT_WORK_REGION','REG_CITY_NOT_LIVE_CITY','REG_CITY_NOT_WORK_CITY','LIVE_CITY_NOT_WORK_CITY'],inplace=True,axis='columns')


# In[86]:


mer_df.shape


# In[87]:


mer_df.head()


# In[88]:


mer_df.isnull().sum()


# In[89]:


mer_df['IncomeRange']=mer_df['AMT_INCOME_TOTAL']
if mer_df['IncomeRange'].any()<157500.0:
    mer_df['IncomeRange']='Low'
else:
    mer_df['IncomeRange']='High'


# In[90]:


mer_df['IncomeRange'].head()


# In[91]:


mer_df['AMT_ANNUITY_y']=mer_df['AMT_ANNUITY_y'].fillna(mer_df['AMT_ANNUITY_y'].mean())
mer_df['AMT_GOODS_PRICE_y']=mer_df['AMT_GOODS_PRICE_y'].fillna(mer_df['AMT_GOODS_PRICE_y'].median())
mer_df['AMT_CREDIT_y']=mer_df['AMT_CREDIT_y'].fillna(mer_df['AMT_CREDIT_y'].mean())


# In[92]:


np.unique(mer_df['NAME_CONTRACT_STATUS'])


# # EFFECT OF BALANCING DATA

# In[93]:


plt.figure(figsize=(12,4))
plt.subplot(121)
mer_df['NAME_CONTRACT_STATUS'].value_counts().plot(kind='bar', color = 'green')
plt.title('NAME_CONTRACT_STATUS')
plt.show()


# In[94]:


mer_Refused = mer_df[mer_df["NAME_CONTRACT_STATUS"]  == 'Refused']
mer_Approved = mer_df[mer_df["NAME_CONTRACT_STATUS"]  == 'Approved']
mer_Canceled = mer_df[mer_df["NAME_CONTRACT_STATUS"]  == 'Canceled']
mer_Unused = mer_df[mer_df["NAME_CONTRACT_STATUS"]  == 'Unused offer']


# In[95]:


mer_Approved.columns


# In[129]:


NUM_APP=mer_Approved.select_dtypes(include=np.number)
NUM_REF=mer_Refused.select_dtypes(include=np.number)
NUM_UN=mer_Unused.select_dtypes(include=np.number)
NUM_CAN=mer_Canceled.select_dtypes(include=np.number)


# In[137]:


import warnings
warnings.filterwarnings('ignore')


# In[119]:


def eff(x,y,std):
    return abs(x-y)/std


# In[138]:


for i in range(17):
    x=NUM_APP[NUM_APP.columns[i]][NUM_APP['TARGET']==0].mean()
    y=NUM_APP[NUM_APP.columns[i]][NUM_APP['TARGET']==1].mean().mean()
    std=NUM_APP[NUM_APP.columns[i]][NUM_APP['TARGET']==1].std()
    res=eff(x,y,std)
    print('columns {},effect {}'.format(NUM_APP.columns[i],res))


# In[139]:


for i in range(17):
    x=NUM_REF[NUM_REF.columns[i]][NUM_REF['TARGET']==0].mean()
    y=NUM_REF[NUM_REF.columns[i]][NUM_REF['TARGET']==1].mean().mean()
    std=NUM_REF[NUM_REF.columns[i]][NUM_REF['TARGET']==1].std()
    res=eff(x,y,std)
    print('columns {},effect {}'.format(NUM_REF.columns[i],res))


# In[140]:


for i in range(17):
    x=NUM_UN[NUM_UN.columns[i]][NUM_UN['TARGET']==0].mean()
    y=NUM_UN[NUM_UN.columns[i]][NUM_UN['TARGET']==1].mean().mean()
    std=NUM_UN[NUM_UN.columns[i]][NUM_UN['TARGET']==1].std()
    res=eff(x,y,std)
    print('columns {},effect {}'.format(NUM_UN.columns[i],res))


# In[141]:


for i in range(17):
    x=NUM_CAN[NUM_CAN.columns[i]][NUM_CAN['TARGET']==0].mean()
    y=NUM_CAN[NUM_CAN.columns[i]][NUM_CAN['TARGET']==1].mean().mean()
    std=NUM_CAN[NUM_CAN.columns[i]][NUM_CAN['TARGET']==1].std()
    res=eff(x,y,std)
    print('columns {},effect {}'.format(NUM_CAN.columns[i],res))


# In[96]:


mer_Approved['AMT_INCOME_TOTAL'].describe()


# In[115]:


plt.scatter(mer_Approved['AMT_INCOME_TOTAL'],mer_Approved['AMT_GOODS_PRICE_y'])


# # ANALYSIS USING NAME_CONTRACT_STATUS

# In[97]:


plt.figure(figsize=(30,5))
sns.countplot(x=mer_Unused['NAME_CASH_LOAN_PURPOSE'],hue=mer_Unused['TARGET'],data=mer_Unused)
plt.xticks(rotation=90)


# THIS SHOWS THAT XAP IN OCCUPATION GIVE LOAN APPROVAL DIFFICULTIES

# In[98]:


plt.figure(figsize=(30,5))
sns.countplot(x=mer_Approved['NAME_CASH_LOAN_PURPOSE'],hue=mer_Approved['TARGET'],data=mer_Approved)
plt.xticks(rotation=90)


# SIMPLY FIND BY NAME_CASH_LOAN_PURPOSE

# In[117]:


plt.figure(figsize=(30,5))
print(sns.countplot(x=mer_Refused['NAME_CASH_LOAN_PURPOSE'],hue=mer_Refused['TARGET'],data=mer_Refused))
plt.xticks(rotation=90)


# In[100]:


plt.figure(figsize=(30,5))
sns.countplot(x=mer_Canceled['NAME_CASH_LOAN_PURPOSE'],hue=mer_Canceled['TARGET'],data=mer_Canceled)
plt.xticks(rotation=90)


# In[101]:


plt.figure(figsize=(30,5))
sns.countplot(x=mer_Approved[(mer_Approved['CODE_GENDER']=='M')&(mer_Approved['IncomeRange']=='Low')]['NAME_EDUCATION_TYPE'],hue=mer_Approved['TARGET'],data=mer_Approved)
plt.xticks(rotation=90)


# THIS PLOT SHOWS THAT SECONDARY SECTOR MALES HAVE DIFFICULTIES IN LOAN APPROVAL

# In[102]:


plt.figure(figsize=(30,5))
sns.countplot(x=mer_Approved[(mer_Approved['CODE_GENDER']=='F')&(mer_Approved['IncomeRange']=='Low')]['NAME_EDUCATION_TYPE'],hue=mer_Approved['TARGET'],data=mer_Approved)
plt.xticks(rotation=90)


# SAME FOR WOMEN ALSO.

# In[103]:


plt.figure(figsize=(30,5))
sns.countplot(x=mer_Refused[(mer_Refused['CODE_GENDER']=='M')&(mer_Refused['IncomeRange']=='Low')]['NAME_EDUCATION_TYPE'],hue=mer_Refused['TARGET'],data=mer_Refused)
plt.xticks(rotation=90)


# In[104]:


plt.figure(figsize=(30,5))
sns.countplot(x=mer_Refused[(mer_Refused['CODE_GENDER']=='F')&(mer_Refused['IncomeRange']=='Low')]['NAME_EDUCATION_TYPE'],hue=mer_Refused['TARGET'],data=mer_Refused)
plt.xticks(rotation=90)


# IN THIS MULTIVARIATE PLOT SHOWS THAT REVOLING LOANS GIVE SOME LOSSES IN BANK.SO PROVIDE A LOAN ACCORDING TO THEIR RESOURSES LIKE  CONTRACT STATUS

# In[105]:


corr=mer_df.corr()


# In[106]:


plt.figure(figsize=(30,5))
sns.heatmap(round(corr,3),annot=True)


# HEATMAP SHOWS THAT CLIENT CHILDREN AND THEIR FAMILY HAVE RELATED TO THE LOAN APPROVAL

# In[107]:


corr1=mer_Approved.corr()
corr2=mer_Refused.corr()
corr3=mer_Canceled.corr()
corr4=mer_Unused.corr()


# In[108]:


plt.figure(figsize=(30,5))
sns.heatmap(round(corr1,3),annot=True)
plt.xlabel('Loans approved')


# In[109]:


plt.figure(figsize=(30,5))
sns.heatmap(round(corr2,3),annot=True)
plt.xlabel('Loan refused')


# In[110]:


plt.figure(figsize=(30,5))
sns.heatmap(round(corr3,3),annot=True)
plt.xlabel('Loan canceled')


# In[111]:


plt.figure(figsize=(30,5))
sns.heatmap(round(corr4,3),annot=True)
plt.xlabel('Loan Unused')


# APPROVAL OF LOAN SHOULD BE PROVIDED TO FAMILY MEMBERS AND THEIR CHILD FOR EDUCATION PURPOSES OR FAMILY NEEDS
# ALSO PROVIDE TO MEN WHO ARE WELL EDUCATED AND AT SAME RETURN THEIR LOANS AT CORRECT TIME 

# THANK YOU,

# In[ ]:




