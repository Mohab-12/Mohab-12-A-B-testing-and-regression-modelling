#!/usr/bin/env python
# coding: utf-8

# ## Analyze A/B Test Results
# 
# 
# ## Table of Contents
# - [Introduction](#intro)
# - [Part I - Probability](#probability)
# - [Part II - A/B Test](#ab_test)
# - [Part III - Regression](#regression)
# 
# 
# <a id='intro'></a>
# ### Introduction
# 
# It is an A/B test run by an e-commerce website.  my goal is to work through this notebook to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.
# 
# <a id='probability'></a>
# #### Part I - Probability
# 
# ><font color='blue'>let's import our libraries.

# In[1]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)


# `1.`<font color='blue'> Now, store the data of `ab_data.csv` in df.

# In[2]:


df=pd.read_csv(r'C:\Users\Carnival\AppData\Local\Temp\Rar$DIa7972.1854\ab_data.csv')


# <font color='blue'>The number of rows in the dataset.

# In[3]:


df.head()


# <font color='blue'>The number of unique users in the dataset.

# In[4]:


df.user_id.unique


# <font color='blue'>The proportion of users converted.

# In[5]:


df.converted.mean()


# <font color='blue'> The number of times the `new_page` and `treatment` don't line up.

# In[6]:


df.query("landing_page=='new_page' & group!='treatment'")


# <font color='blue'> Do any of the rows have missing values?

# In[7]:


df.isnull().sum()


# `2.`<font color='blue'>For the rows where **`treatment`** is not aligned with **`new_page`** or **`control`** is not aligned with **`old_page`**, we cannot be sure if this row truly received the new or old page. We will create a new dataset where will have the consistent values we are looking for. 

# In[8]:


df2=pd.DataFrame()
df2=df.query('group=="treatment" & landing_page=="new_page" | group=="control" & landing_page=="old_page"')


# <font color='blue'>Double Check all of the correct rows were removed - this should be 0
# 

# In[9]:


df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]


# <font color='blue'> How many unique **user_id**s are in **df2**?

# In[10]:


df2.user_id.nunique()


# <font color='blue'>There is one **user_id** repeated in **df2**.  What is it?

# In[11]:


df2[df2.duplicated(subset='user_id')]


# <font color='blue'>What is the row information for the repeat **user_id**? 

# In[12]:


df2[df2.duplicated(subset='user_id')]


# <font color='blue'> Removal solely one of the duplicate values.

# In[13]:


df2.drop_duplicates(subset='user_id')


# 
# <font color='blue'> What is the probability of an individual converting regardless of the page they receive?

# In[14]:


df2.converted.mean()


# <font color='blue'> Given that an individual was in the `control` group, what is the probability they converted?

# In[15]:


df2.query("group=='control'")['converted'].mean()


# <font color='blue'>Given that an individual was in the `treatment` group, what is the probability they converted?

# In[16]:


df2.query("group=='treatment'")['converted'].mean()


# <font color='blue'> What is the probability that an individual received the new page?

# In[17]:


df2.query('landing_page=="new_page"')['landing_page'].count() / df2.shape[0]


# # <font color='black'>Conclusion

# <font color='red'>*According to the number investigated so far, the probability of landing on the new page is not high enough, and the probability of conversion. On the other hand, the conversion rate of the control group is slightly higher than the tratment group. In short, it needs further investigation.*

# <a id='ab_test'></a>
# ### Part II - A/B Test
# 

# <font color='red'> 
# 
# $H_{o}$= $p_{old}$ $\geq$  $p_{new}$
# 
# $H_{A}$= $p_{old}$<$p_{new}$

# <font color='blue'>What is the **convert rate** for $p_{new}$ under the null? 

# In[18]:


df2.query("landing_page=='new_page'")['converted'].mean()


# <font color='blue'> What is the **convert rate** for $p_{old}$ under the null? <br><br>

# In[19]:


df2.query("landing_page=='old_page'")['converted'].mean()


# <font color='blue'> What is $n_{new}$?

# In[20]:


df2.query("group=='treatment'")['group'].count()


# <font color='blue'> What is $n_{old}$?

# In[21]:


df2.query("group=='control'")['group'].count()


# <font color='blue'> Simulation of $n_{new}$ transactions with a convert rate of $p_{new}$ under the null.  

# In[22]:


new_page_converted=[]
for i in range(1000):
    sample=df2.sample(df2.shape[0], replace=True)
    a=sample.query("group=='treatment'")['converted'].mean()
    new_page_converted.append(sample.query("group=='treatment'")['converted'].mean())


# <font color='blue'> Simulation of $n_{old}$ transactions with a convert rate of $p_{old}$ under the null.

# In[23]:


old_page_converted=[]
for i in range(1000):
    sample2=df2.sample(df2.shape[0], replace=True)
    old_page_converted.append(sample2.query("group=='control'")['converted'].mean())


# <font color='blue'> $p_{new}$ - $p_{old}$ for the simulated values.

# In[24]:


np.array(new_page_converted) - np.array(old_page_converted);


# <font color='blue'> Simulation of 1000 $p_{new}$ - $p_{old}$ values using this same process similarly to the one you calculated previously.

# In[25]:


p_diffs=[]
for i in range(1000):
    sample3=df2.sample(df2.shape[0], replace=True)
    sample3.query("group=='treatment'")['converted'].mean()
    sample3.query("group=='control'")['converted'].mean()
    p_diffs.append(sample3.query("group=='treatment'")['converted'].mean()-
                   sample3.query("group=='control'")['converted'].mean())


# <font color='blue'> A histogram of the **p_diffs**.

# In[51]:


plt.style.use('seaborn')
g=plt.hist(p_diffs);
obs_diffs=df2.query("group=='treatment'")['converted'].mean()-df2.query("group=='control'")['converted'].mean()
h=plt.axvline(x=obs_diffs, color='r')
obs_diffs


# <font color='blue'> What proportion of the **p_diffs** are greater than the actual difference observed in **ab_data.csv**?

# In[52]:


plt.hist(np.random.normal(0,np.std(p_diffs),1000))
plt.axvline(x=obs_diffs, color='r', lw=3)
plt.axvspan(obs_diffs,(np.random.normal(0,np.std(p_diffs),1000)).max(),color='r', alpha=0.3)
plt.text(0.0015,190,"The p_value\nis {}".format((np.random.normal(0,np.std(p_diffs),1000)>obs_diffs).mean()),
         color='r', fontsize=15,weight='bold');


# # Conclusion
# <font color='red'>*The **P_value** is greater than 0.05, thereby it is evidently under the null hypothesis we can observe a value equal or extremer than in the alternative. We fail to reject the null hypothesis.*

# <font color='blue'> Another simplified way to calculate the p_value

# In[28]:


import statsmodels.api as sm
convert_old = df2.query("landing_page=='old_page' & converted==1")['converted'].count()
convert_new = df2.query("landing_page=='new_page' & converted==1")['converted'].count()
n_old = df2.query("landing_page=='old_page'")['landing_page'].count()
n_new = df2.query("landing_page=='new_page'")['landing_page'].count()
convert_old, convert_new, n_old, n_new


# <font color='blue'>  Now using `stats.proportions_ztest` to compute your test statistic and p-value.

# In[29]:


from statsmodels.stats.proportion import proportions_ztest
pvalue = sm.stats.proportions_ztest([convert_new, convert_old], [n_new, n_old], alternative='larger')
pvalue


# # Conclusion
# 
# <font color='red'>*We got similar p_value which emphasis on our conclusion.*

# <a id='regression'></a>
# ### Part III - A regression approach
# 
# `1.` In this final part, we will see that the result acheived in the previous A/B test can also be acheived by performing regression.<br><br>
# 
# <font color='blue'>what type of regression should you be performing in this case?

# <font color='red'>*Logisitc.*

# <font color='blue'>​Using **statsmodels** to fit the regression model specified to see if there is a significant difference in conversion based on which page a customer receives.  However, I need to create a column for the intercept, and create a dummy variable column for which page each user received. 

# In[43]:


import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'


df2['intercept']=1;
df2[['new','old']]=pd.get_dummies(df2['landing_page']);
df2[['control','ab_page']]=pd.get_dummies(df2['group']);


# <font color='blue'> Using **statsmodels** to import your regression model anf fit it.

# In[44]:


import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

x=df2[['intercept','ab_page']];
y=df2['converted'];
model=sm.Logit(y,x).fit();


# <font color='blue'> Provide the summary of the model below, and use it as necessary to answer the following questions.

# In[45]:


import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

model.summary2()


# <font color='red'>*The regression implies a null hypothesis for the regression parameters which says that the coeffiencts of the these parameters are zero. The values we have in the summary for the p_values enforce impel us to accept the null hypothesis. None of the parameters invesitaged so far has a big influence on the regression model (conversion number).*

# <font color='blue'> Now along with testing if the conversion rate changes for different pages, we will check the effect based on which country a user lives. We will read in the **countries.csv** dataset and merge together your datasets on the approporiate rows.

# In[46]:


countries_df = pd.read_csv(r'C:\Users\Carnival\AppData\Local\Temp\Rar$DIa7972.8050\countries.csv')
df_new = countries_df.set_index('user_id').join(df2.set_index('user_id'), how='inner')


# <font color='blue'>Checking values in the countries column

# In[47]:


df_new['country'].value_counts()


# <font color='blue'>Create the necessary dummy variables

# In[48]:


df_new['intercept']=1;
df_new[['new','old']]=pd.get_dummies(df_new['landing_page']);
df_new[['control','ab_page']]=pd.get_dummies(df_new['group']);
df_new[['US','UK','CA']]=pd.get_dummies(df_new['country']);
df_new['US_ab_page']=df_new['US']*df_new["ab_page"]
df_new['UK_ab_page']=df_new['UK']*df_new["ab_page"]
x_new=df_new[['intercept','US_ab_page','UK_ab_page','ab_page','US','UK']]
y_new=df_new['converted']


# h. Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if there significant effects on conversion.  Create the necessary additional columns, and fit the new model.  
# 
# Provide the summary results, and your conclusions based on the results.

# In[49]:


model2=sm.Logit(y_new,x_new).fit()
model2.summary()


# # Conclusion
# <font color='red'>*All the p_values as demonstrated before are above 0.05 which make us fail to reject the null hypothesis. Additionally, the model does not tell us much about the conversion rate according to the parameters included. In short, we dont have enough evidence to support the new page or evict the old one.*
