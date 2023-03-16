#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd # Data manipulation and data preparation
import numpy as np # for imputation methods
import matplotlib.pyplot as plt # visualization 
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


cops=pd.read_csv(r"C:\Users\Divyashree K\Downloads\police_deaths_in_america.csv\police_deaths_in_america.csv")


# In[3]:


cops


# In[4]:


cops.describe()


# In[5]:


cops.isnull().sum()


# In[6]:


cops.duplicated().sum()


# In[7]:


cops.info()


# In[8]:


cops.head()


# In[9]:


cops.shape


# ## K9 unit is combined with it,let's separated them and do only police analysis first

# In[10]:


police=cops[cops["K9_Unit"]==0].reset_index(drop=True)
police.drop("K9_Unit",axis=1,inplace=True)
police


# ## There are 25786 cases are registered from 1791 to 2022.

# In[11]:


# counting death count
neg= ['Name','Cause_of_Death','Date','Day','Month','Department','State']
death_count=police.groupby('Year').count().drop(neg,axis=1)
death_count=death_count.rename(columns={'Rank':'Count'})
death_count


# In[12]:


death_count.describe()


# In[13]:


maximum=death_count.max()
maximum_death_year=death_count[death_count['Count']==int(maximum)]
maximum_death_year


# In[14]:


minimum=death_count.min()
minimum_death_year=death_count[death_count['Count']==int(minimum)]
print("Year's which had low cases: ",minimum_death_year.index)


# In[15]:


# Average death cases per year
print(round(np.mean(death_count['Count'])))


# In[16]:


# Top10 years cops are died
print(death_count.sort_values(by='Count',ascending=False).head(10))


# ## Above data clearly shows us in 2021,2020 spike around 70% compared to other year's.

# In[17]:


sns.lineplot(death_count)


# In[18]:


plt.figure(figsize=(20,5))

plot_1 = sns.lineplot(data=death_count)

plt.suptitle("Registered Death Cases",fontsize=30)
plt.title("1791-2022",fontsize=20)

plt.xlabel('Year',fontsize=20)
plt.ylabel('Deaths',fontsize=20)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.fill_between(death_count.index,death_count['Count'],color='Green',alpha=.3)

plt.show()


# ## There is some sudden spike in  deaths above 2018

# In[19]:


deaths_2019=police[police['Year']>2018]
deaths_2019


# In[20]:


neg2=['Name','Date','Year','Month','Day','Department','State']
reason_death=deaths_2019.groupby(by='Cause_of_Death').count().drop(neg2,axis=1)
#reason_death=reason_death.rename(columns={"Rank":"Count"})
reason_death.sort_values('Rank',ascending=False).head(10)


# ## let's see  overall deaths from 1792 to 2022

# In[21]:


over=['Name', 'Date', 'Year', 'Month', 'Day',
       'Department', 'State']
overall_de=police.groupby(by='Cause_of_Death').count().drop(over,axis=1)
#overall_de=overall_de.rename(columns={'Rank':'Count'})
overall_de.sort_values('Rank',ascending=False).head(10)


# In[22]:


print(type(overall_de))


# ## Compare to all, Gunfire deaths are high . Let's us know about cummulative frequency.

# In[23]:


lis=[]

for i in overall_de["Rank"]:
    cummulative=round((i/overall_de['Rank'].sum())*100,2)
    lis.append(cummulative)
overall_de["Cummulative frequency%"]=lis
overall_de.sort_values(by="Cummulative frequency%",ascending=False)


# ## More than 50% deaths held due to Gunfire.

# In[24]:


# Let's see whether monthwise analysis of death.

neg_3=['Rank','Name','Cause_of_Death','Date','Year','Day','Department']
month=police.groupby(by='Month').count().drop(neg_3,axis=1)
month=month.rename(columns={'State':'Count'})
month=month.reindex(['January','February','March','April','May','June','July','August','September','October','November','December'])
month


# In[25]:


month.sort_values(by='Count',ascending=False)


# ## September and December months are top 2 months cases registered.

# In[26]:


plt.figure(figsize=(15,5))

plot_2=sns.barplot(month,x=month['Count'],y=month.index,palette='rocket')

plt.suptitle("Monthwise Cases")
plt.title("months_data")

plt.xlabel("Count",fontsize=20)
plt.ylabel("months",fontsize=20)

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plot_2.bar_label(plot_2.containers[0],fontsize=10)

plt.show()


# ## How days impacted on deaths:-

# In[27]:


neg_4=['Rank','Name','Cause_of_Death','Date','Year','Department','State']
days=police.groupby(by='Day').count().drop(neg_4,axis=1)
days=days.rename(columns={'Month':'Count'})
days.sort_values('Count',ascending=False)


# ## Sunday had deadliest cases.

# In[28]:


plt.figure(figsize=(20,5))

plot_3=sns.barplot(days,x=days.index,y=days['Count'],palette='rocket')

plt.suptitle("Days_wise data",fontsize=15,color='Green')

plt.xlabel("Days",fontsize=20,color='red')
plt.ylabel("Count",fontsize=20,color='red')

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plot_3.bar_label(plot_3.containers[0],fontsize=15,color='black')

plt.show()


# In[29]:


# unique police rank's
len(police['Rank'].unique())


# ## There are 614 unique types of police rank are available.

# In[30]:


neg_5=['Name','Day','Month','Date','State','Department','Cause_of_Death']
uniq=police.groupby(by='Rank').count().drop(neg_5,axis=1)
uniq=uniq.sort_values('Year',ascending=False).head(10).rename(columns={'Year':'Count'})
uniq


# In[31]:


plt.figure(figsize=(20,10))

plot_4=sns.barplot(data=uniq,x=uniq['Count'],y=uniq.index,palette='rocket')


plt.title("Rank_wise Analysis",color='Green',fontsize=20)

plt.xlabel("No_of_cases_registered",fontsize=20)
plt.ylabel("Top 10 unique Rank's",fontsize=20)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plot_4.bar_label(plot_4.containers[0],fontsize=15)
plt.show()


# In[32]:


neg_5=['Name','Month','Date','State','Department','Cause_of_Death']
uniq1=police.groupby(by=['Day','Rank']).count().drop(neg_5,axis=1)
uniq1=uniq1.sort_values('Year',ascending=False).rename(columns={'Year':'Count'}).head(10)
uniq1


# ## We can clearly observed that, weekdays had high cases are registered compared to other day's.

# ## Top 10 departments cases of death

# In[33]:


neg_6=['Name','Day','Month','Date','State','Cause_of_Death','Rank']
dep=police.groupby(by='Department').count().drop(neg_6,axis=1)
dep=dep.sort_values('Year',ascending=False).head(10).rename(columns={'Year':'Count'})
dep


# ## New York city sitted on the top place.

# # state

# In[34]:


neg_7=['Name','Day','Month','Date','Rank','Department','Cause_of_Death']
state=police.groupby(by='State').count().drop(neg_7,axis=1)
state=state.sort_values('Year',ascending=False).head(10).rename(columns={'Year':'Count'})
state


# In[35]:


plt.pie(x=state['Count'],data=state,labels=state.index,autopct='%1.1f%%')


# ## new york, united states and texas taken almost 50% cases compared to others.

# # Let's look at police dog's analysis.

# ### A police dog, also known as K-9 or K9 (a homophone of canine), is a dog specifically trained to assist members of law enforcement. Dogs have been used in law enforcement since the Middle Ages. The most commonly used breeds are German Shepherds and Belgian Malinois, but several other breeds are represented having some unique talents. Basset Hounds, Bloodhounds, and Labrador Retrievers, for example, are known for their tracking, trailing, and detection skills. Used as a means of law enforcement widely across the United States, Police K-9s usually serve in the force for 6 to 9 years. In many countries, the intentional injuring or killing of a police dog is a criminal offense.

# ## For more information please go through this link:-https://www.nationalpolicedogfoundation.org/about-k9s#:~:text=A%20police%20dog%2C%20also%20known,enforcement%20since%20the%20Middle%20Ages.
# 

# In[36]:


k9=cops[cops['K9_Unit']==1]
k9


# In[37]:


len(k9['Name'].unique()) # There are 383 unique name for police dog's


# In[38]:


delete_1=['Rank','Name','Cause_of_Death','Date','Month','Day','Department','K9_Unit']
total_death=k9.groupby(by='Year').count().drop(delete_1,axis=1).rename(columns={'State':'Total_deaths'}).sort_index(ascending=False)
total_death.head(10)


# In[39]:


sns.lineplot(total_death)


# ##  total_deaths are positive skewed  but not follows normal distribution.

# In[40]:


# let's see,which year had highest deaths of police dogs
delete_2=['Rank','Name','Cause_of_Death','Date','Month','Day','Department','K9_Unit']
total_year=k9.groupby(by='Year').count().drop(delete_2,axis=1).rename(columns={'State':'Total_deaths'}).sort_values('Total_deaths',ascending=False)
print("Total deaths dogs :",sum(total_year['Total_deaths']))
total_year=total_year.head(10)
total_year


# In[41]:


plt.figure(figsize=(20,10))

plot_1=sns.barplot(data=total_year,x=total_year.index,y=total_year['Total_deaths'],palette='rocket')

plt.title("Total deaths in year_wise",fontsize=20)

plt.xlabel("Year",fontsize=18)
plt.ylabel("Total_deaths",fontsize=18)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plot_1.bar_label(plot_1.containers[0],fontsize=15)

plt.show()


# ## 2016 was diedliest year for police dog's ,Let's us know why that year is so high.

# In[42]:


deadliest=k9[k9['Year']==2016]
delete_2=['Rank','Name','Date','Month','Day','Department','State','K9_Unit']
deadliest_year=deadliest.groupby(by='Cause_of_Death').count().drop(delete_2,axis=1).rename(columns={'Year':'Total_deaths'}).sort_values('Total_deaths',ascending=False)
deadliest_year


# In[43]:


plt.pie(x=deadliest_year['Total_deaths'],labels=deadliest_year.index,autopct='%1.1f%%')


# ### Heatstroke and Gunfire are those reason for most of the dog's death.

# # Department

# In[44]:


delete_3=['Rank','Name','Cause_of_Death','Date','Year','Month','State','K9_Unit']
depart=k9.groupby(by='Department').count().drop(delete_3,axis=1).rename(columns={'Day':'total_deaths'}).sort_values('total_deaths',ascending=False)
depart=depart.head(10)
depart


# # State

# In[45]:


delete_4=['Rank','Name','Cause_of_Death','Date','Year','Month','Department','K9_Unit']
state=k9.groupby(by='State').count().drop(delete_4,axis=1).rename(columns={'Day':'total_deaths'}).sort_values('total_deaths',ascending=False)
state=state.head(10)
state


# In[46]:


sns.lineplot(state,x=state['total_deaths'],y=state.index,)


# In[47]:


k9


# In[48]:


dele_3=['Rank','Name','Date','Month','Year','Department','State','K9_Unit']
cause=k9.groupby(by='Cause_of_Death').count().drop(dele_3,axis=1)
cause=cause.rename(columns={'Day':'Total_Deaths'}).sort_values('Total_Deaths',ascending=False)
cause


# In[49]:


plt.figure(figsize=(20,10))

pl=sns.barplot(data=cause,x=cause['Total_Deaths'],y=cause.index,palette='rocket')

plt.suptitle("Deaths reasons for dog's",fontsize=25,color='Green')

plt.xlabel('Total_Deaths',fontsize=20,color='red')
plt.ylabel('Cause_for_death',fontsize=20,color='red')

plt.xticks(fontsize=15,color='blue')
plt.yticks(fontsize=15,color='blue')

pl.bar_label(pl.containers[0],fontsize=15)

plt.show()


# ## Gunfire had deadliest reason's for dog's deaths 

# In[50]:


from pivottablejs import pivot_ui


# In[51]:


pivot_ui(k9)


# In[66]:


## Cause of deaths 
lis=[]
for i in cause['Total_Deaths']:
    cum=round((i/cause['Total_Deaths'].sum()*100),2)
    lis.append(cum)
cause['Cummulative _percent_%']=lis
cause


# # Thank you

# In[ ]:




