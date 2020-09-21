#!/usr/bin/env python
# coding: utf-8

from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import pandas
from sqlalchemy import types, create_engine
import pandas as pd
import numpy as np
import random as rnd

# visualization
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import squarify
import seaborn as sns

from pandas import ExcelWriter
from pandas import ExcelFile

#def stock_list(stock_file):
    
    #Iterate through a saved list
    

def update_ticker(ticker, category):
    
    # Your key here
    key1 = '8PKIQC1JK2AO5NRS'
    
    ts = TimeSeries(key=key1, output_format='pandas')
    data, metadata = ts.get_weekly(symbol=ticker_inp)
    
    stock_data=pandas.DataFrame(data)
    '''  This adds a column to differentiate the stock ticker  '''
    stock_data['ID']=ticker
    stock_data['Category']=category
    
    return stock_data, metadata

'''This is the start of the program'''

print('please enter the password to the database')

password = input()

print('What is the Database name?')

db_name = input()

print('What is the table name?')
table_name = input()



'''This enables the connection to the database'''

engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user="root",
                               pw=password,
                               db=db_name))



print('What is the ticker')
ticker_inp =input()

print('What is the category?')
category_inp =input()


stock_data=(update_ticker(ticker_inp,category_inp))
print (stock_data[0])

# Insert whole DataFrame into MySQL
stock_data[0].to_sql('stock_data', con = engine, if_exists = 'append', chunksize = 1000)


#Read sql query into a dataframe

sql ="SELECT * FROM "+db_name+"."+table_name+";"

plot_table = pandas.read_sql(sql, con = engine, index_col=None, coerce_float=True, params=None, parse_dates=None, columns=None, chunksize=None)


print(plot_table)


import plotly.graph_objects as go


import plotly.express as px

plot2= plot_table.groupby(['date', 'Category']).agg({'4. close': ['mean']})
plot2.columns = ['mean']
plot2=plot2.reset_index()


print(plot2)

#import pandas for dataframes
import pandas as pd

#make data frame from csv file
df = pd.read_csv("/Users/sarahlin/Downloads/Farm_Stats-Demographics.csv", header=0, squeeze = True, encoding = "ISO-8859-1")

#print out the table
pd.read_csv("/Users/sarahlin/Downloads/Farm_Stats-Demographics.csv", header=0, squeeze = True, encoding = "ISO-8859-1")

#prints the csv file when all run together
print(df)

#Using slicing to grab statistics
print("The percentage of females in the farm industry in various roles is: ", df.loc[0:0])
print("The age ranges of farm workers in various roles: ", df.loc[1:3])
print("The percentage of farm workers born in the U.S. and Puerto Rico is: ", df.loc[10:10])
print("The number of U.S. citizens in each role is: ", df.loc[11:11])
print("The percentage of schooling attained by U.S. farmworkers is: ", df.loc[12:14])

#Summary statistics for numerical columns
df.describe()

#Iterating through each row and column to show percentage for each item
for i, j in df.iterrows():
    print(i, j)
    print()

#Count the number of times a value shows up
df["Farm laborers, graders and sorters"].value_counts()

#Bar graph of csv data
df.plot(kind='bar', stacked=True, title="Demographic characteristics of hired farmworkers and all wage and salary workers, 2017")
df.plot(kind='bar', stacked=False, title="Demographic characteristics of hired farmworkers and all wage and salary workers, 2017")

xls = pd.ExcelFile('/Users/sarahlin/Downloads/j.xlsx')   

df1 = pd.read_excel(xls, 'Farms Operated by Women')
df2 = pd.read_excel(xls, 'Farms operated by men vs Women')



fig2 = px.line(plot2, x ='date', y='mean', color='Category')


fig = px.line(plot_table, x="date", y="4. close", color='ID')


df2.iloc[:1, 1:3].T.plot.pie(subplots=True)

plt.title('Annual number of farmer-days', color='black')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))


df2.iloc[1:2, 1:3].plot.bar()

plt.title('Years of experience for PO on any farm', color='black')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

df2.iloc[2:3, 1:3].plot.bar()

plt.title('Beginning principal operator', color='black')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

df2.iloc[3:4, 1:3].plot.bar()

plt.title('Value of all assets', color='black')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

df2.iloc[4:5, 1:3].plot.bar(legend=False)

plt.title('Acres operated!', color='black')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))


df2.iloc[4:5, 1:3].plot.bar(legend=False)

plt.title('Acres operated!', color='black')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

df2.iloc[4:5, 1:3].plot.bar(legend=False)

plt.title('Acres operated!', color='black')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

df2.iloc[5:6, 1:3].plot.bar(legend=False)

plt.title('Farm sales', color='black')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

df2.iloc[6:7, 1:3].plot.barh()

plt.title('Net farm income', color='black')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

df2.iloc[7:8, 1:3].plot.bar()

plt.title('Percent of Farms with Community Supported', color='black')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

df2.iloc[8:9, 1:3].plot.bar()

plt.title('Percent of Farms with Direct-to-Consumer Sales', color='black')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

df2.iloc[9:10, 1:3].plot.bar()

plt.title('Percent of Farms with Organic Sales', color='black')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

df2.iloc[10:11, 1:3].plot.barh()

plt.title('Observations', color='black')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

df2.iloc[11:12, 1:3].plot.bar()

plt.title('Population represented', color='black')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))


df1.iloc[:1, 1:5].T.plot.bar(legend=False)

plt.title('Woman principal operator', color='black')

df1.iloc[1:2, 1:5].T.plot.bar(legend=False)

plt.title('Annual number of farmer-days', color='black')

df1.iloc[1:2, 1:5].T.plot.bar(legend=False)

df1['All Farms in Restricte d Sample'].plot.hist()

df1.iloc[1:2, 1:5].T.plot.density()

# Uploading the dataset
missing_values = ["n/a", "na", "--", '*', "**", '#'] #Setting characters for NAs
datasin = pd.read_excel('/Users/sarahlin/Downloads/s.xlsx', na_values = missing_values)

# Data preproccesing
print("Column headings:")
print(datasin.columns)
datasin.drop(['NAICS', 'OCC_CODE', 'EMP_PRSE', 'PCT_RPT', 'H_MEAN', 'PCT_TOTAL', 'MEAN_PRSE', 'H_PCT10', 'H_PCT25', 'H_PCT75', 'H_MEDIAN', 'H_PCT90', 'A_PCT10', 'A_PCT25','A_PCT75','A_PCT90','ANNUAL', 'HOURLY'], axis=1, inplace=True)

pd.options.display.float_format = '{:,.0f}'.format #format of floats in data frames

pd.set_option('display.max_rows', None) #Display option

plt.rcParams['figure.figsize'] = [10, 5] #Plotting option for size

datas=datasin[datasin.OCC_GROUP == 'detailed'] #Selecting rows with necessary observations

datas.describe()

dfs=datas[:] #Copy of the dataset for safety purposes

dffin=datasin[datasin.OCC_TITLE == 'Industry Total']


dffin['A_MEAN'].sort_values(ascending=False).plot.bar().axvline(x=104, color='r', linestyle='--', lw=2)
plt.title('Position of Daity Industry, mean wages', color='black')



dffin['TOT_EMP'].sort_values(ascending=False).plot.bar().axvline(x=94, color='r', linestyle='--', lw=2)
plt.title('Position of Daity Industry, total employess', color='black')

dfs[['OCC_TITLE','A_MEAN']].groupby(['OCC_TITLE'], as_index=False).mean().sort_values(by='A_MEAN',ascending=False)[:5]

dfs['A_MEAN'].plot.hist(bins=50)

dfs.isnull().sum() #Checking the missing values

dfs.groupby(['NAICS_TITLE'], as_index=False).mean().sort_values(by='A_MEAN',ascending=False)[:10].plot.bar(x="NAICS_TITLE", y="A_MEAN")

# Filtering the dairy industry
dfd=dfs.loc[dfs['NAICS_TITLE'] == "Dairy Product Manufacturing"]

dfd.sum()

dfd[['OCC_TITLE', 'TOT_EMP', 'A_MEAN']].sort_values(by='A_MEAN',ascending=False)[:10]

dfd[['OCC_TITLE', 'TOT_EMP', 'A_MEAN']].sort_values(by='A_MEAN',ascending=True)[:10]

# Copy of the set
dfd=dfd[:]

# Creating a new column named capacity which combines mean wage and number of employees
dfd['Capacity'] = dfd['TOT_EMP'] * dfd['A_MEAN']

dfd[['OCC_TITLE', 'TOT_EMP', 'A_MEAN', 'Capacity']].sort_values(by='Capacity',ascending=False)[:10]

dfd[['OCC_TITLE', 'TOT_EMP', 'A_MEAN', 'Capacity']].sort_values(by='Capacity',ascending=True)[:10]

# Splitting on categories total employess
bins3 = [0, 100, 1000, 5000, 10000, 20000, 30000, np.inf]
names3 = ['<100 ppl', '100-1k', '1k-5k', '5k-10k', '10k-20k', '20k-30k', '30k+']
dfd["size"] = pd.cut(dfd['TOT_EMP'], bins3, labels=names3)
dfd.tail()

dfd[['size', 'A_MEAN']].groupby(['size']).count().plot.bar()

# Splitting on categories mean salaries
bins4 = [0, 20000, 30000, 40000, 50000, 60000, 80000, 100000, 140000, 180000, np.inf]
names4 = ['<20K', '20k-30k', '30k-40k', '40k-50k', '50k-60k', '60k-80k', '80k-100k', '100k-140k', '140k-180k', '180k+']
dfd["salary"] = pd.cut(dfd['A_MEAN'], bins4, labels=names4)
dfd.tail()

dfd[['salary', 'A_MEAN']].groupby(['salary']).count().plot.bar(legend=False)
plt.title('Distribution of wages', color='black')

dfd[['salary', 'A_MEAN']].groupby(['salary']).count().plot.pie(subplots=True)
plt.title('Population represented', color='black')
plt.legend(loc='center left', bbox_to_anchor=(1.3, 0.5))

dfd[['salary', 'TOT_EMP']].groupby(['salary']).sum().plot.bar(legend=False)

plt.title('Number of employees in that range', color='black')

dfd.sort_values(by='A_MEAN',ascending=False)

fig = plt.gcf()
ax = fig.add_subplot()
fig.set_size_inches(20, 10)

squarify.plot(label=dfd.OCC_TITLE[:10],sizes=dfd.A_MEAN[:10], alpha=.6)

dfd.boxplot(column=['A_MEAN'], grid=False)

dfd.boxplot(column=['TOT_EMP'], grid=False)

dfd['TOT_EMP'].plot.hist()
plt.title('Employees distribution', color='black')

fd['A_MEAN'].plot.hist()
plt.title('Mean distribution', color='black')

dfd1=datasin[datasin.OCC_GROUP == 'major'] #Selecting rows with necessary observations

dfd2=dfd1.loc[dfd1['NAICS_TITLE'] == "Dairy Product Manufacturing"]

dfd2[['OCC_TITLE', 'TOT_EMP', 'A_MEAN']]

dfd2.boxplot(column=['A_MEAN'], grid=False)

dfd2.boxplot(column=['TOT_EMP'], grid=False)

dfd2=dfd2[:] #Copy of the df

dfd2['OCC_TITLE']=dfd2['OCC_TITLE'].str.strip('Occupations')  #Strip the occupation word

dfd2[['OCC_TITLE', 'TOT_EMP','A_MEAN']].groupby(['OCC_TITLE']).mean().plot.barh(legend=False)
plt.title('Mean wages', color='black')


dfd2[['OCC_TITLE', 'A_MEAN']].groupby(['OCC_TITLE']).mean().sort_values(by='A_MEAN').plot.barh(legend=False)
plt.title('Mean wages', color='black')


dfd2[['OCC_TITLE', 'TOT_EMP']].groupby(['OCC_TITLE']).mean().sort_values(by='TOT_EMP').plot.barh(legend=False)

plt.title('Population represented', color='black')


data = pd.read_csv('/Users/sarahlin/Downloads/usa_6.csv')


data.columns


data.head()


data.drop('SERIAL',axis=1,inplace=True)
data.drop('CBSERIAL',axis=1,inplace=True)
data.drop('SAMPLE',axis=1,inplace=True)

data.head()


data.rename(columns={'HHWT':'HouseholdWeight'},inplace=True)
data.rename(columns={'CLUSTER':'HouseholdCluster'},inplace=True)
data.rename(columns={'FARM':'Farm Status'},inplace=True)
data.rename(columns={'PERNUM':'PersonNumInSample'},inplace=True)
data.rename(columns={'PERWT':'PersonWeight'},inplace=True)
data.rename(columns={'LABFORCE':'LaborForceStatus'},inplace=True)
data.rename(columns={'INCTOT':'TotalPersonalIncome'},inplace=True)
data.rename(columns={'INCWAGE':'WageSalaryIncome'},inplace=True)
data.rename(columns={'GQ':'GroupQuartersStatus'},inplace=True)

data.head()

data.head()


data.plot(kind='scatter',x='HouseholdWeight',y='WageSalaryIncome',color='red')
plt.show()


data.plot(kind='scatter',x='PersonWeight',y='WageSalaryIncome',color='red')
plt.show()


data.isnull().values.any()


data.plot(kind='scatter',x='REGION',y='WageSalaryIncome',color='red')
plt.show()







