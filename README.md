# akurdi_dyp
#facebook
import pandas as pd
import numpy as np
df = pd.read_csv(r"C:\Users\ingaw\OneDrive\Pictures\Documents\dataset_Facebook.csv", sep=";")
df
df.describe()
df.shape 
sub1 = df[['Page total likes','Category','Post Month']].loc[0:15]
merging = pd.concat([sub1,sub2,sub3])
merged_data = sub1.merge(sub2,on='Category')
sort_data = df.sort_values('Page total likes',ascending=False) 
df.transpose()
pivot_table = pd.pivot_table(df,index = ['Type','Category'],values = 'comment')
print(pivot_table)
reshape_arr = np.array([1,2,3,4,5,6,7,8,9,10])
reshape_arr.reshape(5,2)

##graphs
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
print(df.info())
# Visualizing the distribution of age using a histogram
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='age', bins=20, kde=True, color='black')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
 Visualizing the correlation between features using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()
# Visualizing the relationship between age and cholesterol using a scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='age', y='chol', hue='target', palette='coolwarm')
plt.title('Relationship between Age and Cholesterol')
plt.xlabel('Age')
plt.ylabel('Cholesterol')
plt.legend(title='Target')
plt.show()
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='target', palette='Set1')
plt.title('Distribution of Target Variable')
plt.xlabel('Target')
plt.ylabel('Count')
plt.show()
#visualization of pairplot on df dataset
sns.set(style = "dark")
sns.pairplot(df)
plt.show()
sns.heatmap(df)
#boxplot 
df.boxplot(by='age',column=['thal'],grid=False)
#piechart
labels = 'fbs','slope','ca','thal'
sizes = [df['fbs'].mean(),df['slope'].mean(),df['ca'].mean(),df['thal'].mean()]
colors = ['Red','Blue','Green','Yellow']
explode = (0.1,0,0,0)
plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%')
plt.title('Average Data')
#bar plot
import numpy as np 
h = df.iloc[1:10,3]
y_pos = np.arange(len(h))
v = range(1,10)
plt.bar(y_pos,h,align='center',alpha=0.5)
plt.xticks(y_pos,v)
plt.ylabel('range')
plt.xlabel('x label')
plt.title('bar plot')
plt.show()
l = df.iloc[:16,3]
m = df.iloc[:16,7]
plt.plot(l,label='age',marker='*',linestyle='dotted')
plt.plot(m,label='cp',marker='o',linestyle='dashed')
plt.ylim(0,1)
plt.legend()
plt.title("Line plot")
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('plotline.png')


##heart opr
import pandas as pd
df.isnull().sum()
#data cleaning
cleaned_data = df.drop_duplicates()
cleaned_data
#data integration 
subset1 = df[['age','chol']]
subset2 = df[['chol','thal']]
integrated_data = subset1.merge(subset2,on='chol')
#data transformation
sex_mapping = {1:'Male',0:'Female'}
df['sex'] = df['sex'].map(sex_mapping)
#one hot encoding data transformation
onehot_encoded = pd.get_dummies(df,columns=['sex'])
onehot_encoded = onehot_encoded.astype(int)
onehot_encoded
#data model Building
from sklearn.linear_model import LinearRegression
x = df[['age']]
y = df[['trestbps']]
model = LinearRegression().fit(x,y)
r_square = model.score(x,y)
r_square
model.intercept_
model.coef_
import matplotlib.pyplot as plt
plt.scatter(x,y,color='blue')
plt.plot(x,model.predict(x),color='red',label='Linear Regression Model')
plt.legend()


##web scrap
import requests 
import bs4
vr1 = requests.get('https://www.snapdeal.com/product/portronics-toad-25-wireless-mouse/6917529703062864514')
vr1
vr1.content
vr2 = bs4.BeautifulSoup(vr1.text)
o_rating = vr2.find('span',{'class':'avrg-rating'}).get_text()
o_rating
F_price = vr2.find('span',{'class':'payBlkBig'}).get_text()
F_price
Detail = vr2.find('div',{'class':'detailssubbox'}).get_text()
Detail

model.intercept_
