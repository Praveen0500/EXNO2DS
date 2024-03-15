# EXNO2DS-Exploratory Data Analysis
## AIM:
  To perform Exploratory Data Analysis on the given data set.
      
## EXPLANATION:
 The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
## ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT
### Developed by: PRAVEEN S
### Register no: 212222240078
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  
df=pd.read_csv("titanic_dataset.csv")
df
```
![313066300-cbc6ff39-35dd-49dd-a0cd-117037d75fb4](https://github.com/Praveen0500/EXNO2DS/assets/120218611/5bc423f9-735b-472a-bfc5-d886ff120f29)


```py
df.info()
```
![313066350-a7f4c211-ecce-486b-ae56-4c5da1e62762](https://github.com/Praveen0500/EXNO2DS/assets/120218611/a07536c7-d6aa-4831-91a4-050977d7f981)


```py
df.shape
```
![313066396-f4f935b6-380e-45b6-b4f9-d331575e93df](https://github.com/Praveen0500/EXNO2DS/assets/120218611/02e4cc6c-fa06-46b4-b656-eb199555a24b)


```py
df.set_index("PassengerId",inplace=True)
df.describe()
```
![313066450-47a96dfe-0311-4ac2-9d48-fff094ba076a](https://github.com/Praveen0500/EXNO2DS/assets/120218611/92155e6b-75df-444e-81ae-0ca137deec8b)


```py
df.shape
```
![313066494-c08b7c53-a1e4-4652-9dfc-f66731b060dd](https://github.com/Praveen0500/EXNO2DS/assets/120218611/3264c4bc-45ab-495b-ac9d-4c6ed367464a)


### Categorical data analysis
```py
df.nunique()
```
![313066569-441e26e2-3f93-4718-bf52-05e59103306e](https://github.com/Praveen0500/EXNO2DS/assets/120218611/a95ad944-a0e0-4313-8723-ed982b154632)


```py
df["Survived"].value_counts()
```
![313066624-50624d54-2a45-4a10-9ad5-0c5fa0103dc3](https://github.com/Praveen0500/EXNO2DS/assets/120218611/526aad51-7067-4b6b-a5b3-28c39b002be2)

```py
per=(df["Survived"].value_counts()/df.shape[0]*100).round(2)
per
```
![313066675-dab227b9-b81e-47f7-81dd-1eef9e28af61](https://github.com/Praveen0500/EXNO2DS/assets/120218611/ba3c8630-12e0-4a46-8f39-a6458226ae8f)

```py
sns.countplot(data=df,x="Survived")
```
![313066734-b2bdd5b7-481b-46bb-8546-a2c27d05019b](https://github.com/Praveen0500/EXNO2DS/assets/120218611/cfb97b16-d01d-4c08-8a85-d4298b5d2f44)

```py
df
```
![313066764-88592867-d69d-4a5d-ae86-7d8dfa51dc9f](https://github.com/Praveen0500/EXNO2DS/assets/120218611/c8813d9e-7302-4656-a19c-b8a8f019ec44)

```py
df.Pclass.unique()
```
![313066806-7f4159bd-2c63-4f5d-baa6-1a85eeac2639](https://github.com/Praveen0500/EXNO2DS/assets/120218611/509b8c41-dbed-4d4f-94d1-b76922bde362)


```py
df.rename(columns={'Sex':'Gender'},inplace=True)
df
```
![313066854-0456afe7-0374-4b06-b87e-e912d4c364d4](https://github.com/Praveen0500/EXNO2DS/assets/120218611/9aadf1e0-38c8-45b0-9a41-92e26277c0ce)


### Bivariate Analysis
```py
sns.catplot(x="Gender",col="Survived",kind="count",data=df,height=5,aspect=.7)
```
![313066935-74b64d86-3964-4108-8e02-bbad27507b76](https://github.com/Praveen0500/EXNO2DS/assets/120218611/85e4198d-8fdb-4e8d-a68b-dbaf7eb56d33)


```py
sns.catplot(x="Survived",hue="Gender",data=df,kind="count")
```
![313066977-68f675f1-d3fa-4ea2-8204-ff686d841381](https://github.com/Praveen0500/EXNO2DS/assets/120218611/8fb85ab5-1a3f-4215-9419-9a87269b440b)


```py
df.boxplot(column="Age",by="Survived")
```
![313067018-04bc1139-16f7-4d8b-aadb-f9ccbdf3562b](https://github.com/Praveen0500/EXNO2DS/assets/120218611/40c5e826-01d9-4f9e-84ce-30a48bae3584)


```py
sns.scatterplot(x=df["Age"],y=df["Fare"])
```
![313067066-9a62a6ee-97d5-478d-8b9e-99fb77f710cb](https://github.com/Praveen0500/EXNO2DS/assets/120218611/ebbc80b6-c7e1-4de4-abad-04eaf5080e65)


```py
sns.jointplot(x="Age",y="Fare",data=df)
```
![313067126-f5248986-6bfa-4286-9c63-71b72b6850f0](https://github.com/Praveen0500/EXNO2DS/assets/120218611/1ffde51a-60f2-41e8-a17d-3400779103ba)


### Multivariate Analysis
```py
fig, ax1 = plt.subplots(figsize=(8,5))
plt = sns.boxplot(ax=ax1,x='Pclass',y='Age',hue='Gender',data=df)
```
![313067174-e9befbd3-7dee-445b-970d-df3fc86fe761](https://github.com/Praveen0500/EXNO2DS/assets/120218611/587e80d4-5972-4240-ab5a-f8fdd8dbce60)


```py
sns.catplot(data=df,col="Survived",x="Gender",hue="Pclass",kind="count")
```
![313067222-ad4387d0-fb03-4f24-b484-d6a6a57c5f96](https://github.com/Praveen0500/EXNO2DS/assets/120218611/49eff650-2b88-4c62-ad06-f8bd7dcb6442)


# Co-relation
```py
corr=df.corr()
sns.heatmap(corr,annot=True)
```

![313067256-ae70cf87-e7f3-415c-b727-5e3591ac19da](https://github.com/Praveen0500/EXNO2DS/assets/120218611/5e5ebad1-91d8-4823-b402-db980670a75f)


```py
sns.pairplot(df)
```
![313067283-5e3c59a9-b1a4-46a6-bdb7-9db8f3cd767d](https://github.com/Praveen0500/EXNO2DS/assets/120218611/665d317b-12e2-4518-b140-32eb2b1bc111)


## RESULT
We have performed Exploratory Data Analysis on the given data set successfully.
