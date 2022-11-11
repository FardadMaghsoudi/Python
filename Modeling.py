# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 14:45:38 2022

@author: fmaghsoudimoud
"""
## Spectral dataset ##

import pandas as pd
import matplotlib.pyplot as plt
df_FTIR= pd.read_excel (r'C:\Users\fmaghsoudimoud\Desktop\Modeling\FTIR&XRF.xlsx', sheet_name="Data")
df_FTIR.Wavelength = df_FTIR.Wavelength.astype(int)
FTIR_Sheet = df_FTIR.groupby('Wavelength').mean()
FTIR_Sheet3 = FTIR_Sheet.reset_index()
FTIR_Sheet2 = df_FTIR.groupby('Wavelength').mean().transpose() #add .reset_index() to separate wavelength
FTIR_Sheet_transposed = FTIR_Sheet.transpose()
#plt.plot(FTIR_Sheet)
FTIR_SPECTRA=FTIR_Sheet.plot(title= 'FTIR spectra of the Profen area')
FTIR_SPECTRA.set_xlabel ("Wavenumber (cm-1)")
FTIR_SPECTRA.set_ylabel ("Reflectance")

FTIR_Spectrum=plt.plot (FTIR_Sheet.index, FTIR_Sheet['1-1'])
FTIR_Spectrum_T= plt.title('An FTIR spectrum of the Profen area')
FTIR_Spectrum_X= plt.xlabel("Wavenumber (cm-1)")
FTIR_Spectrum_Y= plt.ylabel("Reflectance")


#df_FTIR.Wavelength = (10000000/df_FTIR.Wavelength).astype(int)
#FTIR_Sheet = df_FTIR.groupby('Wavelength').mean()

## ICP dataset ##

from sklearn.linear_model import LinearRegression
df_ICP_List = pd.read_excel (r'C:\Users\fmaghsoudimoud\Desktop\Modeling\ICP\Profen-ICP-FMM.xlsx', sheet_name= "Sheet1")
df_ICP = df_ICP_List.set_index ('Sample')
x = df_ICP['Log_Th'].values.reshape(-1, 1)
y = df_ICP['Log_Ti']
model = LinearRegression()
model.fit (x, y)
model = LinearRegression().fit(x,y)
R_Squared = model.score (x,y)
y_prediction = model.predict(x)
plt.scatter (x, y)
plt.plot(x,y_prediction, color='red')
plt.plot()
df_ICP_T= plt.title('ICP data correlations between two elements, samples of the Profen area')
df_ICP_X= plt.xlabel("Log_Th")
df_ICP_Y= plt.ylabel("Log_Ti")  
print ('coefficient of determination:', R_Squared)
print ('intercept:', model.intercept_)
print ('slope:', model.coef_)

## XRF dataset ##

df_XRF = pd.read_excel (r'C:\Users\fmaghsoudimoud\Desktop\Modeling\Profen-XRF-FMM.xlsx', sheet_name= "Variables")
df_XRF_Sheet = df_XRF.iloc [0:139, 0:29]
new_df_XRF= df_XRF_Sheet.set_index('Sample')
new_df_XRF['X(m)'] = new_df_XRF['X(m)'].astype(int)
df_XRF_samples = new_df_XRF.groupby ('Sample').mean()
x_2 = df_XRF_samples['Log_Th'].values.reshape(-1, 1)
y_2 = df_XRF_samples['Log_Ti']
model = LinearRegression()
model.fit (x_2, y_2)
model = LinearRegression().fit(x_2,y_2)
R_Squared = model.score (x_2,y_2)
y_2_prediction = model.predict(x_2)
plt.scatter (x_2, y_2)
plt.plot(x_2,y_2_prediction, color='red')
plt.plot()
df_XRF_samples_T= plt.title('p-XRF data correlations between two elements, samples of the Profen area')
df_XRF_samples_X= plt.xlabel("Log_Th")
df_XRF_samples_Y= plt.ylabel("Log_Ti")  
print ('coefficient of determination:', R_Squared)
print ('intercept:', model.intercept_)
print ('slope:', model.coef_)

## Merging two datasets ##

Merged_database = pd.concat([df_XRF_samples, df_ICP, FTIR_Sheet_transposed], axis=1)
Total_Database = Merged_database.dropna().copy() 
Total_Database.to_excel (r'C:\Users\fmaghsoudimoud\Desktop\Modeling\Output\Total_Database_Profen.xlsx')
Total_Database.to_csv (r'C:\Users\fmaghsoudimoud\Desktop\Modeling\Output\Total_Database_Profen.csv')
merged_XRFvsFTIR = pd.concat([df_XRF_samples, FTIR_Sheet_transposed], axis=1)
merged_XRFvsICP = pd.concat([df_XRF_samples, df_ICP], axis=1)

## Exploratory Data Analysis (EDA) ##
### ICP versus XRF ###

x_3 = merged_XRFvsICP ['Log_Th'].values.reshape(-1, 1)
y_3 = merged_XRFvsICP['Log_Th']
model = LinearRegression()
model.fit (x_3, y_3)
model = LinearRegression().fit(x_3,y_3)
R_Squared = model.score (x_3,y_3)
y_3_prediction = model.predict(x_3)
plt.scatter (x_3, y_3)
plt.plot(x_3,y_3_prediction, color='red')
plt.plot()
df_XRF_samples_T= plt.title('p-XRF data correlations between two elements, samples of the Profen area')
df_XRF_samples_X= plt.xlabel("Log_Ti_XRF")
df_XRF_samples_Y= plt.ylabel("Log_Ti_ICP")  
print ('coefficient of determination:', R_Squared)
print ('intercept:', model.intercept_)
print ('slope:', model.coef_)

### Correlations and their plots ###

correlation = Total_Database.corr()
XRF_Correlation = correlation.iloc[2:26,2:26]
ICP_Correlation = correlation.iloc[26:138,26:138]
FTIR_Correlation = correlation.iloc[138:3488,138:3488]
XRFvsICP_Correlation = correlation.iloc[3:138,3:138]
ICPvsFTIR_Correlation = correlation.iloc[138:3488,2:138]
merged_XRFvsFTIR_Correlation = merged_XRFvsFTIR.corr()
XRFvsFTIR_Correlation = merged_XRFvsFTIR_Correlation.iloc[26:3488,2:26]
XRFvsFTIR_Correlation.plot(y=["Log_As", "Log_Fe","Log_Th", "Log_Ti","Log_Y"])
XRFvsFTIR_Correlation_T= plt.title('Correlation coefficient FTIR spectra of the Profen samples')
XRFvsFTIR_Correlation_X= plt.xlabel("Wavenumber (cm-1)")
XRFvsFTIR_Correlation_Y= plt.ylabel("Correlation coefficient")
XRFvsFTIR_Correlation.plot(y=["Log_As", "Log_Fe"])
XRFvsFTIR_Correlation.plot(y=["Log_Th", "Log_Ti","Log_Y"])

## Principal Component Analysis (PCA) ##
### PCA FTIR spectral plot ###

from sklearn.preprocessing import StandardScaler
FTIR_PCA = StandardScaler().fit_transform(FTIR_Sheet)
from sklearn.decomposition import PCA
pca = PCA(n_components=5, svd_solver='full')
pca.fit(FTIR_PCA)
principalComponents = pca.fit_transform(FTIR_PCA)
PCDF = pd.DataFrame (data = principalComponents ,columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])
PCs = PCDF
print(pca.explained_variance_ratio_)
print(pca.singular_values_)

pcc = pd.concat([FTIR_Sheet3, PCs], axis = 1).set_index('Wavelength').iloc[:, 26:]
pcc2 = pcc.reset_index()
FTIR_PC = plt.plot (pcc2['Wavelength'], pcc2['PC1'])
FTIR_Spectrum_T= plt.title('FTIR principal component coefficient spectrum of the Profen area')
FTIR_Spectrum_X= plt.xlabel("Wavenumber (cm-1)")
FTIR_Spectrum_Y= plt.ylabel("PC 1")

### PCA FTIR spectral dot plot ###

from sklearn.preprocessing import StandardScaler
FTIR_PCA = StandardScaler().fit_transform(FTIR_Sheet)
from sklearn.decomposition import PCA
pca = PCA(n_components=5, svd_solver='full')
pca.fit(FTIR_PCA)
principalComponents = pca.fit_transform(FTIR_PCA)
PCDF = pd.DataFrame (data = principalComponents ,columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])
PCs = PCDF
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
PCs.plot.scatter (x=['PC1'], y=['PC2'])
PCs.plot.scatter (x=['PC1'], y=['PC3'])
PCs.plot.scatter (x=['PC2'], y=['PC3'])
PCs_T= plt.title('Principal compontent analysis')
PCs_T_X= plt.xlabel("PC1")
PCs_T_Y= plt.ylabel("PC2")

### PCA FTIR Transposed dot plot ###

FTIR_Transposed = FTIR_Sheet.T
from sklearn.preprocessing import StandardScaler
FTIR_PCA2 = StandardScaler().fit_transform(FTIR_Transposed)
from sklearn.decomposition import PCA
pca = PCA(n_components=5, svd_solver='full')
pca.fit(FTIR_PCA2)
principalComponents = pca.fit_transform(FTIR_PCA2)
PCDF = pd.DataFrame (data = principalComponents ,columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])
PCs = PCDF
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
PCs.plot.scatter (x=['PC1'], y=['PC2'])
PCs.plot.scatter (x=['PC1'], y=['PC3'])
PCs.plot.scatter (x=['PC2'], y=['PC3'])
PCs_T= plt.title('Principal compontent analysis')
PCs_T_X= plt.xlabel("PC1")
PCs_T_Y= plt.ylabel("PC2")

### PCA ICP dot plot ###

from sklearn.preprocessing import StandardScaler
ICP_Log = df_ICP.iloc[:,56:]
ICP_PCA = StandardScaler().fit_transform(ICP_Log)
from sklearn.decomposition import PCA
pca = PCA(n_components=5, svd_solver='full')
pca.fit(ICP_PCA)
principalComponents_ICP = pca.fit_transform(ICP_PCA)
PCDF_ICP = pd.DataFrame (data = principalComponents_ICP ,columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])
PCs_ICP = PCDF_ICP
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
PCs_ICP.plot.scatter (x=['PC1'], y=['PC2'])
PCs_ICP.plot.scatter (x=['PC1'], y=['PC3'])
PCs_ICP.plot.scatter (x=['PC2'], y=['PC3'])
PCs_ICP_T= plt.title('Principal compontent analysis')
PCs_ICP_T_X= plt.xlabel("PC1")
PCs_ICP_T_Y= plt.ylabel("PC2")

### PCA XRF dot plot ###

from sklearn.preprocessing import StandardScaler
XRF_Log = df_XRF_samples.iloc[:,14:]
XRF_PCA = StandardScaler().fit_transform(XRF_Log)
from sklearn.decomposition import PCA
pca = PCA(n_components=5, svd_solver='full')
pca.fit(XRF_PCA)
principalComponents_XRF = pca.fit_transform(XRF_PCA)
PCDF_XRF = pd.DataFrame (data = principalComponents_XRF ,columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])
PCs_XRF = PCDF_XRF
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
PCs_XRF.plot.scatter (x=['PC1'], y=['PC2'])
PCs_XRF.plot.scatter (x=['PC1'], y=['PC3'])
PCs_XRF.plot.scatter (x=['PC2'], y=['PC3'])
PCs_XRF_T= plt.title('Principal compontent analysis')
PCs_XRF_T_X= plt.xlabel("PC1")
PCs_XRF_T_Y= plt.ylabel("PC2")

## Modeling ##
### Stepwise multiple linear regression ###
#### As prediction ####

import statsmodels.api as sm
x_model = FTIR_Sheet2
y_model = XRF_Log.iloc[:,0]
model = sm.OLS(y_model,x_model).fit()
predictions = model.predict(x_model)
model.summary()
lm = model.LinearRegression ()
model = lm.fit(x_model, y_model)
lm.score (x_model, y_model)
lm.coef_
lm.intercept_

#https://towardsdatascience.com/simple-and-multiple-linear-regression-in-python-c928425168f9 

