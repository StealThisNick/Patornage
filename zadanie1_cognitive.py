import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf

data = pd.read_csv('../Python/zadanie1_cognitive/salary.csv', na_values=[' '])  # Reading data with na_values=[' '] to treat empty string as missing vale
filtred_data = data.copy()
filtred_data = data[pd.notna(data['salaryBrutto'])]  # creating a copy of dataframe and filtered it of missing value
worked_years = filtred_data['workedYears'] # adding x and y for future plot
salary_brutto = filtred_data['salaryBrutto']
model1 = smf.ols(formula='salaryBrutto~workedYears', data=filtred_data).fit()  # start lineral regression via. salaryBrutto and workedYears
# print (model1.params) # getting alfa nad beta
# print (model1.pvalues) # checking p-values
# print (model1.rsquared) # checking r^2
# print (model1.summary()) # check summary of data
salary_pred = model1.predict(pd.DataFrame(filtred_data['workedYears']))  # predict salary brutto for each row
filtred_data.plot(kind='scatter', x='workedYears', y='salaryBrutto', label='actual salary')  # plot actual salary and predicted salary
plt.plot(pd.DataFrame(filtred_data['workedYears']), salary_pred, c='red', linewidth=3, label='predicted salary brutto')
plt.grid(True)
plt.legend()
plt.title('Salary over years')
plt.xlabel('Worked Years')
plt.ylabel('Salary Brutto')
plt.show()


filtred_data['salary_pred'] = pd.Series(salary_pred, index=filtred_data.index)  # adding predicted slary to dataframe
x = ((filtred_data.loc[:, 'salaryBrutto']-filtred_data.loc[:, 'salary_pred'])**2).tolist()  # calculate Residual Standard Error
RSEd = sum(x)
RSE = np.sqrt(RSEd/45)
salarymean = np.mean(filtred_data.loc[:, 'salaryBrutto'])  # mean of slaryBrutto nedded to calculate actual error of etimate
error = RSE/salarymean  # calculating error
print(f'Predicted salary brutto growth is 695.854694 per year, with error of: {round(error,2)}%')