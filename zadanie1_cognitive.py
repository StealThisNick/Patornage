import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path

salary_test = Path('salary.csv')
if salary_test.exists():
    try:
        DEBUG = False
        # Reading data with na_values=[' '] to treat empty
        # string as missing value:
        data = pd.read_csv('salary.csv', na_values=[' '])
        filtred_data = data.copy()
        # creating a copy of dataframe and filtered it of missing:
        filtred_data = data[pd.notna(data['salaryBrutto'])]
        # adding x and y for future plot:
        worked_years = filtred_data['workedYears']
        salary_brutto = filtred_data['salaryBrutto']
        # start lineral regression via. salaryBrutto and workedYears:
        model1 = smf.ols(formula='salaryBrutto~workedYears', data=filtred_data)
        res = model1.fit(q=.5)
        pred_value = res.params.get_values()
        if DEBUG:
            print(model1.params)  # getting alfa nad beta
            print(model1.pvalues)  # checking p-values
            print(model1.rsquared)  # checking r^2
            print(model1.summary())  # check summary of data

        # predict salary brutto for each row:
        salary_pred = res.predict(pd.DataFrame(filtred_data['workedYears']))
        # plot actual salary and predicted salary:
        filtred_data.plot(
                        kind='scatter',
                        x='workedYears', y='salaryBrutto',
                        label='actual salary')
        plt.plot(
            worked_years, salary_pred, c='red',
            linewidth=3, label='predicted salary brutto')
        plt.grid(True)
        plt.legend()
        plt.title('Salary over years')
        plt.xlabel('Worked Years')
        plt.ylabel('Salary Brutto')
        plt.show()

        # adding predicted slary to dataframe:
        filtred_data['salary_pred'] = pd.Series(
            salary_pred, index=filtred_data.index)
        # calculate Residual Standard Error:
        x = (
            (filtred_data.loc[:, 'salaryBrutto'] -
                filtred_data.loc[:, 'salary_pred'])**2).tolist()
        RSEd = sum(x)
        RSE = np.sqrt(RSEd/45)
        # mean of slaryBrutto nedded to calculate actual error of etimate:
        salarymean = np.mean(filtred_data.loc[:, 'salaryBrutto'])
        error = RSE/salarymean  # calculating error
        print(f'Predicted salary brutto growth is: {round(pred_value[1],6)}', end=' '),
        print(f'per year, with error of: {round(error,2)}%')
    except:
        print(f'File {salary_test}, dont exist or are in other path')
