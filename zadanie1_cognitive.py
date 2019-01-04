import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path

CSV_FILE = 'salary.csv'
DEBUG = False

salary_path = Path(CSV_FILE)
# Checking if .csv file exist:
while not salary_path.exists():
    print(f'Path "{salary_path.as_posix()}" not exist')
    new_csv = input("Type path to new csv file, or empty value for exit: ")
    if not new_csv:
        print('Bye!')
        exit(1)
    salary_path = Path(new_csv)

try:

    # Reading data with na_values=[' '] to treat empty
    # string as missing value:
    data = pd.read_csv(salary_path, na_values=[' '])
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
        print(res.params)  # getting alfa nad beta
        print(res.pvalues)  # checking p-values
        print(res.rsquared)  # checking r^2
        print(res.summary())  # check summary of data

    # predict salary brutto for each row:
    salary_pred = res.predict(pd.DataFrame(worked_years))
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
    filtred_data = filtred_data.assign(
        salary_pred=pd.Series(salary_pred).values)
    # calculate Residual Standard Error:
    x = (
        (filtred_data.loc[:, 'salaryBrutto'] -
            filtred_data.loc[:, 'salary_pred'])**2).tolist()
    RSEd = sum(x)
    RSE = np.sqrt(RSEd/45)
    # mean of slaryBrutto nedded to calculate actual error of etimate:
    salarymean = np.mean(filtred_data.loc[:, 'salaryBrutto'])
    error = RSE/salarymean  # calculating error
    print(f'Predicted salary brutto growth is: {round(pred_value[1],2)}', end=' '),
    print(f'per year, with error of: {round(error,2)}%')
except Exception as exc:
    print(exc)
    exit(1)
