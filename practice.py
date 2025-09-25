import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rd

'''
Завдання 1
    Потрібно проаналізувати взаємозв'язок між користувачами, сесіями та виручкою за днями.
    Усе необхідно запрограмувати в Python з використанням pandas, NumPy і Matplotlib.
        -Сформуйте таблицю мінімум на 30 днів із колонками "date", "users", "sessions", "revenue".
        -Розрахуйте кореляційну матрицю для цих метрик.
        -Побудуйте діаграми розсіювання для пар: users-sessions, users-revenue, sessions-revenue.
        -Побудуйте лінійний графік "revenue" за датами.
        -Виведіть матрицю та всі графіки.
'''

product_data = {
    "date": [day+1 for day in range(30)],
    "users": [rd.randint(0, 5000) for i in range(30)],
    "sessions": [rd.randint(1, 25) for x in range(30)],
    "revenue": [rd.randint(500, 10000) for y in range(30)]
}

product_df = pd.DataFrame(product_data)
corr_matrix = product_df.corr()
print(corr_matrix)

date = product_df['date'].values
users = product_df['users'].values
sessions = product_df['sessions'].values
revenue = product_df['revenue'].values

plt.scatter(users, sessions, color="green", label='Correlation')
plt.title('Users and Sessions Correlation')
plt.legend()
plt.grid(True)
plt.show()

