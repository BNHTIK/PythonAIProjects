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

date = product_df['date'].values
users = product_df['users'].values
sessions = product_df['sessions'].values
revenue = product_df['revenue'].values

fig, axs = plt.subplots(2, 2, figsize=(9, 7))

axs[0,0].scatter(users, sessions, color="green", label='Correlation')
axs[0,0].set_title('Users and Sessions Correlation')
axs[0,0].legend()
axs[0,0].grid(True)

axs[0,1].scatter(users, revenue, color="red", label='Correlation')
axs[0,1].set_title('Users and Revenue Correlation')
axs[0,1].legend()
axs[0,1].grid(True)

axs[1,0].scatter(revenue, sessions, color="blue", label='Correlation')
axs[1,0].set_title('Revenue and Sessions Correlation')
axs[1,0].legend()
axs[1,0].grid(True)

axs[1,1].plot(date, revenue, color="purple", label='Revenue Count')
axs[1,1].set_title('Revenue Count by Day')
axs[1,1].set_xlabel('Day')
axs[1,1].set_ylabel('Revenue')
axs[1,1].legend(loc='best')
axs[1,1].grid(True)

print(corr_matrix)
plt.show()

