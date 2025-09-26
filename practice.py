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




'''
Завдання 2
    Потрібно проаналізувати дані A/B-експерименту та візуалізувати конверсії.
    Усе необхідно запрограмувати в Python з використанням pandas, NumPy і Matplotlib.
        -Сформуйте таблицю з полями "group" (A або B) і "converted" (0/1) з не менш ніж 100 спостереженнями в кожній групі.
        -Розрахуйте конверсію в групах, абсолютну різницю та відносну зміну.
        -Побудуйте 95% довірчі інтервали для конверсії в кожній групі.
        -Побудуйте стовпчасту діаграму конверсій груп із відображенням довірчих інтервалів.
        -Виведіть усі розраховані значення та графік.
'''

experiment_data = {
    "group": ['A' if y < 100 else 'B' for y in range(200)],
    "converted": [rd.randint(0, 1) for i in range(200)]
}

experiment_df = pd.DataFrame(experiment_data)
conversion = experiment_df.groupby('group')['converted'].agg(conversion='mean')
abs_diff = np.abs(conversion.loc['A', 'conversion'] - conversion.loc['B', 'conversion'])
rel_change = (conversion.loc['B', 'conversion'] - conversion.loc['A', 'conversion']) / conversion.loc['A', 'conversion']

t = 2.262

values_a = experiment_df[experiment_df['group'] == 'A']['converted'].to_numpy()
mean_a = np.mean(values_a)
std_a = np.std(values_a, ddof=1)
se_a = std_a / np.sqrt(len(values_a))
a_low = mean_a - t * se_a
a_high = mean_a + t * se_a

values_b = experiment_df[experiment_df['group'] == 'B']['converted'].to_numpy()
mean_b = np.mean(values_b)
std_b = np.std(values_b, ddof=1)
se_b = std_b / np.sqrt(len(values_b))
b_low = mean_b - t * se_b
b_high = mean_b + t * se_b

groups = conversion.reset_index()['group'].tolist()
values = conversion.reset_index()['conversion'].tolist()
errors = [
    [a_high - mean_a, abs(a_low - mean_a)],
    [b_high - mean_b, abs(b_low - mean_b)]
]

print(conversion)
print("Absolute difference:", abs_diff)
print("Relative change:", rel_change)
print(f"Confidence interval A: [{a_low:.2f}, {a_high:.2f}]")
print(f"Confidence interval B: [{b_low:.2f}, {b_high:.2f}]")

plt.bar(groups, values, color='lightgreen', label='Group Conversion', yerr=errors, capsize=5)
plt.title('Experiment Group Conversion')
plt.xlabel('Group')
plt.ylabel('Conversion')
plt.legend()
plt.show()




'''
Завдання 3
    Потрібно перевірити дію центральної граничної теореми на прикладі несиметричного розподілу.
    Усе необхідно запрограмувати в Python з використанням pandas, NumPy і Matplotlib.
        -Згенеруйте генеральну сукупність щонайменше з 50 000 спостережень із несиметричного розподілу.
        -Сформуйте кілька підвибірок фіксованого розміру n і для кожної обчисліть середнє.
        -Збережіть вибіркові середні та побудуйте їхню гістограму.
        -Повторіть процедуру для щонайменше двох різних n і виведіть обидві гістограми.
        -Виведіть середнє і стандартне відхилення вибіркових середніх для кожного n.
'''

asym = {
    'asymmetry': np.random.exponential(scale=5.0, size=50000)
}
asym_df = pd.DataFrame(asym)

def get_sample_means(arr, sample_size, samples_count):
    for _ in range(samples_count):
        sample = asym_df['asymmetry'].sample(sample_size, replace=True)
        arr = np.append(arr, sample.mean())
    return arr

def get_hist(means_arr):
    plt.hist(means_arr, bins=20, color='green', label='Sample Mean', edgecolor='lightgreen', linewidth=1)
    plt.title('Examination Central Limit Theorem')
    plt.xlabel('Mean')
    plt.legend()
    plt.show()

num_samples = 100
means = np.array([])

for _ in range(3):
    n = np.random.randint(200, 500)
    means = get_sample_means(means, n, num_samples)
    mean_val = means.mean()
    std_val = means.std(ddof=1)

    print(f"n={n}: Mean = {mean_val:.3f}, Standard Deviation = {std_val:.3f}")
    get_hist(means)




'''
Завдання 4
    Потрібно проаналізувати часовий ряд продажів і візуалізувати ковзаючі метрики.
    Усе необхідно запрограмувати в Python з використанням pandas, NumPy і Matplotlib.
        -Сформуйте таблицю "date" і "sales" за 90 днів.
        -Додайте ковзне середнє і ковзне стандартне відхилення за обраним вікном.
        -Побудуйте графік вихідних продажів і графік ковзного середнього на одному полі.
        -Побудуйте окремий графік ковзного стандартного відхилення.
        -Виведіть таблицю з першими рядками нових стовпців і обидва графіки.
'''

sales_data = {
    'date': [day+1 for day in range(90)],
    "sales": [rd.randint(0, 200) for i in range(90)]
}

sales_df = pd.DataFrame(sales_data)

days = sales_df['date'].values
sales = sales_df['sales'].values

window = 5
mean_rolling = sales_df['sales'].rolling(window).mean()
std_rolling = sales_df['sales'].rolling(window).std()


print(mean_rolling.head(10))
print(std_rolling.head(10))

plt.plot(days, sales, color='green', label='Sales by Day')
plt.plot(days, mean_rolling, color='red', label='Mean Rolling Sales by Day')
plt.title('Sales by Day Graphic')
plt.xlabel('Days')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(days, std_rolling, color='blue', label='Standard Rolling Deviation Sales by Day')
plt.title('Standard Rolling Deviation Graphic')
plt.xlabel('Days')
plt.ylabel('Devation')
plt.legend()
plt.grid(True)
plt.show()