import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

''' 
Є дані про планові та фактичні продажі за місяцями. Потрібно порівняти два ряди на одному графіку.
    -Створіть список місяців і два списки значень: "План" і "Факт".
    -Побудуйте один графік із двома лініями.
    -Додайте підписи осей і заголовок.
    -Додайте легенду для ліній.
    -Відобразіть графік на екрані.
'''

fact_sales = np.array([200, 125, 366, 580, 250, 135, 300, 520, 740, 600, 475, 200])
plan_sales = np.array([300, 350, 500, 650, 700, 300, 500, 475, 400, 600, 456, 300])
months = np.array(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Avg", "Sep", "Oct", "Nov", "Dec"])

plt.title("Difference between Actual Sales and Planned Sales", fontsize=14)
plt.plot(months, fact_sales, 'r', label="Actual Sales")
plt.plot(months, plan_sales, 'b', label="Planned Sales")
plt.xlabel('Month', fontsize=12)
plt.ylabel('Sales', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()




'''
Дано значення вікових груп 100 осіб. Потрібно показати розподіл.
    -Створіть список (або масив) зі 100 цілих значень віку.
    -Побудуйте гістограму розподілу.
    -Додайте вертикальну лінію середнього значення.
    -Підпишіть осі та додайте заголовок.
    -Відобразіть графік на екрані.
'''

ages = np.random.randint(0, 100, (100, 1))
mean_age = np.mean(ages)

print(ages)

plt.title("Ages distribution")
plt.hist(ages, color='green', label='People', bins=20, edgecolor='lightgreen', linewidth=1)
plt.axvline(mean_age, color='red', linestyle='--', label=f'Mean age value = {mean_age:.1f}')
plt.xlabel("Age")
plt.ylabel("Person")
plt.legend()
plt.show()




'''
Є результати іспитів за трьома групами студентів. Потрібно порівняти розкид оцінок між групами.
    -Створіть три набори числових результатів (по одній вибірці на групу).
    -Побудуйте графік boxplot для трьох груп поруч.
    -Підпишіть групи по осі X.
    -Додайте підписи осей і заголовок.
    -Відобразіть графік на екрані.
'''

groups = np.random.randint(1, 13, (10, 3))
labels = np.array(['First Group', 'Second Group', 'Third Group'])

plt.boxplot(groups, tick_labels=labels)
plt.title("Group Grades Distribution")
plt.xlabel('Groups')
plt.ylabel('Grades')
plt.show()




'''
Є денні значення температури та вологості за тиждень. Потрібно показати обидві метрики на загальній діаграмі.
    -Створіть список дат (7 днів) і два списки значень: температура і вологість.
    -Побудуйте лінійний графік температури.
    -На тій самій області побудуйте графік вологості.
    -Додайте легенду, підписи осей і заголовок.
    -Поверніть підписи дат на осі X для читабельності та відобразіть графік.
'''

days = np.array(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
temperature = np.array([28, 18, 18, 17, 19, 16, 13])
humidity = np.array([60, 39, 43, 45, 45, 62, 54])

plt.title("Week Humidity&Temperature")
plt.plot(days, temperature, color='green', label='Temperature')
plt.plot(days, humidity, color='blue', label='Humidity')
plt.xlabel('Days')
plt.ylabel("Values")
plt.legend()
plt.grid(True)
plt.show()




'''
Є погодинні дані навантаження сервера за добу. Потрібно показати зміни і виділити область між кривою і віссю X.
    -Створіть список годин (0-23) і відповідні значення навантаження.
    -Побудуйте лінійний графік навантаження.
    -Зафарбуйте область під кривою.
    -Додайте сітку, підписи осей і заголовок.
    -Відобразіть графік на екрані.
'''

hours = np.array([hour for hour in range(24)])
server_load = np.array([10, 25, 40, 20, 2, 5, 8, 5, 15, 37, 52, 46, 60, 85, 90, 78, 63, 50, 43, 54, 39, 20, 7, 1])

plt.plot(hours, server_load, color='lightgreen', label='Server Load')
plt.title("Server Load of Day")
plt.xlabel('Hours')
plt.ylabel('Load')
plt.grid(True)
plt.fill_between(hours, server_load, color="green")
plt.show()




'''
Дано чотири набори метрик продукту: конверсія, утримання, середній чек, кількість замовлень. Потрібно показати їх на окремих панелях.
    -Створіть чотири набори числових значень за однією шкалою часу.
    -Створіть сітку з чотирьох підграфіків 2×2.
    -На кожному підграфіку побудуйте відповідний графік.
    -Додайте заголовки для кожного підграфіка та загальний заголовок.
    -Відобразіть результат на екрані.
'''

fig, axs = plt.subplots(2, 2, figsize=(10, 8))
conversion = np.random.randint(0, 101, (24, 1))
retention = np.random.randint(0, 101, (24, 1))
mean_check = np.random.randint(400, 6800, (1, 24))
orders = np.random.randint(0, 500, (1, 24))

axs[0, 0].plot(hours, conversion, color='blue', label='Product Conversion')
axs[0, 0].set_title('Product Conversion Graphic')
axs[0, 0].set_xlabel('Hours')
axs[0, 0].set_ylabel('Conversion')
axs[0, 0].legend()
axs[0, 0].grid(True)

axs[0, 1].bar(hours, np.ravel(orders), color='green', label='Orders Count')
axs[0, 1].set_title('Orders Count Graphic')
axs[0, 1].set_xlabel('Hours')
axs[0, 1].set_ylabel('Orders')
axs[0, 1].legend()
axs[0, 1].grid(True)

axs[1, 0].bar(hours, np.ravel(mean_check), color='purple', label='User Orders')
axs[1, 0].set_title('User Orders Graphic')
axs[1, 0].set_xlabel('Hours')
axs[1, 0].set_ylabel('Orders')
axs[1, 0].legend()
axs[1, 0].grid(True)

axs[1, 1].plot(hours, retention, color='red', label='User Retention')
axs[1, 1].set_title('User Retention Graphic')
axs[1, 1].set_xlabel('Hours')
axs[1, 1].set_ylabel('Retention')
axs[1, 1].legend()
axs[1, 1].grid(True)



fig.tight_layout()
plt.show()












