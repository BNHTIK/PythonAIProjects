import pandas as pd

'''
У вас є дані про книги з полями: "Назва", "Автор", "Рік видання", "Ціна".
    Створіть DataFrame з не менш ніж шістьма рядками.
    Виведіть увесь DataFrame.
    Визначте середню ціну книг.
    Знайдіть книги, видані після 2015 року.
    Відсортуйте DataFrame за ціною в порядку зростання.
'''

books = {
    'Name': ['Grokking Deep Learning', 'The Master and Margarita', '1984', 'Crime and Punishment', 'Three Comrades', 'Harry Potter and the Philosopher’s Stone'],
    'Author': ['Andrew W. Trask', 'Mikhail Bulgakov', 'George Orwell', 'Fyodor Dostoevsky', 'Erich Maria Remarque', 'J.K. Rowling'],
    'Release Date': [2019, 1967, 1949, 1866, 1937, 1997],
    'Price': [450, 350, 280, 400, 320, 450]
}

df = pd.DataFrame(books)
price_avrg = df['Price'].mean()
latest_books = df[df['Release Date'] > 2015]
sorted_df = df.sort_values('Price')

print(f"Data Frame:\n {df}")
print(f"Average book price: {price_avrg}")
print(f"Books which release date higher than 2015:\n {latest_books}")
print(f"Data Frame ordered by price:\n {sorted_df}")


'''
У CSV-файлі містяться дані про замовлення: "Номер замовлення", "Клієнт", "Дата", "Сума".
    Прочитайте дані з файлу в DataFrame.
    Виведіть перші десять рядків.
    Визначте кількість замовлень кожного клієнта.
    Знайдіть максимальну та мінімальну суми замовлень.
    Підрахуйте загальну суму всіх замовлень.
'''

df_loaded = pd.read_csv('orders.csv', encoding='utf-8')
orders_count = df_loaded.groupby('client').count()['order']
min_total = df_loaded['total'].min()
max_total = df_loaded['total'].max()
orders_sum = df_loaded['total'].sum()

print(f"First 10 rows:\n {df_loaded.head(10)}")
print(f"Client orders count:\n {orders_count}")
print(f"Min order total sum: {min_total}")
print(f"Max order total sum: {max_total}")
print(f"Total orders sum: {orders_sum}")




'''
Є таблиця з даними про продукти харчування: "Продукт", "Категорія", "Калорії", "Білки".
    Створіть DataFrame з не менш ніж десятьма рядками.
    Виведіть увесь DataFrame.
    Знайдіть усі продукти з калорійністю вище 300.
    Підрахуйте середню кількість білків за категоріями.
    Відсортуйте DataFrame за калорійністю в порядку убування.
'''

products = {
    'product' : ['tomato', 'icecream', 'pudding', 'dumplings', 'hamburger', 'pie', 'cake', 'cucumber', 'chocolate', 'pasta'],
    'category': ['vegetable', 'sweet', 'sweet', 'dough dishes', 'fastfood', 'bakery', 'sweet', 'vegetable', 'sweet', 'dough dishes'],
    'calories': [30, 250, 120, 700, 1000, 3170, 4500, 26, 546, 262],
    'protein': [1.3, 4.0, 4.7, 30.5, 50.75, 25.5, 60.3, 0.975, 9.0, 10.0]
}

df = pd.DataFrame(products)
calories_df = df[df['calories'] > 300]
protein_avrg = df.groupby('category')['protein'].mean()
sorted_df = df.sort_values('calories', ascending=False)

print(f"Product Data Fram:\n {df}")
print(f"Products where calories more than 300:\n {calories_df}")
print(f"Average protein count group by category:\n {protein_avrg}")
print(f"Descending order by calories:\n {sorted_df}")




'''
Зібрано дані про співробітників та їхні проекти: "Ім'я", "Проект", "Години".
    Створіть DataFrame з не менш ніж вісьмома рядками.
    Виведіть вихідний DataFrame.
    Підрахуйте загальну кількість годин за кожним співробітником.
    Підрахуйте загальну кількість годин за кожним проектом.
    Знайдіть співробітника, який витратив найбільше годин.
'''

workers = {
    'name': ['Alex', 'Daniil', 'Artem', 'Vyacheslav', 'Daniil', 'Artem', 'Alex', 'Victor', 'Victor', 'Daniil'],
    'project': ['Horror Game', 'Shooter Game', 'Shooter Game', 'Sandbox Game', 'Survival Game', 'Web App', 'Web App', 'Shooter Game', 'Horror Game', 'Sandbox Game'],
    'hours': [8, 12, 5, 3, 8, 10, 12, 5, 10, 18]
}

df = pd.DataFrame(workers)

employee_hours = df.groupby('name')['hours'].sum()
project_hours = df.groupby('project')['hours'].sum()
best_employee = employee_hours[employee_hours == employee_hours.max()]

print(f"----Employees data----\n {df}")
print(f"----Employee work hours----\n {employee_hours}")
print(f"----Project work hours----\n {project_hours}")
print(f"----The Best Employee----\n {best_employee}")




'''
У вас є таблиця продажів квитків: "Фільм", "Місто", "Продані квитки".
    Створіть DataFrame з не менш ніж дванадцятьма рядками.
    Виведіть увесь DataFrame.
    Підрахуйте загальну кількість проданих квитків за кожним фільмом.
    Підрахуйте загальну кількість квитків по кожному місту.
    Знайдіть фільм із найбільшою кількістю продажів.
'''

films = {
    'movie' : [
        "Titanic",
        "Inception",
        "Gladiator",
        "Interstellar",
        "Avatar",
        "Titanic",
        "Whiplash",
        "Jaws",
        "Inception",
        "Interstellar",
        "Goodfellas",
        "Whiplash",
        "Gladiator",
        "Interstellar",
    ],
    'city': [
        "Mykolaiv",
        "Kyiv",
        'Odessa',
        "Mykolaiv",
        "Kharkiv",
        "Lviv",
        "Odessa",
        "Lviv",
        "Lviv",
        "Mykolaiv",
        "Kharkiv",
        "Kyiv",
        "Odessa",
        "Kharkiv"
    ],
    'tickets sold': [80, 45, 27, 87, 56, 45, 60, 110, 65, 81, 20, 56, 99, 15]
}

df = pd.DataFrame(films)
movie_tickets = df.groupby('movie')['tickets sold'].sum()
city_tickets = df.groupby('city')['tickets sold'].sum()
best_movie = movie_tickets[movie_tickets == movie_tickets.max()]

print(f"----Movies DataFrame Info----\n{df}")
print(f"----Movie Total Tickets Sold----\n{movie_tickets}")
print(f"----City Total Tickets Sold----\n{city_tickets}")
print(f"----The Best Movie Sold----\n{best_movie}")




'''
Дано дві таблиці: перша з інформацією про клієнтів ("ID", "Ім'я", "Місто"),
друга — з їхніми замовленнями ("ID клієнта", "Замовлення", "Вартість").
    Створіть два DataFrame і заповніть їх не менш ніж п'ятьма рядками кожен.
    Виведіть обидва DataFrame.
    Об'єднайте їх за ідентифікатором клієнта.
    Підрахуйте загальну вартість замовлень за кожним клієнтом.
    Виведіть результат у вигляді нового DataFrame.
'''

clients = {
    'ID': [378, 222, 478, 138, 456],
    'name': ['Daniil', 'Artem', 'Vyacheslav', 'Victor', 'Rumble'],
    'city': ['Mykolaiv', 'Karlsruhe', 'Kyiv', 'Piltover', 'Bundle City']
}
orders = {
    'client_id': [478, 456, 222, 378, 138, 478, 222, 456, 478],
    'order': ["Pizza", "Burger", "Pasta", "Steak", "Sushi", "Pasta", "Burger", "Sushi", "Steak"],
    'total': [379.99, 146.00, 199.99, 479.99, 279.00, 199.99, 146.00, 279.00, 479.99]
}

df1 = pd.DataFrame(clients)
df2 = pd.DataFrame(orders)
merged_df = df1.merge(df2, left_on='ID', right_on="client_id")
client_total = merged_df.groupby(['ID', 'name'])['total'].sum().reset_index()

print(f"----First Data Frame----\n{df1}")
print(f"----Second Data Frame----\n{df2}")
print(f"----Merged Data Frame----\n{merged_df}")
print(f"----Client Total----\n{client_total}")