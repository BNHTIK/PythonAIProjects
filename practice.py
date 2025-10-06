import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns

# Генерація синтетичних даних для 200 ресторанів
np.random.seed(42) # Для відтворюваності
num_samples = 200

# Генеруємо випадкові ознаки для ресторанів
price = np.random.randint(1, 5, num_samples)
food_quality = np.random.randint(1, 5, num_samples)
service = np.random.randint(1, 5, num_samples)
location = np.random.randint(1, 5, num_samples)

# Обраховуємо середній рейтинг та генеруємо цільову змінну (1 - хороший ресторан, 0 -поганий)
average_rating = (price + food_quality + service + location) / 4
target = (average_rating > 3.5).astype(int) # Хороший ресторан, якщо середній рейтинг вишче за 3.5


# Розділяємо дані на тренувальні та тестові набори (80% тренувальні, 20% тестові)
X_train, X_test, y_train, y_test = train_test_split(
    np.column_stack((price, food_quality, service, location)),
    target,
    test_size=0.2,
    random_state=42
)

rf_model = RandomForestClassifier(
    n_estimators=100, # кількість дерев
    random_state=42, # для відтворюваності
    bootstrap=True, # використання бутстрапу для вибірок
    max_features='sqrt'
)


# Навчання моделі на тренувальних даних
rf_model.fit(X_train, y_train)

# Робимо передбачення на тестових даних
y_pred = rf_model.predict(X_test)

# Оцінка точності моделі
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")

# Візуалізація
features_names = ['Price', 'Food Quality', 'Service', 'Location']
importances = rf_model.feature_importances_

# Створюємо бар-чарт для важливості ознак
# plt.figure(figsize=(10, 6))
# plt.barh(features_names, importances, color='skyblue') # горизонтальний бар-чарт
# plt.title('Feature Importances in Random Forest Classifier')
# plt.xlabel('Importance')
# for i, v in enumerate(importances):
#     plt.text(v + 0.002, i, f"{v:.2f}", color='blue', va='center') # Додаємо текст до важливості
# plt.show()


# new_restaurant = np.array([[4, 3, 3, 5]])
# prediction = rf_model.predict(new_restaurant)
#
# print(
#     f"New restaurant with grades {new_restaurant[0]} "
#     f"predicted as {'good' if prediction[0] == 1 else 'bad'}."
# )


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, # дані
    annot=True, # інверсія
    fmt='d', # формат
    cmap='Blues',
    xticklabels=['Bad', 'Good'],
    yticklabels=['Good', 'Bad']
)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('CF Matrix')
plt.tight_layout()
plt.show()