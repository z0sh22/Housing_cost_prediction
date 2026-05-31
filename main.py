import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib 
import os

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


'''
Пояснения к кускам из скалерна
1. взяла датасет для своей задачи
2. штука чтоб датасет делился на трейн и тест часть 
3. маштабирование
4. самая простая моделька с гиперплоскостью
5. ансамбли моделей деревьев, предскаханеи = усредненный ответ
6. метрики оценки
'''

os.makedirs("outputs", exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")

# 1. Загрузка датасета 
data = fetch_california_housing(as_frame=True)
df = data.frame 

print(f"Размер датасета: {df.shape[0]} на {df.shape[1]}")
print(df.head())

'''
1. as_frame=True означает данные как пандас датафрейм а не нумпай массив
2. data.frame берем именно табличку
3. df.shape возвращаемт размер таблицы в виде (строки, столбцы)
4. df.head() - первые 5 строк таблицы 
'''
# 2. EDA - Exploratory Data Analysis 
col = df.columns.tolist()
print(df.info())

'''
MedInc' - Медианный доход жителей района (в $10 000)
'HouseAge', - Медианный возраст домов в районе (лет)
'AveRooms', - Среднее кол-во комнат на один дом
'AveBedrms',  - Среднее кол-во спален на один дом
'Population', - Население района
'AveOccup', - Среднее кол-во людей в одном доме
'Latitude', - Широта (географическая координата)
'Longitude', - Долгота (географическая координата)
'MedHouseVal' - ЦЕЛЬ: медианная цена дома (в $100 000)

'''

print(df.describe().round(2))
print(df.isnull().sum())
print(df.corr()["MedHouseVal"].sort_values(ascending=False).round(3))

'''
1. мин, макс, среднее и медиана для каждого столбца
2. сколько пропусков
3. корреляция каждого признака с ценой. Число от -1 до 1: чем ближе к 1 — тем сильнее связь
'''

'''
Ну пора сделать некторые выводы по этому поводу.
1. Датасет чист и без пропусков 
2. минимальная цена $15к максимальная $500к 
3. доход людей от 5к до 150к 
Большие разбросы довольно таки
А теперь корреляции
Самая большая у  MedInc (в принципе логично), слабая у комнат и возрасат дома, у остальных считай ее нет
'''

# 3.Графики

fig, axes = plt.subplots(1, 2, figsize = (14, 5))

axes[0].hist(df["MedHouseVal"], bins = 50, color="steelblue", edgecolor="white")
axes[0].set_title("Распределение цен на дома")
axes[0].set_xlabel("Цена (×$100k)")
axes[0].set_ylabel("Количество районов")

axes[1].scatter(df["MedInc"], df["MedHouseVal"], alpha=0.1, color="steelblue")
axes[1].set_title("Доход vs Цена")
axes[1].set_xlabel("Медианный доход (×$10k)")
axes[1].set_ylabel("Цена (×$100k)")

plt.tight_layout()
plt.savefig("outputs/eda.png", dpi=150)
plt.show()

'''
1. создали холс с двумя графиками в одну строку
2. на первый график из столбца цен берем данные и разбиваем на 50 столбиков, заливка стилблу, и белая граница между столбиками
3. на второй график точечный на оси х доход, на оси у цена
plt.tight_layout() - авто-отступы между двумя графиками
'''

# 4. Подготовка данных для модели
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scalar = StandardScaler()
X_train_scaled = scalar.fit_transform(X_train)
X_test_scaled = scalar.transform(X_test)

'''
Ну тут разделяем данные на тест и трейн и так же скалируем
!!! scalar.fit_transform — изучает среднее и std, потом масштабирует
!!! scalar.transform — использует уже изученные параметры с train
Если сделать fit_transform на test — модель "подсмотрит" информацию из тестовых данных.
'''

# 5. Модели и обучение

# Модель 1 Линейная регрессия
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

# Модель 2 Рандомный лес
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

'''
В целом пару слов об этих моделях)
Линейная регрессия ищет гиперплоскость чтоб минимизировать ошибку
между всех точек моего датасета
вывод цены выглядит так
цена_оценочная = w0 + w1*MedInc + w2*HouseAge + ... + w8 * Longitude
Из плюсов очень простая и быстрая, из минусов не видит лин зависимостей (легко может промахнуться в таком случае)

А Рандомный лес можно сказать состоит из деревьев, каждый из которых это игра из 20 вопросов
По результатом которой (можно сказать да нетка):
Доход > 5.0?
  ДА → Возраст дома > 30?
          ДА → предсказываю $350k
          НЕТ → предсказываю $420k
  НЕТ → Широта > 37?
          ДА → предсказываю $180k
          ...
можно увидеть ответ на мой вопрос
у меня таких 100 деревьев предсказывают стоимость, и потом берем среднее 
'''

# 6. Метрики

def evaluate(name, model, X, y_true):
    y_pred = model.predict(X)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred)**0.5
    r2 = r2_score(y_true, y_pred)

    print(f"\n{name}")
    print(f"  MAE:  {mae:.3f}  (≈ ${mae*100:.0f}k средняя ошибка)")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  R²:   {r2:.3f}")
    return y_pred

y_pred_lr = evaluate("LinearRegression", lr, X_test_scaled, y_test)
y_pred_rf = evaluate("RandomForest", rf, X_test, y_test)


'''
Ну рандомный лес выиграл, конечно не все так идеально, но тут самый простой пайплайн который можно придумать, так что
довольно хороший результат)
Пару слов о метриках
mae = 1/n * ∑|y_i - y_i(pred)|
rmse = (1/n * ∑(y_i - y_i(pred))**2)**0.5
R² = 1 - ∑(y_i - y_i(pred))**2 * 1 / ∑(y_i - y_i(sr))**2
y_i(sr) = среднее значение цен датасета

'''

# 7. Графики предсказания против реальности

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, y_pred, title in zip(
    axes,
    [y_pred_lr, y_pred_rf],
    ["LinearRegression", "RandomForest"]
):
    ax.scatter(y_test, y_pred, alpha=0.1, color="steelblue")
    ax.plot([0, 5], [0, 5], color="red", linewidth=2)  # идеальная линия
    ax.set_xlabel("Реальная цена")
    ax.set_ylabel("Предсказанная цена")
    ax.set_title(title)

plt.tight_layout()
plt.savefig("outputs/predictions.png", dpi=150)
plt.show()


joblib.dump(rf, "outputs/model.joblib")
joblib.dump(scalar, "outputs/scaler.joblib")

print("Модель сохранена в outputs/model.joblib")
print("Скейлер сохранён в outputs/scaler.joblib")