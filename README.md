# Housing Cost Prediction

Легкий ML-проект: предсказание цен на жильё в Калифорнии с помощью линейной регрессии и случайного леса.

## Описание

Проект демонстрирует полный пайплайн машинного обучения:

```
Данные → EDA → Предобработка → Обучение → Метрики → Визуализация → Сохранение модели
```

**Датасет:** California Housing (20 640 районов, 8 признаков)  
**Задача:** регрессия — предсказание медианной стоимости дома (`MedHouseVal`)

##  Результаты

| Модель | MAE | RMSE | R² |
|---|---|---|---|
| LinearRegression | 0.533 (~$53k) | 0.746 | 0.576 |
| **RandomForest** | **0.328 (~$33k)** | **0.505** | **0.805** |

RandomForest объясняет **80% разброса цен** и ошибается в среднем на $33k.

## Технологии

- **Python 3.10+**
- **scikit-learn** — данные, модели и метрики
- **matplotlib** — визуализация

## Запуск

```bash
# 1. Клонировать репозиторий
git clone https://github.com/z0sh22/Housing_cost_prediction.git
cd Housing_cost_prediction

# 2. Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

# 3. Установить зависимости
pip install -r requirements.txt

# 4. Запустить
python main.py
```

##  Структура проекта

```
Housing_cost_prediction/
├── main.py              # основной пайплайн
├── requirements.txt     # зависимости
├── README.md            # документация
└── outputs/
    ├── eda_distributions.png     # распределения признаков
    ├── eda_correlation.png       # корреляционная матрица
    ├── predictions_plot.png      # предсказания vs реальность
    ├── feature_importance.png    # важность признаков
    ├── model.joblib              # сохранённая модель
    └── scaler.joblib             # сохранённый скейлер
```

##  Что внутри пайплайна
В коде так же присутствуют комментарии с анализом пайплайна и его составляющих. Такая структура больше конечно подходит для Jupyter-ноутбука, но мне было интересно поработать именно с цельным .py файлом. 

1. **Загрузка данных** — `fetch_california_housing()` из sklearn
2. **EDA** — распределения, корреляции, статистики
3. **Разделение** — 80% train / 20% test
4. **Масштабирование** — `StandardScaler` (fit только на train)
5. **Обучение** — `LinearRegression` и `RandomForestRegressor`
6. **Метрики** — MAE, RMSE, R²
7. **Визуализация** — графики предсказаний и важности признаков
8. **Сохранение** — `joblib.dump()` для модели и скейлера

##  Графики

| График | Описание |
|---|---|
| `eda_distributions.png` | Гистограммы всех признаков |
| `eda_correlation.png` | Тепловая карта корреляций |
| `predictions_plot.png` | Предсказанные vs реальные цены |
| `feature_importance.png` | Топ признаков по важности в RandomForest |

##  Ключевой признак

По результатам EDA и feature importance самый важный признак — **MedInc** (медианный доход района, корреляция 0.69 с целевой переменной).


***