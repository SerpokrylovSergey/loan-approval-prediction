# Loan Approval Prediction Service (CatBoost + FastAPI)

Пет-проект по кредитному скорингу: предсказание одобрения кредита на основе финансовых показателей клиента и его кредитного рейтинга (`cibil_score`).

## Результаты модели
* **Алгоритм:** CatBoostClassifier
* **ROC-AUC на тестовой выборке:** 0.9988
* **Accuracy:** 0.98
* **Главные признаки (Feature Importance):** `cibil_score`, `loan_term`, `loan_amount`, `income_annum`.

## Как запустить API
1. Установить зависимости:
   ```bash
   pip install -r requirements.txt