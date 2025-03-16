import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time

# Список активів
assets = {
    "ОВДП (Облігації внутрішньої державної позики)": "GC=F",
    "Долар США (Валюта)": "EURUSD=X",
    "Біткоїн (Криптовалюта)": "BTC-USD",
    "Акції Tesla (Виробник електромобілів)": "TSLA",
    "Акції Apple (Технологічна компанія)": "AAPL",
    "Ф'ючерси на каву (Сировина)": "KC=F",
    "Ф'ючерси на какао-боби (Сировина)": "CC=F",
    "Золото (Дорогоцінний метал)": "GC=F",
    "Нерухомість (Фонд нерухомості REITs)": "VNQ"
}

# Завантаження даних
@st.cache_data
def get_stock_data(tickers):
    try:
        data = yf.download(list(tickers.values()), period="360d", interval="1d", auto_adjust=True)
        return data["Close"].dropna() if not data.empty else None
    except Exception as e:
        return None

data = get_stock_data(assets)

st.title("Інтерактивна інвестиційна гра")

if data is None or len(data) < 180:
    st.error("Недостатньо даних для аналізу.")
else:
    df_past = data.iloc[:180]
    df_future = data.iloc[180:]

    # Початкові значення інвестицій
    investment = {asset: 10 for asset in df_past.columns}

    # Функція для встановлення 100% у вибраний актив
    def set_full_investment(selected_asset):
        for key in investment.keys():
            investment[key] = 100 if key == selected_asset else 0

    # Функція для випадкового розподілу інвестицій
    def randomize_investment():
        remaining = 100
        random_weights = {}
        keys = list(investment.keys())
        for i, asset in enumerate(keys):
            if i == len(keys) - 1:
                random_weights[asset] = remaining
            else:
                random_part = np.random.randint(0, remaining + 1)
                random_weights[asset] = random_part
                remaining -= random_part
        return random_weights

    st.subheader("Ваш інвестиційний портфель")
    
    # Кнопка для випадкового розподілу
    if st.button("Випадково розподілити"):
        investment = randomize_investment()

    # Вибір активів за допомогою повзунків
    total_percentage = 0
    cols = st.columns(len(investment))

    for i, (asset, weight) in enumerate(investment.items()):
        with cols[i]:
            investment[asset] = st.slider(f"{asset}", 0, 100, weight, key=f"slider_{i}")
            if st.button("Все сюди", key=f"btn_{i}"):
                set_full_investment(asset)
            total_percentage += investment[asset]

    # Перевірка на правильність розподілу
    if total_percentage != 100:
        st.warning("Сума всіх відсотків має дорівнювати 100%!")
    else:
        st.success("Ваш портфель коректно розподілений.")

        # Симуляція інвестування
        returns_past = df_past.pct_change().dropna()
        mean_returns = returns_past.mean()
        cov_matrix = returns_past.cov().to_numpy()
        num_assets = len(returns_past.columns)
        returns_future = (data.iloc[-1] / data.iloc[180]) - 1  # Дохідність між t-180 і t

        def sharpe_ratio(weights):
            return -((weights @ mean_returns) / np.sqrt(weights @ cov_matrix @ weights))

        def optimize_portfolio():
            init_guess = np.ones(num_assets) / num_assets
            bounds = tuple((0, 1) for _ in range(num_assets))
            constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
            result = minimize(sharpe_ratio, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
            return result.x if result.success else init_guess

        optimal_weights_mpt = optimize_portfolio()

        portfolio_results = {
            "Ваш портфель": sum([(investment[asset] / 100) * returns_future[asset] for asset in returns_past.columns]),
            "Марковіц (MPT)": sum([optimal_weights_mpt[i] * returns_future[returns_past.columns[i]] for i in range(num_assets)])
        }

        st.subheader("Результати")
        total_money = 10000
        for method, result in portfolio_results.items():
            absolute_return = total_money * (1 + result)
            relative_return = result * 100
            st.write(f"{method}: **{absolute_return:.2f} грн** ({relative_return:.2f}%)")

        st.subheader("Графіки динаміки цін активів")
        fig, axes = plt.subplots(4, 3, figsize=(15, 10))
        axes = axes.flatten()
        for i, col in enumerate(data.columns):
            axes[i].plot(data.index, data[col], label=col, color='blue')
            axes[i].axvline(data.index[180], color='red', linestyle='--', label="Момент інвестування")
            axes[i].set_title(f"{col}")
            axes[i].set_xlabel("Дата")
            axes[i].set_ylabel("Ціна ($)")
            axes[i].legend()
        plt.tight_layout()
        st.pyplot(fig)
