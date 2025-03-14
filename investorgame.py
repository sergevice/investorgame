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
    "Нерухомість (Фонд нерухомості REITs)": "VNQ",
    "ОВДП": "GC=F",
    "Долар США": "EURUSD=X",
    "Біткоїн": "BTC-USD",
    "Акції Tesla": "TSLA",
    "Акції Apple": "AAPL",
    "Ф'ючерси на каву": "KC=F",
    "Ф'ючерси на какао-боби": "CC=F",
    "Золото": "GC=F",
    "Нерухомість (REITs)": "VNQ"
}

# Функція для завантаження даних
@st.cache_data
def get_stock_data(tickers, retries=3, delay=5):
    for i in range(retries):
        try:
            data = yf.download(list(tickers.values()), period="360d", interval="1d", auto_adjust=True)
            if not data.empty:
                return data["Close"].dropna()
        except Exception as e:
            time.sleep(delay)
    return None

data = get_stock_data(assets)

st.write(f"Довжина отриманих даних: {len(data) if data is not None else 'None'}")

if data is not None and len(data) >= 246:
    data = data.iloc[-246:]  # Гарантуємо, що є рівно 246 днів даних (всі робочі дні за рік)
    data = data.iloc[-360:]  # Гарантуємо, що є рівно 360 днів даних

if data is None or len(data) < 180:
    st.error("Недостатньо даних для аналізу. Спробуйте пізніше.")
else:
    half_period = len(data) // 2  # Половина доступних робочих днів
    df_past = data.iloc[:123]  # Дані від t-246 до t-123
st.write(f"Довжина df_past: {len(df_past)}")
st.write(f"Довжина df_past: {len(df_past)}")  # Динаміка за період t-360d - t-180d  # Перші 180 днів
df_future = data.iloc[123:]  # Дані від t-123 до t
st.write(f"Довжина df_future: {len(df_future)}")
st.write(f"Довжина df_future: {len(df_future)}")  # Динаміка за період t-360d - t  # Останні 180 днів
    
st.title("Інтерактивна інвестиційна гра")

# Відображення графіків динаміки цін активів
st.subheader("Динаміка активів (від рік тому до 6 місяців тому)")
df_past_display = data.iloc[:180]  # Точне визначення діапазону від t-360d до t-180d
fig, axes = plt.subplots(4, 3, figsize=(15, 10))
axes = axes.flatten()
for i, col in enumerate(df_past_display.columns):
    axes[i].plot(df_past.index, df_past[col], label=col, color='blue')
    axes[i].set_title(f"{col}")
    axes[i].set_xlabel("Дата")
    axes[i].set_ylabel("Ціна ($)")
    axes[i].legend()
plt.tight_layout()
st.pyplot(fig)

st.subheader("Ваш інвестиційний портфель")
st.write("Ви маєте ознайомитися з графіками динаміки цін активів, а потім обрати, як розподілити ваш інвестиційний капітал.")
st.write("Пам'ятайте, що ваш вибір вплине на кінцевий результат!")

investment = {}
total_money = 10000
total_percentage = 0

if st.button("Випадково"):
    remaining = 100
    random_weights = {}
    for i, asset in enumerate(df_past.columns):
        if i == len(df_past.columns) - 1:
            random_weights[asset] = remaining
        else:
            random_part = np.random.randint(0, remaining)
            random_weights[asset] = random_part
            remaining -= random_part
    investment = random_weights
else:
    investment = {}

for asset in df_past.columns:
    investment[asset] = st.slider(f"% вкласти у {asset}", 0, 100, 10)
    total_percentage += investment[asset]

if total_percentage != 100:
    st.warning("Сума всіх відсотків має дорівнювати 100%!")
else:
    if st.button("Інвестувати"):
        returns_past = df_past.pct_change().dropna()
        mean_returns = returns_past.mean()
        cov_matrix = returns_past.cov().to_numpy()
        num_assets = len(returns_past.columns)
        returns_future = (data.iloc[-1] / data.iloc[123]) - 1  # Дохідність між t-123 і t
        
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
        for method, result in portfolio_results.items():
            absolute_return = total_money * (1 + result)
            relative_return = result * 100
            st.write(f"{method}: **{absolute_return:.2f} грн** ({relative_return:.2f}%)")
        
        st.subheader("Графіки динаміки цін активів за весь період (від рік тому до сьогодні)")
        fig, axes = plt.subplots(4, 3, figsize=(15, 10))
        axes = axes.flatten()
        for i, col in enumerate(data.columns):
            axes[i].plot(data.index, data[col], label=col, color='blue')
            axes[i].axvline(data.index[half_period - 1], color='red', linestyle='--', label="Момент інвестування")
            axes[i].set_title(f"{col}")
            axes[i].set_xlabel("Дата")
            axes[i].set_ylabel("Ціна ($)")
            axes[i].legend()
        plt.tight_layout()
        st.pyplot(fig)
        
        st.subheader("Пояснення методу Марковіца (MPT)")
        st.write("Метод Марковіца допомагає розподілити гроші між різними активами так, щоб отримати **найбільший прибуток при мінімальному ризику**. Ідея проста: **не вкладати все в один актив**, а створити **збалансований портфель**.")
        st.write("### Як працює оптимізація?")
        st.write("1. **Очікуваний дохід** кожного активу – середній приріст його ціни за минулий період.")
        st.write("2. **Ризик (волатильність)** – наскільки ціна активу коливалася.")
        st.write("3. **Кореляція активів** – як рух одного активу пов'язаний із рухом іншого.")
        st.write("Метод шукає **найкраще поєднання активів**, щоб:")
        st.write("- Зменшити ризик (інвестиції розподілені між різними активами).")
        st.write("- Отримати якомога більший прибуток.")
        
        st.write("### Що таке коефіцієнт Шарпа?")
        st.write("Він допомагає оцінити, наскільки хороший портфель. Формула:")
        st.write("\( Шарп = rac{(Очікуваний дохід - Безризикова ставка)}{Ризик} \)")
        st.write("- **Очікуваний дохід** – середня прибутковість портфеля.")
        st.write("- **Безризикова ставка** – дохід, який можна отримати без ризику (наприклад, за депозитом).")
        st.write("- **Ризик** – стандартне відхилення прибутковості портфеля.")
        
        st.write("**Чим вищий коефіцієнт Шарпа, тим вигідніший портфель!** 🚀")
        st.write("Метод Марковіца шукає оптимальний портфель, який максимізує коефіцієнт Шарпа.")
        st.write("Коефіцієнт Шарпа визначається як відношення очікуваної дохідності до ризику.")
        st.write("Формула коефіцієнта Шарпа: **(Очікуваний дохід портфеля - Безризикова ставка) / Волатильність портфеля**")
        st.write("Чим вищий коефіцієнт Шарпа, тим вигідніший портфель.")
