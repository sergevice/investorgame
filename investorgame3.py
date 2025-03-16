import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import scipy.optimize as sco
import numpy as np

def get_asset_tickers():
    return {
        "Apple Inc. (AAPL)": "AAPL",
        "Tesla Inc. (TSLA)": "TSLA",
        "USD/UAH (Курс долара до гривні)": "USDUAH=X",
        "Gold (Золото)": "GC=F",
        "Cocoa Futures (Ф'ючерси на какао)": "CC=F",
        "Coffee Futures (Ф'ючерси на каву)": "KC=F",
        "Bitcoin to USD (Біткоїн за долари)": "BTC-USD",
        "Ethereum to USD (Ефір за долари)": "ETH-USD",
        "S&P 500 Index (SP500)": "^GSPC",
        "Real Estate Investment Trust (VNQ)": "VNQ"
    }

def get_stock_data(tickers, retries=3, delay=5):
    for i in range(retries):
        try:
            data = yf.download(list(tickers.values()), period="360d", interval="1d", auto_adjust=True)
            if not data.empty:
                return data["Close"].dropna()
        except Exception as e:
            time.sleep(delay)
    return None

def plot_price_dynamics(df, show):
    st.subheader("Динаміка цін активів")
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    date_index = df.index
    midpoint = len(date_index) // 2  # Halfway point to plot only first half of data
    
    for i, asset in enumerate(df.columns):
        if i >= 12:  # Limit to 12 charts
            break
        
        axes[i].plot(date_index, df[asset], color='orange', alpha=show)  # Full range for context
        axes[i].plot(date_index[:midpoint], df[asset].iloc[:midpoint], color='blue')  # First half of data
        axes[i].set_title(asset)
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig)

def calculate_returns(df):
    midpoint = len(df) // 2
    returns = (df.iloc[-1] - df.iloc[midpoint]) / df.iloc[midpoint]
    return returns.to_frame(name="Доходність")

def calculate_markowitz_portfolio_old(df_prices):
    """
    Обчислює оптимальний портфель за Марковіцем.
    
    Параметри:
        df_prices (pd.DataFrame): DataFrame, де колонки - активи, рядки - історичні ціни.
    
    Повертає:
        pd.DataFrame: df_investments з оптимальним розподілом портфеля.
    """
    returns = df_prices.pct_change().dropna()  # Обчислюємо щоденні дохідності
    mean_returns = returns.mean()  # Середня дохідність активів
    cov_matrix = returns.cov()  # Матриця коваріацій
    num_assets = len(df_prices.columns)
    
    # Функція для мінімізації ризику (волатильності портфеля)
    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # Обмеження: сума ваг активів = 1
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    
    # Межі ваг активів (0-100%)
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    # Початкові ваги (рівномірно розподілені)
    initial_weights = np.array(num_assets * [1.0 / num_assets])
    
    # Оптимізація для знаходження мінімальної волатильності
    result = sco.minimize(portfolio_volatility, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if not result.success:
        raise ValueError("Оптимізація не зійшлася. Спробуйте змінити вхідні дані.")
    
    optimal_weights = result.x
    
    # Формуємо df_investments
    df_investments = pd.DataFrame({
        "Тікер": df_prices.columns,
        "% вкладення": optimal_weights * 100
    })
    
    return df_investments

def calculate_markowitz_portfolio(historic_asset_prices):
    """
    Обчислює оптимальний портфель за методом Марковіца, максимізуючи Sharpe Ratio.
    
    Параметри:
        historic_asset_prices (pd.DataFrame): Датафрейм, де рядки - дні, колонки - тікери активів.
    
    Повертає:
        pd.DataFrame: Датафрейм із колонками "Тікер" та "% вкладення" (оптимальні ваги).
    """
    # Обчислення щоденної дохідності
    returns = historic_asset_prices.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(historic_asset_prices.columns)
    
    # Функція для мінімізації (від'ємне Sharpe Ratio)
    def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(portfolio_return - risk_free_rate) / portfolio_volatility
    
    # Обмеження: сума ваг активів = 1
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    
    # Межі ваг активів (0-100%)
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    # Початкові ваги (рівномірний розподіл)
    initial_weights = np.array(num_assets * [1.0 / num_assets])
    
    # Оптимізація для максимізації Sharpe Ratio
    result = sco.minimize(neg_sharpe_ratio, initial_weights, args=(mean_returns, cov_matrix), 
                          method='SLSQP', bounds=bounds, constraints=constraints)
    
    if not result.success:
        raise ValueError("Оптимізація не зійшлася. Спробуйте змінити вхідні дані.")
    
    optimal_weights = result.x
    
    # Формуємо датафрейм із результатами
    df_investments = pd.DataFrame({
        "Тікер": historic_asset_prices.columns,
        "% вкладення": optimal_weights  # Перетворення в %
    })
    
    return df_investments

def calculate_yield(df_yield, df_investment, total_investment):
    """
    Обчислює підсумкову вартість інвестованих активів і дохід за ними.
    
    Параметри:
        df_yield (pd.DataFrame): DataFrame з колонками "Ticker" і "Дохідність".
        df_investment (pd.DataFrame): DataFrame з колонками "Тікер" і "% вкладення".
        total_investment (float): Загальна сума інвестування.
    
    Повертає:
        pd.DataFrame: Датафрейм із колонками "Тікер", "Вкладено", "Вартість на сьогодні", "Дохід".
    """
    # Об'єднання даних по тікерам
    df_result = df_investment.merge(df_yield, left_on="Тікер", right_on="Ticker", how="left")
    
    # Обчислення вкладених коштів у кожен актив
    df_result["Вкладено"] = (df_result["% вкладення"]) * total_investment
    
    # Обчислення вартості активу на сьогодні
    df_result["Вартість на сьогодні"] = df_result["Вкладено"] * (1 + df_result["Доходність"])
    
    # Обчислення доходу
    df_result["Дохід"] = df_result["Вкладено"] * df_result["Доходність"]
    
    # Вибираємо потрібні колонки та повертаємо результат
    return df_result[["Тікер", "Вкладено", "Вартість на сьогодні", "Дохід"]]


def main():
    st.title("Інтерактивна інвестиційна гра")
    st.write("Інвестування - запорука фінансового добробуту. Припустимо, Ви назбирали 10 тис. грн і бажаєте примножити заощадження, вклавши їх у різні активи. Нижче перелічено 10 можливих активів для вкладення:")
    
    assets = get_asset_tickers()
    for asset, ticker in assets.items():
        st.write(f"**{asset}**: `{ticker}`")
    
    st.write("Щоб обрати активи, ознайомтеся з динамікою їх цін за останні 6 місяців до моменту інвестування на графіках:")

    assets = get_asset_tickers()

    historic_assets_prices = get_stock_data(assets)
    st.write("Ціни закриття за останній рік:")
    plot_price_dynamics(historic_assets_prices, 0)
    st.dataframe(historic_assets_prices)
    
    st.write("Тепер розподіліть 10 тис. грн у відсотках між запропонованими активами і зберіть Ваш перший інвестиційний портфель!")

    st.subheader("Розподіл інвестицій")
    st.title("Розподіл інвестицій")
    st.write("Виберіть, скільки відсотків вашого портфеля вкладати в кожен актив.")
    
    
    total_investment = st.number_input("Сума до інвестування (ГРН)", min_value=0.0, value=1000.0, step=100.0)
    
    if "investment" not in st.session_state:
        st.session_state["investment"] = {asset: 0 for asset in assets.keys()}
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        for asset in assets.keys():
            if st.button(f"Все в {asset}", key=f"btn_{asset}"):
                for reset_asset in assets.keys():
                    st.session_state["investment"][reset_asset] = 0
                st.session_state["investment"][asset] = 100
                st.rerun()
    
    total_percentage = 0
    with col1:
        for asset in assets.keys():
            st.session_state["investment"][asset] = st.slider(
                f"% вкласти у {asset}", 0, 100, st.session_state["investment"][asset], key=f"slider_{asset}"
            )
            total_percentage += st.session_state["investment"][asset]
    
    if total_percentage != 100:
        st.warning("Сума всіх відсотків має дорівнювати 100%!")
    else:
        st.write("### Підсумковий розподіл інвестицій")
        user_portfolio = pd.DataFrame({
            "Тікер": list(assets.values()),
            "Актив": list(assets.keys()),
            "% вкладення": [st.session_state["investment"][asset] / 100 for asset in assets.keys()]
        })
        st.dataframe(user_portfolio)
        
        if st.button("Інвестувати"):
            st.success("Інвестиція розподілена успішно!")
            plot_price_dynamics(historic_assets_prices, 1)
            df_yield = calculate_returns(historic_assets_prices)
            st.dataframe(df_yield)
            user_yield = calculate_yield(df_yield, user_portfolio, total_investment)
            st.dataframe(user_yield)
            df_train_historic_prices = historic_assets_prices.iloc[:len(historic_assets_prices) // 2]
            markowitz_portfolio = calculate_markowitz_portfolio(df_train_historic_prices)
            markowitz_yield = calculate_yield(df_yield, markowitz_portfolio, total_investment)
            st.dataframe(markowitz_yield)


if __name__ == "__main__":
    main()
