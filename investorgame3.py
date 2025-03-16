import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.optimize import minimize
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

def compute_sharpe_ratio(df_structure, df_prices, risk_free_rate):
    """
    Calculate the Sharpe ratio of a portfolio.
    
    Parameters:
        df_structure (pd.DataFrame): DataFrame with columns ["Тікер", "% вкладення"]
        df_prices (pd.DataFrame): DataFrame where index is dates, columns are tickers, and values are prices
        risk_free_rate (float): Annualized risk-free rate (default: 2%)
    
    Returns:
        float: Sharpe ratio of the portfolio
    """
    
    # Convert allocation percentages to weights (ensure sum = 1)
    df_structure = df_structure.copy()
    df_structure["% вкладення"] /= 100  # Convert from percent to decimal
    
    # Filter df_prices to include only assets in the portfolio
    tickers = df_structure["Тікер"].values
    df_prices = df_prices[tickers]
    
    # Compute daily returns
    df_returns = df_prices.pct_change().dropna()
    
    # Compute portfolio return: weighted sum of asset returns
    weights = df_structure.set_index("Тікер")["% вкладення"].reindex(df_returns.columns).values
    portfolio_returns = df_returns @ weights  # Matrix multiplication

    # Compute portfolio statistics
    mean_return = portfolio_returns.mean() * 252  # Annualized return
    std_dev = portfolio_returns.std() * np.sqrt(252)  # Annualized volatility

    # Compute Sharpe Ratio
    sharpe_ratio = (mean_return - risk_free_rate) / std_dev

    return sharpe_ratio

def Markowitz_optimised_portfolio(df_prices, risk_free_rate=0.02):
    """
    Optimizes portfolio allocation using Markowitz mean-variance optimization 
    to maximize the Sharpe ratio and returns a readable DataFrame.
    """
    
    # Compute daily returns
    df_returns = df_prices.pct_change().dropna()
    num_assets = df_returns.shape[1]
    
    # Initial equal-weight allocation
    initial_weights = np.ones(num_assets) / num_assets
    tickers = df_prices.columns

    # Function to minimize (negative Sharpe ratio)
    def negative_sharpe(weights):
        df_structure = pd.DataFrame({"Тікер": tickers, "% вкладення": weights * 100})
        return -compute_sharpe_ratio(df_structure, df_prices, risk_free_rate)

    # Constraints: sum of weights must be 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    
    # Bounds: each weight between 0 and 1 (no short selling)
    bounds = [(0, 1) for _ in range(num_assets)]

    # Optimization
    result = minimize(negative_sharpe, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    # Extract optimized weights
    optimized_weights = result.x

    # Create portfolio structure DataFrame
    df_optimized = pd.DataFrame({
        "Тікер": tickers,
        "% вкладення": optimized_weights  # Convert to percentage
    })

    # Format percentage values to 2 decimal places
    '''df_optimized["% вкладення"] = df_optimized["% вкладення"].apply(lambda x: f"{x:.2f}%")'''

    return df_optimized


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
        st.session_state["investment"] = {asset: 100 / len(assets) for asset in assets}
    
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
                f"% вкласти у {asset}", 0.0, 100.0, st.session_state["investment"][asset], key=f"slider_{asset}"
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
            markowitz_portfolio = Markowitz_optimised_portfolio(df_train_historic_prices)
            markowitz_yield = calculate_yield(df_yield, markowitz_portfolio, total_investment)
            st.dataframe(markowitz_portfolio)
            st.dataframe(markowitz_yield)


if __name__ == "__main__":
    main()
