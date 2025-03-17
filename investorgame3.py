import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np
import gspread
import logging
import json
import base64
from oauth2client.service_account import ServiceAccountCredentials

# Налаштування логування
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def get_google_credentials():
    """Load Google Sheets credentials from a local file or Streamlit secrets."""
    try:
        # Try loading from local file first
        with open("service_account.json", "r") as f:
            credentials_dict = json.load(f)
            logging.info("✅ Завантажено Google Credentials з локального файлу.")
    except FileNotFoundError:
        try:
            # If file not found, try loading from Streamlit secrets
            encoded_creds = st.secrets["GOOGLE_CREDENTIALS"]
            creds_json = base64.b64decode(encoded_creds).decode("utf-8")
            credentials_dict = json.loads(creds_json)
            logging.info("✅ Завантажено Google Credentials з Streamlit secrets.")
        except Exception as e:
            logging.error(f"❌ Помилка завантаження Google Credentials: {e}")
            st.error("Помилка при завантаженні Google Credentials. Перевірте секрети Streamlit або наявність локального файлу.")
            return None
    
    return ServiceAccountCredentials.from_json_keyfile_dict(credentials_dict)

def send_to_google_sheets(name, phone):
    """Записує дані користувача в Google Sheets."""
    logging.info("Початок запису в Google Sheets")
    try:
        credentials = get_google_credentials()
        if not credentials:
            return
        
        client = gspread.authorize(credentials)
        sheet = client.open("future cybernetics from investment game").sheet1
        
        row = [name, phone]
        sheet.append_row(row)
        
        logging.info(f"Успішно записано в Google Sheets: {row}")
        st.success("Дані успішно надіслані!")
    except Exception as e:
        logging.error(f"Помилка запису в Google Sheets: {e}")
        st.error(f"Помилка при записі в Google Sheets: {e}")

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

def calculate_returns(a_date_prices,b_date_prices):
    returns = (b_date_prices.iloc[0] - a_date_prices.iloc[0]) / a_date_prices.iloc[0]
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

def show_yield_histogram(df_yield):
    """
    Displays a histogram comparing asset returns in a Streamlit app.

    Parameters:
        df_yield (pd.DataFrame): DataFrame with columns "Тікер" (ticker) and "Дохідність" (return).

    Returns:
        None (renders the histogram in Streamlit).
    """
    if df_yield.empty:
        st.warning("⚠️ DataFrame is empty. Please provide valid data.")
        return

    # Extract tickers and yield values
    tickers = df_yield.index
    yields = df_yield["Доходність"]

    # Plot histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(tickers, yields, color=['green' if y >= 0 else 'red' for y in yields], alpha=0.75)

    # Formatting
    ax.set_xlabel("Тікер", fontsize=12)
    ax.set_ylabel("Дохідність", fontsize=12)
    ax.set_title("Порівняння дохідностей активів", fontsize=14)
    ax.set_xticklabels(tickers, rotation=45, ha="right")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Show the plot in Streamlit
    st.pyplot(fig)

def analyze_multiple_portfolios(portfolios: dict, a_date_prices: pd.DataFrame, b_date_prices: pd.DataFrame, total_investment: float):
    """
    Analyzes multiple portfolios and returns a combined DataFrame with performance results.

    Parameters:
        portfolios (dict): Dictionary of portfolio DataFrames with structure {'Portfolio Name': df_structure}
        a_date_prices (pd.DataFrame): Historical prices on the start date.
        b_date_prices (pd.DataFrame): Historical prices on the end date.
        total_investment (float): The total amount invested at the start.

    Returns:
        pd.DataFrame: Combined results with portfolio performance, indexed by portfolio names.
    """

    results = []

    for portfolio_name, df_portfolio in portfolios.items():
        # Compute performance for this portfolio
        df_result = calculate_portfolio_performance(df_portfolio, a_date_prices, b_date_prices, total_investment)
        
        # Add portfolio name as index
        df_result.index = [portfolio_name]
        
        # Append result to list
        results.append(df_result)

    # Combine all results into a single DataFrame
    df_combined = pd.concat(results)

    return df_combined

def calculate_portfolio_performance(df_portfolio, a_date_prices, b_date_prices, total_investment):
    """
    Calculates the return and final value of a portfolio based on asset prices on two dates.

    Parameters:
        df_portfolio (pd.DataFrame): DataFrame with "Тікер" (tickers) and "% вкладення" (weights).
        a_date_prices (pd.DataFrame): DataFrame with asset prices on the start date (index should be one date).
        b_date_prices (pd.DataFrame): DataFrame with asset prices on the end date (index should be one date).
        total_investment (float): The total amount invested at the start.

    Returns:
        pd.DataFrame: DataFrame with columns "Дохідність портфеля", "Вартість портфеля".
    """
    
    # Ensure tickers in portfolio exist in price data
    tickers = df_portfolio["Тікер"].values
    valid_tickers = [ticker for ticker in tickers if ticker in a_date_prices.columns and ticker in b_date_prices.columns]
    
    if not valid_tickers:
        raise ValueError("⚠️ No valid tickers found in price data.")

    # Extract prices for selected tickers
    initial_prices = a_date_prices[valid_tickers].iloc[0]  # Prices at date A
    final_prices = b_date_prices[valid_tickers].iloc[0]    # Prices at date B

    # Calculate return for each asset
    returns = final_prices / initial_prices  # Return factor (e.g., 1.05 means +5%)

    # Calculate each asset's contribution to portfolio value
    df_portfolio = df_portfolio.set_index("Тікер")  # Ensure index is tickers
    df_portfolio = df_portfolio.loc[valid_tickers]  # Keep only valid tickers

    df_portfolio["Дохідність активу"] = returns
    df_portfolio["Вартість активу"] = total_investment * df_portfolio["% вкладення"] * df_portfolio["Дохідність активу"]

    # Calculate total portfolio value and return
    portfolio_value = df_portfolio["Вартість активу"].sum()
    portfolio_return = portfolio_value / total_investment - 1  # Convert to percentage return

    # Create output DataFrame
    df_result = pd.DataFrame({
        "Дохідність портфеля": [portfolio_return],
        "Вартість портфеля": [portfolio_value]
    })

    return df_result

def plot_portfolio_asset_distribution_streamlit(portfolios):
    """
    Displays a grouped bar chart in Streamlit showing asset allocations in different portfolios.

    Parameters:
        portfolios (dict): Dictionary where keys are portfolio names and values are DataFrames 
                          with columns "Тікер" (ticker) and "% вкладення" (allocation as fractions).

    Returns:
        None (renders the histogram in Streamlit).
    """
    if not portfolios:
        st.warning("⚠️ No portfolios provided.")
        return

    # Extract all unique tickers from all portfolios
    unique_tickers = sorted(set(ticker for df in portfolios.values() for ticker in df["Тікер"]))

    # Number of assets and portfolios
    num_assets = len(unique_tickers)
    num_portfolios = len(portfolios)

    # Set width of bars
    bar_width = 0.8 / num_portfolios  # Adjust for better spacing

    # X-axis positions for tickers
    x_positions = np.arange(num_assets)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Loop through each portfolio and plot its allocations
    for i, (portfolio_name, df_portfolio) in enumerate(portfolios.items()):
        # Get allocation for each asset (set 0 if asset is not in portfolio)
        asset_allocations = [df_portfolio.set_index("Тікер")["% вкладення"].get(ticker, 0) for ticker in unique_tickers]

        # Shift bars for different portfolios
        ax.bar(x_positions + i * bar_width, asset_allocations, width=bar_width, label=portfolio_name, alpha=0.75)

    # Formatting
    ax.set_xlabel("Активи", fontsize=12)
    ax.set_ylabel("Частка у портфелі", fontsize=12)
    ax.set_title("Розподіл активів у портфелях", fontsize=14)
    ax.set_xticks(x_positions + bar_width * (num_portfolios - 1) / 2)  # Centering labels
    ax.set_xticklabels(unique_tickers, rotation=45, ha="right")
    ax.legend(title="Портфелі")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Show the plot in Streamlit
    st.subheader("Давай порівняємо твій портфель з портфелями зібраними різними ШІ та алгортимами оптимізації")
    st.pyplot(fig)

def analyze_player_performance_with_leaderboard(df_performance):
    """
    Analyzes the ranking of the player's portfolio, displays a leaderboard, and shows a message in Streamlit.

    Parameters:
        df_performance (pd.DataFrame): DataFrame with index as portfolio names and 
                                       "Дохідність портфеля" (portfolio return) as a column.

    Returns:
        None (renders leaderboard and message in Streamlit).
    """
    if df_performance.empty or "Дохідність портфеля" not in df_performance.columns:
        st.warning("⚠️ Немає даних для аналізу портфелів.")
        return

    # Sort portfolios by return in descending order
    df_sorted = df_performance.sort_values(by="Дохідність портфеля", ascending=False)

    # Display leaderboard
    st.subheader("🏆 Таблиця лідерів портфелів")
    st.dataframe(df_sorted.style.format({"Дохідність портфеля": "{:.2%}"}))

    # Check player's position
    if "Гравець" in df_sorted.index:
        player_rank = df_sorted.index.get_loc("Гравець")

        if player_rank == 0:
            st.success("🎉 Вау! В тебе талант до інвестицій! Вступай на кафедру економіки та економічної кібернетики аби в повній мірі розвинути свої здібності!")
        elif player_rank == len(df_sorted) - 1:
            st.error("📉 Хочеш покращити свої прибутки? Вступай на кафедру економіки та економічної кібернетики і дізнайся як використовувати сучасні моделі для створення оптимальних портфелів!")
        else:
            st.info("📈 Непогано, але є куди зростати! Вступай на кафедру економіки та економічної кібернетики і дізнайся як використовувати сучасні моделі для створення оптимальних портфелів!")

def show_dataframe_with_total(df):
    # Клонуємо датафрейм, щоб залишити оригінальний незмінним
    df_copy = df.copy()

    # Визначаємо всі числові колонки
    numeric_cols = df_copy.select_dtypes(include=['number']).columns

    # Визначаємо колонки, які містять частки (значення між 0 і 1)
    fraction_cols = [col for col in numeric_cols if df_copy[col].between(0, 1).all()]

    # Створюємо рядок "Всього" із сумами для числових колонок
    total_row = {col: df_copy[col].sum() for col in numeric_cols}
    total_row["Тікер"] = "Всього"

    # Додаємо цей рядок до датафрейму
    df_copy = pd.concat([df_copy, pd.DataFrame([total_row])], ignore_index=True)

    # Форматуємо числові значення
    format_dict = {col: "{:,.2f}" for col in numeric_cols}  # Двома знаками після коми
    for col in fraction_cols:
        format_dict[col] = "{:.2%}"  # Форматування у відсотки

    # Відображаємо у Streamlit
    st.dataframe(df_copy.style.format(format_dict))


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
    
    st.write("Тепер розподіліть 10 тис. грн у відсотках між запропонованими активами і зберіть Ваш перший інвестиційний портфель!")

    st.subheader("Розподіл інвестицій")
    st.title("Розподіл інвестицій")
    st.write("Виберіть, скільки відсотків вашого портфеля вкладати в кожен актив.")
    
    
    total_investment = st.number_input("Сума до інвестування (ГРН)", min_value=0.0, value=10000.0, step=1000.0)
    
    if "investment" not in st.session_state:
        st.session_state["investment"] = {asset: 100 / len(assets) for asset in assets}
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        for asset in assets.keys():
            if st.button(f"Все в {asset}", key=f"btn_{asset}"):
                for reset_asset in assets.keys():
                    st.session_state["investment"][reset_asset] = 0.0
                st.session_state["investment"][asset] = 100.0
                st.rerun()
    
    total_percentage = 0
    with col1:
        for asset in assets.keys():
            st.session_state["investment"][asset] = st.slider(
                f"% вкласти у {asset}", 0.0, 100.0, st.session_state["investment"][asset], 1.0, key=f"slider_{asset}"
            )
            total_percentage += st.session_state["investment"][asset]
    
    if total_percentage != 100:
        st.warning(f"Поточна сума відсотків: {total_percentage}%. Сума всіх відсотків має дорівнювати 100%!")
    else:
        st.write("### Підсумковий розподіл інвестицій")
        user_portfolio = pd.DataFrame({
            "Тікер": list(assets.values()),
            "Актив": list(assets.keys()),
            "Сума": [st.session_state["investment"][asset] / 100 * total_investment for asset in assets.keys()],
        