import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

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

def calculate_absolute_yield(df_yield, df_investments):
    """
    Обчислює абсолютний прибуток для кожного активу.
    
    Параметри:
        df_yield (pd.DataFrame): DataFrame, де індекси - тікери активів, колонка "Доходність" містить дохідність кожного активу.
        df_investments (pd.DataFrame): DataFrame, де колонка "Тікер" містить тікери, "% вкладення" - відсотки, "Сума (ГРН)" - сума інвестування.
    
    Повертає:
        pd.DataFrame: df_investments з доданою колонкою "Прибуток".
    """
    # Об'єднуємо df_investments з df_yield за тікером
    df_result = df_investments.merge(df_yield, left_on="Тікер", right_index=True, how="left")
    
    # Розрахунок абсолютного прибутку
    df_result["Прибуток"] = df_result["Доходність"] * df_result["Сума (ГРН)"]
    
    return df_result


def main():
    st.title("Інтерактивна інвестиційна гра")
    st.write("Інвестування - запорука фінансового добробуту. Припустимо, Ви назбирали 10 тис. грн і бажаєте примножити заощадження, вклавши їх у різні активи. Нижче перелічено 10 можливих активів для вкладення:")
    
    assets = get_asset_tickers()
    for asset, ticker in assets.items():
        st.write(f"**{asset}**: `{ticker}`")
    
    st.write("Щоб обрати активи, ознайомтеся з динамікою їх цін за останні 6 місяців до моменту інвестування на графіках:")

    data = get_stock_data(get_asset_tickers())
    st.write("Ціни закриття за останній рік:")
    plot_price_dynamics(data, 0)
    st.dataframe(data)
    
    st.write("Тепер розподіліть 10 тис. грн у відсотках між запропонованими активами і зберіть Ваш перший інвестиційний портфель!")

    st.subheader("Розподіл інвестицій")
    st.title("Розподіл інвестицій")
    st.write("Виберіть, скільки відсотків вашого портфеля вкладати в кожен актив.")
    
    assets = get_asset_tickers()
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
        df_investment = pd.DataFrame({
            "Тікер": list(assets.values()),
            "Актив": list(assets.keys()),
            "% вкладення": [st.session_state["investment"][asset] for asset in assets.keys()],
            "Сума (ГРН)": [(st.session_state["investment"][asset] / 100) * total_investment for asset in assets.keys()]
        })
        st.dataframe(df_investment)
        
        if st.button("Інвестувати"):
            st.success("Інвестиція розподілена успішно!")
            plot_price_dynamics(data, 1)
            df_yield = calculate_returns(data)
            user_yield = calculate_absolute_yield(df_yield, df_investment)
            st.dataframe(user_yield)
            total_initial = user_yield["Сума (ГРН)"].sum()  # Початкова сума інвестування
            total_profit = user_yield["Прибуток"].sum()  # Загальний прибуток
            total_final = total_initial + total_profit  # Фінальна сума після інвестицій

            st.write(f"📈 Початкова сума інвестицій: {total_initial:,.2f} ГРН")
            st.write(f"💰 Загальний дохід від інвестицій: {total_profit:,.2f} ГРН")
            st.write(f"🏆 Загальна сума після інвестицій: {total_final:,.2f} ГРН")


if __name__ == "__main__":
    main()
