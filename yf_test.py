import yfinance as yf
import matplotlib.pyplot as plt
import time

# Вибір активу (Apple)
stock_symbol = "AAPL"

# Функція для завантаження даних з повторними спробами
def download_stock_data(symbol, retries=3, delay=5):
    for i in range(retries):
        try:
            print(f"Спроба {i+1}: Завантаження даних для {symbol}...")
            data = yf.download(symbol, period="360d", interval="1d", auto_adjust=True)
            if not data.empty:
                print("Дані успішно завантажені!")
                print(data.head())
                return data
            else:
                print("Отримано порожній набір даних. Спроба ще раз...")
        except Exception as e:
            print(f"Помилка: {e}. Повторюємо спробу через {delay} секунд...")
            time.sleep(delay)
    print("Не вдалося отримати дані після декількох спроб.")
    return None

# Завантаження даних
stock_data = download_stock_data(stock_symbol)

# Перевірка, чи є отримані дані
if stock_data is None:
    print("Помилка: Дані не були завантажені. Перевірте з'єднання або коректність символу активу.")
else:
    print(stock_data.head())
    
    # Перевірка, які колонки є в завантажених даних
    print("Доступні колонки у датафреймі:", stock_data.columns)
    
    # Використовуємо 'Close' замість 'Adj Close'
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data.index, stock_data["Close"], label=f"Ціна {stock_symbol}")
    plt.xlabel("Дата")
    plt.ylabel("Ціна закриття ($)")
    plt.title(f"Динаміка вартості {stock_symbol} за останні 360 днів")
    plt.legend()
    plt.grid()
    plt.show()
