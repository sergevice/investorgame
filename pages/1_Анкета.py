import streamlit as st
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

# Інтерфейс Streamlit
st.title("Гра завершена! 🎉")

st.subheader("Давай знайомитися! Залишай заявку, аби отримувати актуальну інформацію!")

# Використання session_state для збереження введених даних між оновленнями
if "name" not in st.session_state:
    st.session_state["name"] = ""
if "phone" not in st.session_state:
    st.session_state["phone"] = ""

name = st.text_input("Імʼя",  key="name")
phone = st.text_input("Телефон",  key="phone")

if st.button("Відправити"):
    if name.strip() and phone.strip():
        send_to_google_sheets(name, phone)
    else:
        st.warning("Будь ласка, заповніть усі поля.")
