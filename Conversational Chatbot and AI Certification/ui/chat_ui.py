import streamlit as st
import requests
import pandas as pd
import sqlite3
import os
from datetime import datetime

# --- CONFIGURATION ---
ST_PAGE_TITLE = "University Chatbot"
RASA_API_URL = os.getenv("RASA_URL", "http://localhost:5005/webhooks/rest/webhook")
DB_PATH = "sessions.db" # Path to the DB created by the Rasa actions

st.set_page_config(page_title=ST_PAGE_TITLE, layout="wide")

# --- CSS FOR STYLING ---
st.markdown("""
<style>
    .user-msg {
        background-color: #e6f3ff;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        text-align: right;
        color: #000;
    }
    .bot-msg {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        text-align: left;
        color: #000;
    }
    /* Hide Streamlit main menu and footer for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR: ANALYTICS ---
def load_data():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM sessions ORDER BY start_time DESC", conn)
    conn.close()
    return df

with st.sidebar:
    st.header("📊 Analytics Dashboard")
    
    df = load_data()
    
    if not df.empty:
        total_sessions = len(df)
        active_today = df[pd.to_datetime(df['start_time']).dt.date == datetime.now().date()].shape[0]
        
        col1, col2 = st.columns(2)
        col1.metric("Total Chats", total_sessions)
        col2.metric("Today", active_today)
        
        st.divider()
        st.subheader("Session History")
        st.dataframe(df[['session_id', 'start_time', 'message_count']], use_container_width=True)
    else:
        st.info("No session data available yet.")
        
    if st.button("Refresh Data"):
        st.rerun()

# --- MAIN CHAT INTERFACE ---

# Header with Export Button
col_title, col_btn = st.columns([8, 2])
with col_title:
    st.title(f"🤖 {ST_PAGE_TITLE}")
with col_btn:
    # Export Chat Logic
    if "messages" in st.session_state and len(st.session_state.messages) > 0:
        chat_df = pd.DataFrame(st.session_state.messages)
        csv = chat_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Chat",
            data=csv,
            file_name='chat_history.csv',
            mime='text/csv'
        )
    else:
        st.button("📥 Export Chat", disabled=True)

st.divider()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input & Microphone Area
# Note: Streamlit doesn't support a mic button *inside* chat_input yet, 
# so we place a disabled button nearby or use a column layout below.

prompt = st.chat_input("Type your message here...")

if prompt:
    # 1. Display User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Get Bot Response
    try:
        # Generate a unique session ID for Streamlit user if not exists
        if "session_id" not in st.session_state:
            import uuid
            st.session_state.session_id = str(uuid.uuid4())

        payload = {"sender": st.session_state.session_id, "message": prompt}
        response = requests.post(RASA_API_URL, json=payload)
        
        if response.status_code == 200:
            bot_responses = response.json()
            for resp in bot_responses:
                bot_text = resp.get("text", "I received that, but have no text response.")
                
                with st.chat_message("assistant"):
                    st.markdown(bot_text)
                st.session_state.messages.append({"role": "assistant", "content": bot_text})
        else:
            st.error("Error connecting to Rasa server.")
            
    except Exception as e:
        st.error(f"Connection failed: {e}")

# Footer Bar (Microphone Placeholder)
with st.container():
    col_mic, col_spacer = st.columns([1, 10])
    with col_mic:
        st.button("🎤", disabled=True, help="ASR coming soon!")