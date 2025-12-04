# database_logger.py
import sqlite3
import datetime

DB_NAME = "sessions.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # Create a table to store session summaries
    c.execute('''CREATE TABLE IF NOT EXISTS sessions
                 (session_id TEXT PRIMARY KEY, 
                  start_time TIMESTAMP, 
                  end_time TIMESTAMP, 
                  message_count INTEGER DEFAULT 0)''')
    conn.commit()
    conn.close()

def log_session_start(session_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    start_time = datetime.datetime.now()
    try:
        c.execute("INSERT INTO sessions (session_id, start_time) VALUES (?, ?)", 
                  (session_id, start_time))
        conn.commit()
    except sqlite3.IntegrityError:
        pass # Session already exists
    conn.close()

def update_session_end(session_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    end_time = datetime.datetime.now()
    # Update end time and increment message count
    c.execute("""UPDATE sessions 
                 SET end_time = ?, message_count = message_count + 1 
                 WHERE session_id = ?""", (end_time, session_id))
    conn.commit()
    conn.close()