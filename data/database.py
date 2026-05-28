import sqlite3
import bcrypt
import pandas as pd
import os

DB_PATH = "crypto_app.db"

def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = get_connection()
    c = conn.cursor()
    
    # Create Users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    ''')
    
    # Create Portfolios table
    c.execute('''
        CREATE TABLE IF NOT EXISTS portfolios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            asset TEXT NOT NULL,
            amount REAL NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    
    # Create Chat History table
    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    
    conn.commit()
    conn.close()

def create_user(username, password):
    conn = get_connection()
    c = conn.cursor()
    
    try:
        # Check if user exists
        c.execute("SELECT * FROM users WHERE username = ?", (username,))
        if c.fetchone():
            return False, "Username already exists."
            
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        
        c.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, hashed))
        conn.commit()
        return True, "Account created successfully."
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()

def verify_user(username, password):
    conn = get_connection()
    c = conn.cursor()
    
    try:
        c.execute("SELECT id, password_hash FROM users WHERE username = ?", (username,))
        result = c.fetchone()
        
        if result:
            user_id, hashed = result
            if bcrypt.checkpw(password.encode('utf-8'), hashed):
                return user_id, "Login successful."
        return None, "Invalid username or password."
    finally:
        conn.close()

def reset_password(username, new_password):
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute("SELECT id FROM users WHERE username = ?", (username,))
        if not c.fetchone():
            return False, "Username not found."
            
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(new_password.encode('utf-8'), salt)
        
        c.execute("UPDATE users SET password_hash = ? WHERE username = ?", (hashed, username))
        conn.commit()
        return True, "Password reset successfully."
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()

def get_user_portfolio(user_id):
    conn = get_connection()
    try:
        df = pd.read_sql_query("SELECT asset as Asset, amount as Amount FROM portfolios WHERE user_id = ?", conn, params=(user_id,))
        if df.empty:
            # Return default template if no portfolio exists
            return pd.DataFrame({'Asset': ['Bitcoin', 'Ethereum', 'USDT'], 'Amount': [0.0, 0.0, 0.0]})
        return df
    finally:
        conn.close()

def save_user_portfolio(user_id, df):
    conn = get_connection()
    c = conn.cursor()
    try:
        # Clear existing portfolio
        c.execute("DELETE FROM portfolios WHERE user_id = ?", (user_id,))
        
        # Insert new rows
        for _, row in df.iterrows():
            if pd.notnull(row['Amount']) and float(row['Amount']) >= 0:
                c.execute("INSERT INTO portfolios (user_id, asset, amount) VALUES (?, ?, ?)", 
                          (user_id, row['Asset'], float(row['Amount'])))
                
        conn.commit()
        return True
    except Exception as e:
        print(f"Error saving portfolio: {e}")
        return False
    finally:
        conn.close()

def get_chat_history(user_id):
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute("SELECT role, content FROM chat_history WHERE user_id = ? ORDER BY timestamp ASC", (user_id,))
        rows = c.fetchall()
        return [{"role": row[0], "content": row[1]} for row in rows]
    finally:
        conn.close()

def save_chat_message(user_id, role, content):
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO chat_history (user_id, role, content) VALUES (?, ?, ?)", (user_id, role, content))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error saving chat message: {e}")
        return False
    finally:
        conn.close()

def clear_chat_history(user_id):
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute("DELETE FROM chat_history WHERE user_id = ?", (user_id,))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error clearing chat history: {e}")
        return False
    finally:
        conn.close()
