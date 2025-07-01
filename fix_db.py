import sqlite3

conn = sqlite3.connect("review_data.db")
cursor = conn.cursor()

# Add missing columns if they don't already exist
try:
    cursor.execute("ALTER TABLE reviews ADD COLUMN date TEXT;")
    print("✅ Added 'date' column.")
except sqlite3.OperationalError:
    print("ℹ️ 'date' column already exists.")

try:
    cursor.execute("ALTER TABLE reviews ADD COLUMN admin_tag TEXT DEFAULT NULL;")
    print("✅ Added 'admin_tag' column.")
except sqlite3.OperationalError:
    print("ℹ️ 'admin_tag' column already exists.")

conn.commit()
conn.close()
