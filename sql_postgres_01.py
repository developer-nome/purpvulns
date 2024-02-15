import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()
dbname = os.getenv('DB_NAME')
user = os.getenv('DB_USER')
password = os.getenv('DB_PASSWORD')
conn = psycopg2.connect(database=dbname, user=user, password=password)
cur = conn.cursor()

user_input = input("Enter the item id: ")
query = f"SELECT * FROM mytable WHERE id = {user_input}"
cur.execute(query)
row = cur.fetchone()
print(row)

conn.close()
