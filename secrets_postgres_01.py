import psycopg2
import os

def get_order(order_id):
    database = "postgres"
    user = "postgres"
    password = "postgres"
    host = "localhost"

    # This is a comment line placed here to replicate a real-world example
    conn = psycopg2.connect(database=database, user=user, password=password, host=host)
    cur = conn.cursor()
    query = "SELECT * FROM items WHERE id=%s"
    cur.execute(query, (order_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row

if __name__ == "__main__":
    order_id = int(input("Enter the order id: "))
    order = get_order(order_id)
    print(f"Order: {order}")
