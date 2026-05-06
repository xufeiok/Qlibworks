import clickhouse_connect

def main():
    client = clickhouse_connect.get_client(
        host="192.168.10.102",
        port=18123,
        user="xufei",
        password="xf1987216",
        database="quant_db"
    )
    
    for table in ['daily_indicators', 'financial_indicators']:
        print(f"--- {table} ---")
        res = client.query(f"DESCRIBE TABLE {table}")
        for row in res.result_rows:
            print(f"{row[0]}: {row[1]}")
        print()

if __name__ == '__main__':
    main()