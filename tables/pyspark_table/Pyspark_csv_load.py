import csv
import psycopg2

CSV_FILE = "new_pyspark_data_generation.csv"  # Change to your CSV file path

# PostgreSQL connection details — change as needed
PG_HOST = ""
PG_PORT = 5432
PG_DB = ""
PG_USER = ""
PG_PASS = ""

TABLE_CREATION_SQL = """
CREATE TABLE IF NOT EXISTS pyspark_data_table (
    pyspark_version VARCHAR(10),
    module VARCHAR(255),
    qualname VARCHAR(500) PRIMARY KEY,
    type VARCHAR(50),
    parameters TEXT,
    returns TEXT,
    deprecated BOOLEAN,
    versionadded VARCHAR(20),
    versionchanged VARCHAR(20),
    versionchanged_note TEXT,
    deprecated_in VARCHAR(20),
    deprecated_note TEXT,
    replacement TEXT,
    docstring TEXT
);
"""

def create_table_and_load_csv():
    conn = psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        dbname=PG_DB,
        user=PG_USER,
        password=PG_PASS
    )
    cursor = conn.cursor()

    # Create table
    cursor.execute(TABLE_CREATION_SQL)
    conn.commit()
    print("Table created or already exists.")

    # Load CSV
    with open(CSV_FILE, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        count = 0

        for row in reader:
            deprecated_val = row.get('deprecated', '').strip().lower() == 'true'
            try:
                cursor.execute("""
                    INSERT INTO pyspark_data_table (
                        pyspark_version, module, qualname, type, parameters, returns,
                        deprecated, versionadded, versionchanged, versionchanged_note,
                        deprecated_in, deprecated_note, replacement, docstring
                    ) VALUES (
                        %(pyspark_version)s, %(module)s, %(qualname)s, %(type)s, %(parameters)s, %(returns)s,
                        %(deprecated)s, %(versionadded)s, %(versionchanged)s, %(versionchanged_note)s,
                        %(deprecated_in)s, %(deprecated_note)s, %(replacement)s, %(docstring)s
                    )
                    ON CONFLICT (qualname) DO NOTHING;
                """, {
                    'pyspark_version': row.get('pyspark_version'),
                    'module': row.get('module'),
                    'qualname': row.get('qualname'),
                    'type': row.get('type'),
                    'parameters': row.get('parameters'),
                    'returns': row.get('returns'),
                    'deprecated': deprecated_val,
                    'versionadded': row.get('versionadded'),
                    'versionchanged': row.get('versionchanged'),
                    'versionchanged_note': row.get('versionchanged_note'),
                    'deprecated_in': row.get('deprecated_in'),
                    'deprecated_note': row.get('deprecated_note'),
                    'replacement': row.get('replacement'),
                    'docstring': row.get('docstring')
                })
                count += 1
            except Exception as e:
                print(f"Error inserting row {row.get('qualname')}: {e}")

    conn.commit()
    cursor.close()
    conn.close()
    print(f"Inserted {count} rows into pyspark_apis table.")

if __name__ == "__main__":
    create_table_and_load_csv()
