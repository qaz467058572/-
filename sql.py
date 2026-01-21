import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
import openpyxl
# 讀取
df = pd.read_excel('屬性表.xlsx', skiprows=1)
df_clean = df.dropna(how='all')
df_clean = df_clean.drop('基礎經驗', axis=1)
df_clean = df_clean.dropna(subset=['boss名'])
df_clean = df_clean.reset_index(drop=True)

# 創建SQLite
db_name = 'boss_database.db'
conn = sqlite3.connect(db_name)
cursor = conn.cursor()

# 創建boss表
create_table_sql = '''
CREATE TABLE IF NOT EXISTS bosses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    location TEXT,
    level REAL,
    experience REAL,
    attribute TEXT,
    name TEXT,
    physical_inertia REAL,
    magic_inertia REAL,
    general_inertia REAL,
    physical_defense REAL,
    magic_defense REAL,
    physical_resistance REAL,
    magic_resistance REAL,
    evasion REAL,
    c_resistance REAL,
    dodge_rate REAL,
    block_rate REAL,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
'''

cursor.execute(create_table_sql)
conn.commit()

# 導入
for index, row in df_clean.iterrows():
    insert_sql = '''
    INSERT INTO bosses (
        location, level, experience, attribute, name,
        physical_inertia, magic_inertia, general_inertia,
        physical_defense, magic_defense, physical_resistance,
        magic_resistance, evasion, c_resistance, dodge_rate,
        block_rate, notes
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    '''

    values = (
        row['所在地'], row['等級'], row['經驗'], row['屬性'], row['boss名'],
        row['物理慣性'], row['魔法慣性'], row['一般慣性'],
        row['物理防禦'], row['魔法防禦'], row['物理抗性'],
        row['魔法抗性'], row['迴避'], row['c抵抗'], row['閃躲率'],
        row['阻擋率'], row['補充']
    )

    cursor.execute(insert_sql, values)

conn.commit()

# 驗證
cursor.execute("SELECT COUNT(*) FROM bosses")
count = cursor.fetchone()[0]
print(f"數據庫成功導入 {count} 條Boss數據")


# 关闭连接
conn.close()
print(f"\nSQLite數據庫 '{db_name}' 創建成功！")