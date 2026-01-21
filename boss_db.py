# -*- coding: utf-8 -*-
import aiosqlite
import random

# 数据库文件路径（和你创建的boss_database.db一致）
DB_PATH = "boss_database.db"


# 初始化数据库（创建bosses表）
async def init_boss_db():
    async with aiosqlite.connect(DB_PATH) as db:
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
        await db.execute(create_table_sql)
        await db.commit()
        print("BOSS数据库初始化完成（表结构匹配Excel导入脚本）")


# 模糊查询BOSS名称
async def query_boss_info(keyword):
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute('''
            SELECT 
                location, level, experience, attribute, name,
                physical_inertia, magic_inertia, general_inertia,
                physical_defense, magic_defense, physical_resistance,
                magic_resistance, evasion, c_resistance, dodge_rate,
                block_rate, notes
            FROM bosses
            WHERE name LIKE ?
        ''', (f'%{keyword}%',))

        rows = await cursor.fetchall()
        if not rows:
            return f"未查詢到包含「{keyword}」的BOSS信息～"

        result = "📌 BOSS屬性查詢結果：\n"
        for idx, row in enumerate(rows, 1):
            (location, level, experience, attribute, name,
             physical_inertia, magic_inertia, general_inertia,
             physical_defense, magic_defense, physical_resistance,
             magic_resistance, evasion, c_resistance, dodge_rate,
             block_rate, notes) = row

            notes = notes if notes is not None else "無補充信息"

            result += f"""
【{idx}. {name}】
📍 所在地：{location}
⚡ 等級：{level} | 📈 經驗值：{experience} | 📌 屬性：{attribute}
🛡️ 防禦相關：
  物理防禦：{physical_defense} | 魔法防禦：{magic_defense}
  物理抗性：{physical_resistance} | 魔法抗性：{magic_resistance}
  C抵抗：{c_resistance}
🚫 慣性相關：
  物理慣性：{physical_inertia} | 魔法慣性：{magic_inertia} | 一般慣性：{general_inertia}
💨 躲避相關：
  迴避：{evasion} | 閃躲率：{dodge_rate} | 阻擋率：{block_rate}
📝 補充：{notes}
"""
        return result.strip()


# 查看所有BOSS名称
async def get_all_boss_names():
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute('SELECT DISTINCT name FROM bosses ORDER BY name')
        rows = await cursor.fetchall()
        if not rows:
            return "數據庫中暫無BOSS數據～"
        names = [row[0] for row in rows]
        return "📋 數據庫中所有BOSS列表：\n" + "\n".join([f"• {name}" for name in names])


# 自动推荐BOSS（无参数时备用）
async def auto_recommend_boss():
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute('''
            SELECT 
                location, level, experience, attribute, name,
                physical_inertia, magic_inertia, general_inertia,
                physical_defense, magic_defense, physical_resistance,
                magic_resistance, evasion, c_resistance, dodge_rate,
                block_rate, notes
            FROM bosses
        ''')
        all_bosses = await cursor.fetchall()

        if not all_bosses:
            return "⚠️ 數據庫中暫無BOSS數據，無法自動推薦～"

        all_names = [boss[4] for boss in all_bosses]
        names_str = "\n".join([f"• {name}" for name in all_names])
        recommend_count = min(2, len(all_bosses))
        recommend_bosses = random.sample(all_bosses, recommend_count)

        result = f"""
🤖 你未指定BOSS关键词，為你自動查詢所有BOSS並推薦以下內容：

📋 數據庫中所有BOSS列表：
{names_str}

🌟 隨機推薦 {recommend_count} 個BOSS詳細信息：
"""
        for idx, row in enumerate(recommend_bosses, 1):
            (location, level, experience, attribute, name,
             physical_inertia, magic_inertia, general_inertia,
             physical_defense, magic_defense, physical_resistance,
             magic_resistance, evasion, c_resistance, dodge_rate,
             block_rate, notes) = row

            notes = notes if notes is not None else "無補充信息"

            result += f"""
【{idx}. {name}】
📍 所在地：{location}
⚡ 等級：{level} | 📈 經驗值：{experience} | 📌 屬性：{attribute}
🛡️ 防禦相關：
  物理防禦：{physical_defense} | 魔法防禦：{magic_defense}
  物理抗性：{physical_resistance} | 魔法抗性：{magic_resistance}
  C抵抗：{c_resistance}
🚫 慣性相關：
  物理慣性：{physical_inertia} | 魔法慣性：{magic_inertia} | 一般慣性：{general_inertia}
💨 躲避相關：
  迴避：{evasion} | 閃躲率：{dodge_rate} | 阻擋率：{block_rate}
📝 補充：{notes}
"""
        result += "\n💡 你可以輸入「-boss + BOSS名」查詢指定BOSS（例如：-boss 火龍）"
        return result.strip()


# 支持的筛选字段和运算符（手动精准筛选用）
SUPPORTED_FIELDS = {
    "屬性": "attribute",
    "所在地": "location",
    "等級": "level",
    "經驗": "experience",
    "物理防禦": "physical_defense",
    "魔法防禦": "magic_defense",
    "物理抗性": "physical_resistance",
    "魔法抗性": "magic_resistance",
    "物理慣性": "physical_inertia",
    "魔法慣性": "magic_inertia",
    "迴避": "evasion",
    "閃躲率": "dodge_rate",
    "阻擋率": "block_rate",
    "c抵抗": "c_resistance"
}
SUPPORTED_OPERATORS = {
    "=": "=",
    "==": "=",
    ">": ">",
    "<": "<",
    "≥": ">=",
    "<=": "<=",
    "≤": "<=",
    "≠": "!=",
    "!=": "!="
}


# 手动精准筛选BOSS
async def filter_boss_by_conditions(condition_str):
    async with aiosqlite.connect(DB_PATH) as db:
        parsed_conditions = []
        params = []
        condition_parts = condition_str.strip().split()

        if not condition_parts:
            return "⚠️ 請輸入篩選條件，例如：\n-boss 篩選 屬性=火 等級≥80\n-boss 篩選 所在地=火山 物理防禦>500"

        for part in condition_parts:
            operator = None
            op_pos = -1
            for op in sorted(SUPPORTED_OPERATORS.keys(), key=len, reverse=True):
                if op in part:
                    operator = SUPPORTED_OPERATORS[op]
                    op_pos = part.index(op)
                    break

            if not operator or op_pos == -1:
                return f"❌ 條件「{part}」格式錯誤！\n支持的格式：字段=值/字段>值/字段≥值 等，例如：屬性=火、等級>80"

            field_cn = part[:op_pos].strip()
            value = part[op_pos + len(op):].strip()

            if field_cn not in SUPPORTED_FIELDS:
                supported_fields = "、".join(SUPPORTED_FIELDS.keys())
                return f"❌ 不支持篩選字段「{field_cn}」！\n支持的字段：{supported_fields}"

            field_en = SUPPORTED_FIELDS[field_cn]
            if field_en in ["level", "experience", "physical_defense", "magic_defense",
                            "physical_resistance", "magic_resistance", "physical_inertia",
                            "magic_inertia", "evasion", "dodge_rate", "block_rate", "c_resistance"]:
                try:
                    value = float(value)
                except ValueError:
                    return f"❌ 字段「{field_cn}」的值必須是數字！你輸入的是「{value}」"

            parsed_conditions.append(f"{field_en} {operator} ?")
            params.append(value)

        base_sql = '''
            SELECT 
                location, level, experience, attribute, name,
                physical_inertia, magic_inertia, general_inertia,
                physical_defense, magic_defense, physical_resistance,
                magic_resistance, evasion, c_resistance, dodge_rate,
                block_rate, notes
            FROM bosses
        '''
        if parsed_conditions:
            where_clause = " WHERE " + " AND ".join(parsed_conditions)
            sql = base_sql + where_clause
        else:
            sql = base_sql

        cursor = await db.execute(sql, params)
        rows = await cursor.fetchall()

        if not rows:
            return f"🚫 未找到符合條件「{condition_str}」的BOSS！"

        result = f"🎯 符合條件「{condition_str}」的BOSS查詢結果：\n"
        for idx, row in enumerate(rows, 1):
            (location, level, experience, attribute, name,
             physical_inertia, magic_inertia, general_inertia,
             physical_defense, magic_defense, physical_resistance,
             magic_resistance, evasion, c_resistance, dodge_rate,
             block_rate, notes) = row

            notes = notes if notes is not None else "無補充信息"

            result += f"""
【{idx}. {name}】
📍 所在地：{location}
⚡ 等級：{level} | 📈 經驗值：{experience} | 📌 屬性：{attribute}
🛡️ 防禦相關：
  物理防禦：{physical_defense} | 魔法防禦：{magic_defense}
  物理抗性：{physical_resistance} | 魔法抗性：{magic_resistance}
  C抵抗：{c_resistance}
🚫 慣性相關：
  物理慣性：{physical_inertia} | 魔法慣性：{magic_inertia} | 一般慣性：{general_inertia}
💨 躲避相關：
  迴避：{evasion} | 閃躲率：{dodge_rate} | 阻擋率：{block_rate}
📝 補充：{notes}
"""
        return result.strip()


# 【关键函数】读取全量BOSS数据并结构化（给AI筛选用）
async def get_all_boss_data_structured():
    """读取所有BOSS数据，返回AI易理解的结构化文本"""
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute('''
            SELECT 
                location, level, attribute, name,
                physical_defense, magic_defense, physical_resistance,
                magic_resistance, c_resistance
            FROM bosses
        ''')
        rows = await cursor.fetchall()

        if not rows:
            return "數據庫中暫無BOSS數據"

        structured_data = "以下是所有BOSS的完整數據：\n"
        for idx, row in enumerate(rows, 1):
            (location, level, attribute, name,
             physical_defense, magic_defense, physical_resistance, magic_resistance, c_resistance) = row

#            notes = notes if notes is not None else "無"

            structured_data += f"""
【BOSS{idx}】
名稱：{name}
所在地：{location}
等級：{level}
屬性：{attribute}
物理防禦：{physical_defense}
魔法防禦：{magic_defense}
物理抗性：{physical_resistance}
魔法抗性：{magic_resistance}
C抵抗：{c_resistance}
"""
        return structured_data.strip()