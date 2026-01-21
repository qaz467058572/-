# -*- coding: utf-8 -*-
import aiosqlite
import asyncio
import numpy as np
import os
import sys
import gc
import json
import re
import requests
from typing import List, Dict, Any, Optional, Tuple
import threading

# ===================== 全局配置 =====================
OPENAI_API_BASE = "http://198.18.0.1:1234/v1/responses"
OPENAI_API_KEY = ""
OPENAI_MODEL_NAME = "bubu1123"
OPENAI_TIMEOUT = 30
OPENAI_MAX_TOKENS = 64
OPENAI_TEMPERATURE = 0.0

# M3E模型配置（优先本地，本地不存在则使用默认模型）
M3E_MODEL_PATH = "./m3e-small" if os.path.exists("./m3e-small") else "all-MiniLM-L6-v2"

# 数据库与向量配置
LOCAL_DB_IMPORT_PATH = "./boss_database.db"
TARGET_DB_PATH = "bosses.db"
VECTOR_STORAGE_PATH = "./boss_vectors.npy"
VECTOR_META_PATH = "./boss_vectors_meta.json"
BATCH_SIZE = 50
SHOW_PROGRESS = True
SIMILARITY_THRESHOLD = 0.8

# 本地数据库字段映射
FIELD_MAPPING = {
    "boss_name": "name",
    "location": "location",
    "level": "level",
    "attribute": "attribute",
    "physical_resistance": "physical_resistance",
    "magic_resistance": "magic_resistance",
    "physical_defense": "physical_defense",  # 物理防禦
    "magic_defense": "magic_defense",  # 魔法防禦
    "physical_inertia": "physical_inertia",  # 物理慣性
    "magic_inertia": "magic_inertia",  # 魔法慣性
    "general_inertia": "general_inertia",  # 一般慣性
    "c_resistance": "c_resistance",  # C抵抗
    "notes": "notes"
}

# ===================== 全局状态管理 =====================
# 初始化状态标记（确保仅初始化一次）
_SYSTEM_INITIALIZED = False
# 初始化锁（防止并发调用时重复初始化）
_INIT_LOCK = threading.Lock()


# ===================== 模型管理器 =====================
class ModelManager:
    def __init__(self):
        self.m3e_model = None
        self.boss_vectors: Optional[np.ndarray] = np.array([])
        self.boss_vector_meta: Dict[str, int] = {}
        self.m3e_dim: int = 0
        self.vectors_loaded = False

    def init_m3e_model(self) -> bool:
        """
        初始化M3E向量模型（仅首次调用生效）
        返回：是否初始化成功
        """
        if self.m3e_model is not None:
            print("✅ M3E模型已初始化，跳过重复初始化")
            return True

        print("=== 初始化M3E向量模型 ===")
        try:
            from sentence_transformers import SentenceTransformer
            self.m3e_model = SentenceTransformer(M3E_MODEL_PATH)
            self.m3e_dim = len(self.m3e_model.encode("测试文本"))
            print(f"✅ M3E加载完成 | 向量维度：{self.m3e_dim}")
            return True
        except Exception as e:
            print(f"❌ M3E加载失败：{e}")
            return False

    def _load_existing_vectors(self) -> bool:
        """
        加载已存在的向量库（避免重复生成）
        返回：是否加载成功
        """
        if self.vectors_loaded:
            return True

        try:
            if os.path.exists(VECTOR_STORAGE_PATH) and os.path.exists(VECTOR_META_PATH):
                self.boss_vectors = np.load(VECTOR_STORAGE_PATH)
                with open(VECTOR_META_PATH, "r", encoding="utf-8") as f:
                    self.boss_vector_meta = json.load(f)
                self.vectors_loaded = True
                print(f"✅ 加载已有向量库 | 向量数量：{len(self.boss_vectors)}")
                return True
            else:
                print("⚠️ 未找到已存在的向量库，需要重新生成")
                return False
        except Exception as e:
            print(f"❌ 加载向量库失败：{e}")
            return False

    def _clean_old_vectors(self):
        """清理旧的向量库文件"""
        print("=== 清理旧向量库数据 ===")
        try:
            if os.path.exists(VECTOR_STORAGE_PATH):
                os.remove(VECTOR_STORAGE_PATH)
                print(f"✅ 删除旧向量文件：{VECTOR_STORAGE_PATH}")
            if os.path.exists(VECTOR_META_PATH):
                os.remove(VECTOR_META_PATH)
                print(f"✅ 删除旧向量元数据：{VECTOR_META_PATH}")
        except Exception as e:
            print(f"⚠️ 删除旧向量库失败：{e}")

        self.boss_vectors = np.array([])
        self.boss_vector_meta = {}
        self.vectors_loaded = False

    def _save_vectors(self):
        """保存向量库"""
        if self.boss_vectors.size > 0:
            try:
                np.save(VECTOR_STORAGE_PATH, self.boss_vectors)
                with open(VECTOR_META_PATH, "w", encoding="utf-8") as f:
                    json.dump(self.boss_vector_meta, f, ensure_ascii=False)
                self.vectors_loaded = True
                print(f"✅ 向量库已保存到：{VECTOR_STORAGE_PATH}")
            except Exception as e:
                print(f"❌ 保存向量库失败：{e}")

    def cleanup(self):
        """释放资源"""
        self.m3e_model = None
        self.boss_vectors = None
        self.boss_vector_meta = {}
        self.vectors_loaded = False
        gc.collect()
        print("✅ 模型资源释放完成")

    async def encode_text(self, text: str) -> np.ndarray:
        """
        异步编码文本为向量
        参数：text - 要编码的文本
        返回：文本对应的向量
        """
        loop = asyncio.get_running_loop()
        vec = await loop.run_in_executor(
            None,
            self.m3e_model.encode,
            text
        )
        return vec


# ===================== 全局实例 =====================
model_manager = ModelManager()


# ===================== 工具函数 =====================
def safe_float_convert(v: Any, default: float = 0.0) -> float:
    """安全转换为浮点数"""
    try:
        return float(v) if v is not None else default
    except:
        return default


def safe_str_convert(v: Any, default: str = "未知") -> str:
    """安全转换为字符串"""
    return str(v).strip() if v and str(v).strip() else default


def cosine_similarity(query_vec: np.ndarray, corpus_vecs: np.ndarray) -> np.ndarray:
    """
    计算余弦相似度
    参数：
        query_vec - 查询向量
        corpus_vecs - 语料库向量矩阵
    返回：相似度数组
    """
    query_norm = np.linalg.norm(query_vec)
    corpus_norm = np.linalg.norm(corpus_vecs, axis=1)

    if query_norm == 0:
        return np.zeros(len(corpus_vecs))
    corpus_norm = np.where(corpus_norm == 0, 1e-8, corpus_norm)

    query_normalized = query_vec / query_norm
    corpus_normalized = corpus_vecs / corpus_norm[:, np.newaxis]

    return np.dot(corpus_normalized, query_normalized)


def _print_progress(current: int, total: int, prefix: str = ""):
    """打印进度条"""
    if not SHOW_PROGRESS:
        return
    percent = (current / total) * 100
    bar = "█" * int(percent // 5) + "-" * (20 - int(percent // 5))
    print(f"\r{prefix} |{bar}| {current}/{total} ({percent:.1f}%)", end="")
    if current == total:
        print()


# ===================== 核心1：关键词提取 =====================
def openai_api_get_keywords(raw_query: str) -> str:
    """提取JSON结构化关键词"""
    prompt = f"""
    你是BOSS检索关键词生成器，严格遵守以下规则输出：
    1. 仅提取与BOSS【名称、所在地、属性、等级】相关的核心词汇
    2. 必须输出标准JSON格式，结构为：{{"keywords": ["词1", "词2", ...]}}
    3. JSON中仅包含keywords字段，无其他字段
    4. 禁止输出JSON以外的任何内容
    5. 示例：
       用户查询："找找空達的龍族BOSS"
       输出：{{"keywords": ["空達", "龍族"]}}

    用户查询：{raw_query}
    """

    payload = {
        "input": prompt.strip(),
        "model": OPENAI_MODEL_NAME,
        "temperature": OPENAI_TEMPERATURE,
        "max_tokens": OPENAI_MAX_TOKENS,
        "stream": False
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            url=OPENAI_API_BASE,
            headers=headers,
            json=payload,
            timeout=OPENAI_TIMEOUT
        )
        response.raise_for_status()
        result = response.json()

        print(f"📜 API原始返回：{json.dumps(result, ensure_ascii=False, indent=2)}")
        keywords = raw_query

        if "output" in result and isinstance(result["output"], list):
            for output_item in result["output"]:
                if output_item.get("type") == "message":
                    content_list = output_item.get("content", [])
                    for content_item in content_list:
                        if content_item.get("type") == "output_text":
                            text_content = content_item.get("text", "").strip()
                            if text_content:
                                try:
                                    clean_text = re.sub(r"^\s+|\s+$", "", text_content)
                                    json_data = json.loads(clean_text)
                                    keyword_list = json_data.get("keywords", [])
                                    if isinstance(keyword_list, list) and len(keyword_list) > 0:
                                        keywords = " ".join(keyword_list)
                                    break
                                except:
                                    keywords = text_content
                                    break

        keywords = re.sub(r"[\n\t\r，。！？；：\"\'()（）【】]", " ", keywords)
        keywords = re.sub(r"\s+", " ", keywords).strip()
        print(f"✅ 最终提取关键词：{keywords}")

        return keywords
    except Exception as e:
        print(f"❌ API调用失败：{str(e)}")
        return raw_query


# ===================== 核心2：数据库初始化 =====================
async def _init_database_if_needed() -> bool:
    """
    初始化数据库（仅在数据库不存在或为空时执行）
    返回：是否初始化成功
    """
    # 检查数据库是否已存在且有数据
    try:
        if os.path.exists(TARGET_DB_PATH):
            async with aiosqlite.connect(TARGET_DB_PATH) as db:
                cursor = await db.execute("SELECT COUNT(*) FROM bosses")
                count = (await cursor.fetchone())[0]
                if count > 0:
                    print(f"✅ 数据库已存在且有{count}条数据，跳过数据库初始化")
                    return True
    except Exception as e:
        print(f"⚠️ 检查数据库失败：{e}")

    print("=== 初始化数据库 ===")
    try:
        # 删除旧数据库文件（如果存在但无数据）
        if os.path.exists(TARGET_DB_PATH):
            os.remove(TARGET_DB_PATH)
            print(f"✅ 删除无效数据库文件：{TARGET_DB_PATH}")

        # 重建表结构
        async with aiosqlite.connect(TARGET_DB_PATH) as db:
            await db.execute("DROP TABLE IF EXISTS bosses")
            await db.execute("""
                CREATE TABLE IF NOT EXISTS bosses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    boss_name TEXT NOT NULL,
                    location TEXT,
                    level REAL,
                    attribute TEXT,
                    physical_resistance REAL,
                    magic_resistance REAL,
                    physical_defense REAL,
                    magic_defense REAL,
                    physical_inertia REAL,
                    magic_inertia REAL,
                    general_inertia REAL,
                    c_resistance REAL,
                    notes TEXT
                )
            """)
            await db.commit()

        # 导入数据
        imported_count = await _import_local_database()
        if imported_count == 0:
            print("⚠️ 无可用的BOSS数据，数据库初始化完成但无数据")
        else:
            print("✅ 数据库初始化完成")

        return True
    except Exception as e:
        print(f"❌ 数据库初始化失败：{e}")
        return False


async def _import_local_database() -> int:
    """
    从本地数据库导入BOSS数据
    返回：导入的数据条数
    """
    if not os.path.exists(LOCAL_DB_IMPORT_PATH):
        print(f"⚠️ 本地数据库文件不存在：{LOCAL_DB_IMPORT_PATH}")
        return 0

    try:
        async with aiosqlite.connect(LOCAL_DB_IMPORT_PATH) as local_db:
            cursor = await local_db.execute("PRAGMA table_info(bosses)")
            local_fields = [row[1] for row in await cursor.fetchall()]

            valid_fields = []
            target_fields = []
            for target_field, local_field in FIELD_MAPPING.items():
                target_fields.append(target_field)
                if local_field in local_fields:
                    valid_fields.append(local_field)
                else:
                    print(f"⚠️ 本地数据库缺少字段：{local_field}，将使用默认值")
                    valid_fields.append("''")

            select_sql = f"SELECT {', '.join(valid_fields)} FROM bosses"
            cursor = await local_db.execute(select_sql)
            local_data = await cursor.fetchall()
            if len(local_data) == 0:
                print(f"⚠️ 本地数据库中无BOSS数据")
                return 0

            async with aiosqlite.connect(TARGET_DB_PATH) as target_db:
                await target_db.execute("DELETE FROM bosses")
                insert_sql = f"""
                    INSERT INTO bosses ({', '.join(target_fields)})
                    VALUES ({', '.join(['?' for _ in target_fields])})
                """
                await target_db.executemany(insert_sql, local_data)
                await target_db.commit()

                print(f"✅ 成功从本地数据库导入 {len(local_data)} 条BOSS数据")
                return len(local_data)

    except Exception as e:
        print(f"❌ 导入本地数据库失败：{e}")
        return 0


# ===================== 核心3：向量同步 =====================
async def _sync_boss_vectors_if_needed() -> Dict[str, Any]:
    """
    同步BOSS向量库（仅在向量库不存在时执行）
    返回：同步结果字典
    """
    # 尝试加载已有向量
    if model_manager._load_existing_vectors():
        return {
            "status": "success",
            "processed": len(model_manager.boss_vectors),
            "total": len(model_manager.boss_vectors),
            "msg": "使用已有向量库，无需重新生成"
        }

    # 先初始化数据库
    db_init_ok = await _init_database_if_needed()
    if not db_init_ok:
        return {"status": "failed", "processed": 0, "total": 0, "msg": "数据库初始化失败"}

    # 获取BOSS数据
    async with aiosqlite.connect(TARGET_DB_PATH) as db:
        cursor = await db.execute("SELECT id, boss_name, location, level, attribute FROM bosses")
        rows = await cursor.fetchall()
        total = len(rows)
        if total == 0:
            return {"status": "failed", "processed": 0, "total": 0, "msg": "无BOSS数据"}

    # 清理旧向量（确保生成新向量）
    model_manager._clean_old_vectors()

    print("=== 开始生成BOSS向量库 ===")
    batch_vectors = []
    processed = 0

    for row in rows:
        try:
            boss_id = row[0]
            boss_name = row[1]
            location = row[2]
            attribute = row[4]

            # 生成向量文本
            boss_text = f"BOSS名称：{boss_name} 所在地：{location} 属性：{attribute}"
            vec = await model_manager.encode_text(boss_text)
            batch_vectors.append(vec)
            model_manager.boss_vector_meta[str(boss_id)] = processed
            processed += 1
            _print_progress(processed, total, "📤 向量同步进度")

            # 批量保存
            if len(batch_vectors) >= BATCH_SIZE or processed == total:
                vec_arr = np.array(batch_vectors)
                if model_manager.boss_vectors.size == 0:
                    model_manager.boss_vectors = vec_arr
                else:
                    model_manager.boss_vectors = np.vstack([model_manager.boss_vectors, vec_arr])
                batch_vectors = []
        except Exception as e:
            print(f"\n❌ 处理BOSS {row[1]} 失败：{e}")
            continue

    # 保存最终向量
    model_manager._save_vectors()

    return {
        "status": "success",
        "processed": processed,
        "total": total,
        "msg": f"成功生成 {processed}/{total} 条BOSS向量"
    }


# ===================== 核心4：BOSS检索 =====================
async def _do_search(keywords: str, limit: int = 5, threshold: float = SIMILARITY_THRESHOLD) -> List[Dict]:
    """
    实际执行检索逻辑（内部函数）
    """
    # 拆分关键词
    keyword_list = list(set([k.strip() for k in keywords.split() if k.strip()]))
    if not keyword_list:
        return []

    # 第一步：精准文本匹配
    async with aiosqlite.connect(TARGET_DB_PATH) as db:
        where_conditions = []
        params = []
        for keyword in keyword_list:
            where_conditions.append("(boss_name LIKE ? OR location LIKE ? OR attribute LIKE ? OR notes LIKE ?)")
            params.extend([f"%{keyword}%", f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"])

        where_sql = " AND ".join(where_conditions)
        query_sql = f"""
            SELECT id, boss_name, location, level, attribute, physical_resistance, magic_resistance,
            physical_defense, magic_defense, physical_inertia, magic_inertia, general_inertia, c_resistance, notes 
            FROM bosses 
            WHERE {where_sql}
        """

        cursor = await db.execute(query_sql, params)
        text_match_results = await cursor.fetchall()

    # 处理精准匹配结果
    if text_match_results:
        weighted_results = []
        for row in text_match_results:
            boss_name = row[1]
            location = row[2]
            attribute = row[4] if row[4] else ""
            notes = row[13] if row[13] else ""
            total_text = f"{boss_name}{location}{attribute}{notes}"

            # 计算匹配权重
            match_count = 0
            for keyword in keyword_list:
                if keyword in total_text:
                    match_count += 1

            weighted_results.append({
                "row": row,
                "match_count": match_count
            })

        # 排序并构建结果
        weighted_results.sort(key=lambda x: x["match_count"], reverse=True)
        results = []
        for idx, item in enumerate(weighted_results[:limit]):
            row = item["row"]
            results.append({
                "排名": idx + 1,
                "名稱": row[1],
                "所在地": row[2],
                "等級": row[3],
                "屬性": row[4],
                "物理抗性": row[5],
                "魔法抗性": row[6],
                "物理防禦": row[7],
                "魔法防禦": row[8],
                "物理慣性": row[9],
                "魔法慣性": row[10],
                "一般慣性": row[11],
                "C抵抗": row[12],
                "相似度": round(1.0 - (0.1 * (len(keyword_list) - item["match_count"])), 3),
                "備註": row[13]
            })
        return results

    # 第二步：向量检索兜底
    if model_manager.boss_vectors.size == 0:
        print("❌ 向量库为空，无法进行向量检索")
        return []

    # 生成查询向量
    query_text = f"BOSS名称：{keywords} 所在地：{keywords} 属性：{keywords}"
    query_vec = await model_manager.encode_text(query_text)

    # 计算相似度
    similarities = cosine_similarity(query_vec, model_manager.boss_vectors)

    # 筛选结果
    valid_mask = similarities >= threshold
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) == 0:
        return []

    sorted_idx = valid_idx[np.argsort(-similarities[valid_idx])][:limit]

    # 查询数据库获取详细信息
    async with aiosqlite.connect(TARGET_DB_PATH) as db:
        boss_ids = [list(model_manager.boss_vector_meta.keys())[int(i)] for i in sorted_idx]
        placeholders = ",".join(["?"] * len(boss_ids))
        cursor = await db.execute(f"""
            SELECT id, boss_name, location, level, attribute, physical_resistance, magic_resistance,
            physical_defense, magic_defense, physical_inertia, magic_inertia, general_inertia, c_resistance, notes 
            FROM bosses WHERE id IN ({placeholders})
        """, boss_ids)
        boss_rows = await cursor.fetchall()

    # 构建结果
    boss_map = {str(row[0]): row for row in boss_rows}
    results = []
    for idx, vec_idx in enumerate(sorted_idx):
        boss_id = list(model_manager.boss_vector_meta.keys())[int(vec_idx)]
        boss = boss_map.get(boss_id)
        if boss:
            results.append({
                "排名": idx + 1,
                "名稱": boss[1],
                "所在地": boss[2],
                "等級": boss[3],
                "屬性": boss[4],
                "物理抗性": boss[5],
                "魔法抗性": boss[6],
                "物理防禦": boss[7],
                "魔法防禦": boss[8],
                "物理慣性": boss[9],
                "魔法慣性": boss[10],
                "一般慣性": boss[11],
                "C抵抗": boss[12],
                "相似度": round(float(similarities[vec_idx]), 3),
                "備註": boss[13]
            })
    return results


# ===================== 对外公开的核心接口 =====================
async def init_system(force_reinit: bool = False) -> bool:
    """
    初始化整个BOSS检索系统（仅首次调用生效，可强制重新初始化）
    参数：
        force_reinit - 是否强制重新初始化（默认False）
    返回：是否初始化成功
    """
    global _SYSTEM_INITIALIZED

    # 加锁防止并发初始化
    with _INIT_LOCK:
        # 如果已初始化且不强制重新初始化，直接返回成功
        if _SYSTEM_INITIALIZED and not force_reinit:
            print("✅ 系统已初始化，跳过重复初始化")
            return True

        print("=== 初始化BOSS检索系统 ===")

        # 1. 初始化模型
        model_ok = model_manager.init_m3e_model()
        if not model_ok:
            return False

        # 2. 同步向量库（仅在需要时）
        sync_result = await _sync_boss_vectors_if_needed()
        if sync_result["status"] != "success":
            print(f"❌ 向量库同步失败：{sync_result['msg']}")
            return False

        # 标记为已初始化
        _SYSTEM_INITIALIZED = True
        print("✅ 系统初始化完成")
        return True


async def search_boss(raw_query: str, limit: int = 5) -> List[Dict]:
    """
    对外公开的BOSS检索接口（自动检查并初始化系统）
    参数：
        raw_query - 用户原始查询字符串
        limit - 返回结果数量
    返回：检索结果列表
    """
    # 自动初始化（仅首次调用）
    await init_system()

    # 提取关键词
    keywords = openai_api_get_keywords(raw_query)
    if not keywords:
        keywords = raw_query

    # 执行检索
    results = await _do_search(keywords, limit, SIMILARITY_THRESHOLD)
    return results


async def cleanup_system():
    """
    清理系统资源（重置初始化状态）
    """
    global _SYSTEM_INITIALIZED

    with _INIT_LOCK:
        model_manager.cleanup()
        _SYSTEM_INITIALIZED = False
        print("✅ 系统资源已清理，初始化状态已重置")


def is_system_initialized() -> bool:
    """
    检查系统是否已初始化
    返回：初始化状态
    """
    return _SYSTEM_INITIALIZED

async def search_by_keywords(keywords: str, limit: int = 5, threshold: float = SIMILARITY_THRESHOLD) -> List[Dict]:
    """
    兼容旧版本的检索接口（对外暴露）
    """
    # 确保系统已初始化
    await init_system()
    # 调用内部检索逻辑
    return await _do_search(keywords, limit, threshold)