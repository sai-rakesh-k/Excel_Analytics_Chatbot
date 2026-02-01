import duckdb
import re
from collections import Counter

# ===================== CONFIG =====================

DB_PATH = "customercare.duckdb"
TABLE_NAME = "customer"
SAMPLE_SIZE = 50

# ===================== CONNECT =====================

con = duckdb.connect(DB_PATH)

# ===================== GET SCHEMA =====================

def get_schema(con, table):
    rows = con.execute(f"DESCRIBE {table};").fetchall()
    return [{"column": r[0], "type": r[1]} for r in rows]

# ===================== SAMPLE COLUMN VALUES =====================

def sample_column(con, table, column, limit=SAMPLE_SIZE):
    q = f'''
    SELECT "{column}"
    FROM {table}
    WHERE "{column}" IS NOT NULL
    LIMIT {limit}
    '''
    return [str(r[0]) for r in con.execute(q).fetchall()]

# ===================== HELPER FUNCTIONS =====================

def is_numeric_string(val):
    return bool(re.fullmatch(r"-?\d+(\.\d+)?", val.strip()))

def numeric_ratio(values):
    if not values:
        return 0.0
    return sum(is_numeric_string(v) for v in values) / len(values)

def unique_ratio(values):
    if not values:
        return 0.0
    return len(set(values)) / len(values)

def looks_like_datetime(values):
    """
    TRUE datetime columns have numbers in ALL values
    (dates, timestamps, times all contain digits)
    """
    if not values:
        return False
    for v in values:
        if not any(ch.isdigit() for ch in v):
            return False
    return True

# ===================== SEMANTIC INFERENCE =====================

def infer_semantic(column, dtype, samples):
    joined = " ".join(samples).lower()
    u_ratio = unique_ratio(samples)
    n_ratio = numeric_ratio(samples)

    # 1️⃣ Boolean
    if set(joined.split()).issubset({"yes", "no", "true", "false", "0", "1"}):
        return "boolean"

    # 2️⃣ Identifier
    if u_ratio > 0.95 and ("int" in dtype.lower() or n_ratio > 0.9):
        return "identifier"

    # 3️⃣ Numeric measure
    if "int" in dtype.lower() or "double" in dtype.lower() or "float" in dtype.lower():
        if u_ratio > 0.2:
            return "measure"
        else:
            return "countable"

    # 4️⃣ Datetime (VALUE-BASED ONLY ✅ FIX)
    if looks_like_datetime(samples):
        return "datetime"

    # 5️⃣ Status-like text (SLA, states, stages)
    if any(word in joined for word in [
        "above", "below", "within", "pending", "completed",
        "failed", "success", "open", "closed", "sla"
    ]):
        return "status_text"

    # 6️⃣ Categorical enum
    if u_ratio < 0.2:
        return "categorical"

    # 7️⃣ Free text
    return "free_text"

# ===================== BUILD COLUMN DEFINITIONS =====================

def build_column_definitions(con, table):
    schema = get_schema(con, table)
    definitions = {}

    for col in schema:
        name = col["column"]
        dtype = col["type"]

        samples = sample_column(con, table, name)
        semantic = infer_semantic(name, dtype, samples)

        definitions[name] = {
            "semantic_type": semantic,
            "duckdb_type": dtype,
            "sample_values": samples[:5]
        }

        # Extra metadata
        if semantic in {"categorical", "status_text"}:
            definitions[name]["allowed_values"] = list(
                Counter(samples).keys()
            )[:10]

        if semantic == "measure":
            nums = [
                float(v) for v in samples
                if is_numeric_string(v)
            ]
            if nums:
                definitions[name]["range"] = [min(nums), max(nums)]

    return definitions

# ===================== RUN =====================

if __name__ == "__main__":
    column_definitions = build_column_definitions(con, TABLE_NAME)

    print("\n===== AUTO COLUMN DEFINITIONS (FIXED) =====\n")
    for col, meta in column_definitions.items():
        print(f"{col}:")
        for k, v in meta.items():
            print(f"  {k}: {v}")
        print()