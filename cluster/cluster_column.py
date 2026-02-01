import duckdb
import pandas as pd
import os
from llm_cluster import call_llm


def get_cluster_texts_from_question(
    question: str,
    file_path: str
):
    """
    End-to-end function:
    Question -> column selection (LLM) -> SQL extract text -> list[str]
    """

    # =========================
    # 1. Ask LLM for column name
    # =========================
    prompt = f"""
You are a column selection engine for clustering tasks.

Available columns:
Employee_Name
Department
Designation
Location
Project_Code
Salary_INR
Joining_Date
Performance_Rating
Status
Review

Rules:
- Select ONLY text-based columns
- If question refers to reviews / feedback / issues -> Review
- Output ONLY column name(s)
- Comma-separated if multiple
- No explanations

Question:
{question}
"""

    column_string = call_llm(
        prompt=prompt,
        system_prompt="Return only column name(s).",
        model="llama3:8b",
        temperature=0.0
    ).strip()

    if not column_string:
        raise ValueError("LLM did not return any column")

    columns = [c.strip() for c in column_string.split(",")]

    # =========================
    # 2. Load file via Pandas (ingestion only)
    # =========================
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # =========================
    # 3. Register in DuckDB
    # =========================
    con = duckdb.connect(":memory:")
    con.register("data", df)

    # =========================
    # 4. Validate columns
    # =========================
    for col in columns:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found. Available columns: {list(df.columns)}"
            )

    # =========================
    # 5. SQL extraction
    # =========================
    sql_cols = ", ".join([f'"{c}"' for c in columns])

    query = f"""
    SELECT {sql_cols}
    FROM data
    WHERE { " OR ".join([f'"{c}" IS NOT NULL' for c in columns]) }
    """

    rows = con.execute(query).fetchall()

    # =========================
    # 6. Format output (same as before)
    # =========================
    if len(columns) == 1:
        return [str(r[0]) for r in rows]

    return [" | ".join(map(str, r)) for r in rows]
texts = get_cluster_texts_from_question(
    question="what is most popular review",
    file_path="clean_unique_reviews_1000_rows.xlsx"
)
print(texts[:5])
