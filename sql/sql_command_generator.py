import requests
import json
from llm_sql import call_llm

prompt = """
You are an expert SQL query generator specialized in SQLite.

Your task is to convert the user's natural language request into a valid SQLite SQL query
using ONLY the provided table schema and column names.

STRICT RULES:
1. Use ONLY column names exactly as provided.
2. Do NOT invent, infer, guess, or rename column names.
3. Do NOT replace spaces with underscores.
4. If a column name contains spaces, it MUST be wrapped in double quotes.
5. Do NOT assume the meaning of data unless a column explicitly exists.
6. The output must be valid SQLite SQL (use LIMIT, not TOP).

IRRELEVANT OR INVALID REQUESTS:
If the user's request refers to information that is NOT present in the schema,
or cannot be answered using the available columns,
do NOT generate SQL.

Instead, respond with:
"I'm sorry — this request cannot be answered using the available data."

OUTPUT RULES:
- If valid → output ONLY the SQL query.
- If invalid → output ONLY the polite message.
- No explanations. No markdown.

table name:
employees
columns and their types and semantic meanings:
Employee_Name:
  semantic_type: datetime
  duckdb_type: VARCHAR
  sample_values: ['Employee_1', 'Employee_2', 'Employee_3', 'Employee_4', 'Employee_5']

Department:
  semantic_type: categorical
  duckdb_type: VARCHAR
  sample_values: ['HR', 'R&D', 'HR', 'R&D', 'Logistics']
  allowed_values: ['HR', 'R&D', 'Logistics', 'Operations', 'IT', 'Finance', 'Marketing']

Designation:
  semantic_type: categorical
  duckdb_type: VARCHAR
  sample_values: ['Consultant', 'Engineer', 'Consultant', 'Analyst', 'Consultant']
  allowed_values: ['Consultant', 'Engineer', 'Analyst', 'Lead', 'Manager', 'Senior Engineer']        

Location:
  semantic_type: categorical
  duckdb_type: VARCHAR
  sample_values: ['Bangalore', 'Delhi', 'Mumbai', 'Mumbai', 'Mumbai']
  allowed_values: ['Bangalore', 'Delhi', 'Mumbai', 'Chennai', 'Kolkata', 'Hyderabad', 'Pune']        

Project_Code:
  semantic_type: datetime
  duckdb_type: VARCHAR
  sample_values: ['PRJ315', 'PRJ370', 'PRJ277', 'PRJ437', 'PRJ768']

Salary_INR:
  semantic_type: identifier
  duckdb_type: BIGINT
  sample_values: ['1614198', '1552031', '549060', '366105', '1733204']

Joining_Date:
  semantic_type: datetime
  duckdb_type: DATE
  sample_values: ['2015-12-25', '2020-12-04', '2015-11-09', '2020-07-17', '2019-01-22']

Performance_Rating:
  semantic_type: categorical
  duckdb_type: VARCHAR
  sample_values: ['D', 'A', 'B', 'A', 'C']
  allowed_values: ['D', 'A', 'B', 'C']

Status:
  semantic_type: categorical
  duckdb_type: VARCHAR
  sample_values: ['On Leave', 'Active', 'Active', 'Active', 'On Leave']
  allowed_values: ['On Leave', 'Active', 'Resigned']

Review:
  semantic_type: categorical
  duckdb_type: VARCHAR
  sample_values: ['Needs improvement in communication but technically sound.', 'Outstanding performer with leadership potential.', 'Reliable employee but should improve documentation habits.', 'Shows initiative and adapts quickly to new challenges.', 'Performance is satisfactory but lacks proactiveness.']
  allowed_values: ['Needs improvement in communication but technically sound.', 'Outstanding performer with leadership potential.', 'Reliable employee but should improve documentation habits.', 'Shows initiative and adapts quickly to new challenges.', 'Performance is satisfactory but lacks proactiveness.', 'Great attitude and strong commitment to company goals.', 'Consistently meets expectations and delivers tasks on time.', 'Excellent problem-solving skills and strong team collaboration.']


Request:
"display all status"
"""


sql_query = call_llm(
    prompt=prompt,
    system_prompt="Return only valid SQLite SQL. No explanation.",
    model="llama3:8b",
    temperature=0.1
)

print(sql_query)