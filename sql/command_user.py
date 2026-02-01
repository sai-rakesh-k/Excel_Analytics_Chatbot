import duckdb

con = duckdb.connect("customercare.duckdb")

# con.execute("""
# CREATE TABLE customer AS
# SELECT * FROM read_csv_auto('excel_sample\Customer_care.csv');
# """)

query = """
SELECT *
FROM customer;

"""
query2 = """
SELECT AVG("CSAT Score") FROM customer WHERE "Channel" = 'Call-Center' AND State = 'Texas';
 """

results = con.execute(query2).fetchall()
print(results)