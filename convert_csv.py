import csv, json

with open("jeopardy/JEOPARDY_CSV.csv", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    reader.fieldnames = [name.strip() for name in reader.fieldnames]
    rows = [{"question": r["Question"]} for r in reader]

with open("jeopardy/jeopardy.json", "w", encoding="utf-8") as f:
    json.dump(rows, f)

print(len(rows), "questions written")