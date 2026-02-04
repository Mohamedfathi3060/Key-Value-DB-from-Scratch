from client import KVClient
import time

db = KVClient()
db.set("hussain", "is the best boy at math")
db.set("khaled", "is the best boy forever")
db.set("omar", "is the worst boy forever")

A = time.time()
print(db.get("omar"))
print(db.fulltext_search("math"))
print(db.semantic_search("best boy at math"))
B = time.time()
print(B - A)