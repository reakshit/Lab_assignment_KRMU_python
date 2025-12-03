# library_inventory_single.py
# Combined single-file Library Inventory Manager
import json, logging, sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

class Book:
    def __init__(self, title, author, isbn, status="available"):
        self.title = title
        self.author = author
        self.isbn = str(isbn)
        self.status = status
    def __str__(self):
        return f"{self.title} by {self.author} (ISBN: {self.isbn}) - {self.status}"
    def to_dict(self):
        return {"title": self.title, "author": self.author, "isbn": self.isbn, "status": self.status}
    def issue(self):
        if self.status != "available":
            raise ValueError("Book already issued.")
        self.status = "issued"
    def return_book(self):
        if self.status == "available":
            raise ValueError("Book is not issued.")
        self.status = "available"
    def is_available(self):
        return self.status == "available"

class LibraryInventory:
    def __init__(self, storage_path="books.json"):
        self.storage_path = Path(storage_path)
        self.books = []
        self.load()
    def add_book(self, book):
        if self.search_by_isbn(book.isbn):
            raise ValueError("ISBN already exists.")
        self.books.append(book)
        self.save()
    def search_by_title(self, title):
        q = title.lower()
        return [b for b in self.books if q in b.title.lower()]
    def search_by_isbn(self, isbn):
        for b in self.books: 
            if b.isbn == str(isbn): return b
        return None
    def display_all(self):
        return [b.to_dict() for b in self.books]
    def issue_book(self, isbn):
        b = self.search_by_isbn(isbn)
        if not b: raise ValueError("Not found.")
        b.issue()
        self.save()
    def return_book(self, isbn):
        b = self.search_by_isbn(isbn)
        if not b: raise ValueError("Not found.")
        b.return_book()
        self.save()
    def save(self):
        data = [b.to_dict() for b in self.books]
        self.storage_path.write_text(json.dumps(data, indent=2))
    def load(self):
        if not self.storage_path.exists():
            self.books=[]
            return
        try:
            raw = self.storage_path.read_text()
            data = json.loads(raw)
            self.books = [Book(**d) for d in data]
        except:
            self.books = []

def print_menu():
    print("\nLibrary Inventory Manager")
    print("1. Add Book")
    print("2. Issue Book")
    print("3. Return Book")
    print("4. View All Books")
    print("5. Search by Title")
    print("6. Search by ISBN")
    print("7. Exit")

def input_nonempty(prompt):
    while True:
        v = input(prompt).strip()
        if v: return v
        print("Cannot be empty.")

def main():
    inv = LibraryInventory()
    while True:
        print_menu()
        ch = input("Enter choice: ").strip()
        try:
            if ch=="1":
                t=input_nonempty("Title: "); a=input_nonempty("Author: "); i=input_nonempty("ISBN: ")
                inv.add_book(Book(t,a,i)); print("Book added.")
            elif ch=="2":
                i=input_nonempty("ISBN: "); inv.issue_book(i); print("Issued.")
            elif ch=="3":
                i=input_nonempty("ISBN: "); inv.return_book(i); print("Returned.")
            elif ch=="4":
                books=inv.display_all()
                print("\nTitle | Author | ISBN | Status")
                print("-"*50)
                for b in books:
                    print(f"{b['title']} | {b['author']} | {b['isbn']} | {b['status']}")
            elif ch=="5":
                q=input_nonempty("Title: ")
                for r in inv.search_by_title(q): print(r)
            elif ch=="6":
                i=input_nonempty("ISBN: ")
                b=inv.search_by_isbn(i); print(b if b else "Not found.")
            elif ch=="7":
                print("Goodbye!"); sys.exit(0)
            else:
                print("Invalid.")
        except Exception as e:
            print("Error:", e)

if __name__=="__main__":
    main()
