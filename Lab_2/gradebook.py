# gradebook.py
# Author: Rakshit Bisht
# Date: 2025-12-03
# Title: GradeBook Analyzer CLI

import csv

def calculate_average(marks_dict):
    return sum(marks_dict.values()) / len(marks_dict) if marks_dict else 0

def calculate_median(marks_dict):
    values = sorted(marks_dict.values())
    n = len(values)
    if n == 0:
        return 0
    mid = n // 2
    return (values[mid] if n % 2 != 0 else (values[mid-1] + values[mid]) / 2)

def find_max_score(marks_dict):
    return max(marks_dict.values()) if marks_dict else 0

def find_min_score(marks_dict):
    return min(marks_dict.values()) if marks_dict else 0

def assign_grades(marks_dict):
    grades = {}
    for student, mark in marks_dict.items():
        if mark >= 90:
            grades[student] = "A"
        elif mark >= 80:
            grades[student] = "B"
        elif mark >= 70:
            grades[student] = "C"
        elif mark >= 60:
            grades[student] = "D"
        else:
            grades[student] = "F"
    return grades

def load_csv(path):
    marks = {}
    with open(path, newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) == 2:
                name, score = row
                marks[name] = int(score)
    return marks

def manual_input():
    marks = {}
    print("Enter student data (type 'exit' to stop):")
    while True:
        name = input("Name: ")
        if name.lower() == "exit":
            break
        mark = int(input("Marks: "))
        marks[name] = mark
    return marks

def print_table(marks, grades):
    print("\nName\tMarks\tGrade")
    print("-----------------------------------")
    for student in marks:
        print(f"{student}\t{marks[student]}\t{grades[student]}")
    print()

def main():
    print("WELCOME TO GRADEBOOK ANALYZER")
    while True:
        print("\n1. Manual input")
        print("2. Load from CSV")
        print("3. Exit")
        choice = input("Enter choice: ")

        if choice == "1":
            marks = manual_input()
        elif choice == "2":
            path = input("Enter CSV file path: ")
            marks = load_csv(path)
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid option.")
            continue

        if not marks:
            print("No data found. Try again.")
            continue

        avg = calculate_average(marks)
        med = calculate_median(marks)
        max_s = find_max_score(marks)
        min_s = find_min_score(marks)

        print(f"Average Marks: {avg}")
        print(f"Median Marks: {med}")
        print(f"Highest Score: {max_s}")
        print(f"Lowest Score: {min_s}")

        grades = assign_grades(marks)
        print_table(marks, grades)

        passed_students = [s for s, m in marks.items() if m >= 40]
        failed_students = [s for s, m in marks.items() if m < 40]

        print("Passed:", passed_students)
        print("Failed:", failed_students)

if __name__ == "__main__":
    main()
