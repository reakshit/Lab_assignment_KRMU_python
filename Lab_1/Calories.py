# ----------------------------------------------------------
# Name: Rakshit Bisht
# Date: 08 November 2025
# Project Title: Daily Calorie Tracker CLI
# Roll no. : 2501730357 -------------------------------------------------


print("===============================================")
print("     Welcome to the Daily Calorie Tracker CLI  ")
print("===============================================")
print("This tool helps you log your meals, track calories,")
print("and compare your total intake with your daily goal.\n")


meal_names = []
calories = []

num_meals = int(input("How many meals did you have today? "))

for i in range(num_meals):
    meal = input(f"Enter the name of meal #{i+1}: ")
    cal = float(input(f"Enter calories for {meal}: "))
    meal_names.append(meal)
    calories.append(cal)


total_calories = sum(calories)
average_calories = total_calories / len(calories)
daily_limit = float(input("\nEnter your daily calorie limit: "))


if total_calories > daily_limit:
    status = "Warning: You exceeded your daily calorie limit!"
else:
    status = "Good job! You are within your calorie limit."


print("\n===============================================")
print("           DAILY CALORIE SUMMARY")
print("===============================================")
print("Meal Name\tCalories")
print("-----------------------------------------------")
for i in range(len(meal_names)):
    print(f"{meal_names[i]:<15}\t{calories[i]:>6.2f}")

print("-----------------------------------------------")
print(f"Total:\t\t{total_calories:.2f}")
print(f"Average:\t{average_calories:.2f}")
print(status)
print("===============================================\n")


save = input("Would you like to save this session to a file? (yes/no): ").lower()

if save == "yes":
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("calorie_log.txt", "w") as file:
        file.write("Daily Calorie Tracker Log\n")
        file.write(f"Date & Time: {timestamp}\n")
        file.write("-----------------------------------------------\n")
        for i in range(len(meal_names)):
            file.write(f"{meal_names[i]:<15}\t{calories[i]:>6.2f}\n")
        file.write("-----------------------------------------------\n")
        file.write(f"Total: {total_calories:.2f}\n")
        file.write(f"Average: {average_calories:.2f}\n")
        file.write(status + "\n")
    print("Session saved successfully as 'calorie_log.txt'.")
else:
    print("Session not saved. Thank you for using the tracker!")

print("\nProgram Ended. Stay healthy and balanced! ")

