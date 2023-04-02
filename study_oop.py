
# Four Principles of OOP
# 1. Inheritance
# 2. Polymorphism
# 3. Encapsulation
# 4. Abstraction

# position, name, age, level, salary
se1 = ["Software Engineer", "Max", 20, "Junior", 5000]
se2 = ["Software Engineer", "Lisa", 25, "Senior", 7000]

# class
class SoftwareEngineer:

    # class attribute
    alias = "Keyboard Magician"

    def __init__(self, name, age, level, salary):
        # instance attributes
        self.name = name
        self.age = age
        self.level = level
        self.salary = salary

    # instance method
    def code(self):
        print(f"{self.name} is writing code...")
        
    def code_in_language(self, language):
        print(f"{self.name} is writing code in {language}...")

    # dunder method
    def __str__(self):
        information = f"name = {self.name}, age = {self.age}, level = {self.level}, salary = {self.salary}"
        return information
        
    def __eq__(self, other):
        return self.name == other.name and self.age == other.age

    # static method
    @staticmethod
    def entry_salary(age):
        if age < 25:
            return 5000
        if age < 30:
            return 7000
        return 9000


# instance
se1 = SoftwareEngineer("Max", 20, "Junior", 5000)
se2 = SoftwareEngineer("Lisa", 20, "Senior", 7000)
se3 = SoftwareEngineer("Lisa", 20, "Senior", 7000)

se1.code()
se2.code()
se1.code_in_language("Python")
se2.code_in_language("C++")

print(se2)
print(se2 == se3)

print(se1.entry_salary(24))
print(SoftwareEngineer.entry_salary(27))

print('The end')
print('I am Martin')

print("Good bye!")
