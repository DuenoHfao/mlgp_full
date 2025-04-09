class Parent:
    def __init__(self, name):
        self.name = name

class Teacher:
    def __init__(self, subject):
        self.subject = subject

class Child(Parent, Teacher):
    def __init__(self, name, subject):
        Parent.__init__(self, name)
        Teacher.__init__(self, subject)

    def display(self):
        print(f"Name: {self.name}, Subject: {self.subject}")

child_obj = Child("John", "Math")
child_obj.display()