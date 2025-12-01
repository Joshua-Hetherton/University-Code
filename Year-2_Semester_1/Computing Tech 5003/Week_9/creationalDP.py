class Animal ():
    def speak(self):
        raise NotImplementedError("Subclasses must implement this method")
    
class Dog(Animal):
    def speak(self):
        return "Woof!"
    

if __name__ == "__main__":
    my_dog = Dog()
    print(my_dog.speak())  # Output: Woof!