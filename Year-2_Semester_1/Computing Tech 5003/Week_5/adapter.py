class Adapter:
    def __init__(self, obj, adapted_methods):
        self.obj = obj
        self.__dict__.update(adapted_methods)

    def __getattr__(self, attr):
        return getattr(self.obj, attr)
    
# Example usage:
class Dog:
    def bark(self):
        return "Woof!"

class Cat:
    def meow(self):
        return "Meow!"

dog = Dog()
adapted_dog = Adapter(dog, {"make_sound": dog.bark})

print(adapted_dog.make_sound())  # Output: Woof!

cat = Cat()
adapted_cat = Adapter(cat, {"make_sound": cat.meow})

print(adapted_cat.make_sound())  # Output: Meow!