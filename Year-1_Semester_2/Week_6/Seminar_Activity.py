class Vehicle():
    def __init__(self, registration, seats, wheels):
        self.__registration=registration
        self.__seats=seats
        self.__wheels=wheels
    
    def get_details(self):
        print(f"Reg: {self.__registration},Number of seats:{self.__seats},Wheels: {self.__wheels}")


class Car(Vehicle):
    def __init__(self, registration, seats, wheels, doors):
        super().__init__(self, registration, seats, wheels)
        self.doors=doors
    def get_details(self):
        print(f"Reg: {self.__registration},Number of seats:{self.__seats},Wheels: {self.__wheels}, Doors: {self.doors}")
    

class Motorcycle(Vehicle):
    def __init__(self, registration, seats, wheels, gear):
        super().__init__(self, registration, seats, wheels)
        self.__gear=gear
    def get_details(self):
        print(f"Reg: {self.__registration},Number of seats:{self.__seats},Wheels: {self.__wheels}, Gear: {self.__gear}")
    

class Truck(Car):
    def __init__(self, registration, seats, wheels, doors, towing_capacity):
        super().__init__(registration, seats, wheels, doors)
        self.__towing_capacity=towing_capacity
    pass