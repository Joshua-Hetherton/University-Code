## __ means that it is a private attribute (-)

class Camera:
    def __init__(self,brand,model, resolution):
        self.__brand = brand
        self.__model= model
        self.__resolution= resolution
    
    def get_details(self):
        return f"Camera:{self.__brand} {self.__model} {self.__resolution} MP"
    
    def set_resolution(self, new_resolution):
        if (new_resolution>0):
            self.__resolution= new_resolution
        else:
            print("Invalid Resolution given")
    def set_model(self,new_model):
        if(new_model !=""):
            self.__model=new_model




my_camera=Camera("Cannon", "2.1", 30)

print(my_camera.get_details())

my_camera.set_resolution(45)
print(my_camera.get_details())

my_camera.set_model("Test")
print(my_camera.get_details())



