from abc import ABC, abstractmethod

class Camera(ABC):
    @abstractmethod
    def take_picture(self):
        pass #No implementation given
    
class DSLR(Camera):
    def take_picture(self):
        print("DSLR Capturing Photo")

class SmartphoneCamera(Camera):
    def take_picture(self):
        print("Smartphone is taking Photo")


dslr=DSLR()
phone=SmartphoneCamera()

dslr.take_picture()
phone.take_picture()
