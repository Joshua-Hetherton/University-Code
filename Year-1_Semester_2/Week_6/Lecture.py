class Camera:
    def __init__(self,brand,model, resolution):
        self.brand = brand
        self.model= model
        self.resolution= resolution
        self.battery=100


    def show_brand(self):
        print(f"This camera is a {self.brand}")

    def turn_on(self):
        self.is_on=True
        print(f"Camer is on")

    def turn_off(self):
        self.is_on=False
        print(f"Camera is off")

    def take_picture(self,):
        
        if(self.is_on):

            if(self.battery>0):

                self.battery-=1
                print(f"Taking picture")

            else:
                print("Not enough Battery")
        else:
            print(f"Error: Camera is Off")

    def zoom(self, level):
        print(f"Zooming in by {level}")

    def record_video(self,duration):
        if(self.is_on):
            if(self.battery>duration):

                count=0
                while duration>0:
                    
                    self.battery-=1
                    
                    count+=1
                print(f"Camera has taken video")
            else:
                print(f"Not enough Battery")
        else:
            print(f"Error: Camera is Off")

    def battery_level(self):
        print(f"{self.battery}")

        

my_camera=Camera("Cannon", "2.1", 30)
print(f"""------
{my_camera.brand}
{my_camera.model}
{my_camera.resolution}
------""")
my_camera.turn_on()
my_camera.zoom(20)
my_camera.take_picture()
my_camera.record_video(60)

my_camera.turn_off()


