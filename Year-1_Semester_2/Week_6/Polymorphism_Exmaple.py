## __ means that it is a private attribute (-)

class Camera:

    
    def take_picture(self):
        print(f"Took a picture with the Normal Camera")

class DSLRCamera(Camera):

      def take_picture(self):
        print(f"Taking High Res photo using DSLR")

class ActionCamera(Camera):

      def take_picture(self):
        print(f"Taking Action Shot!")

              

cameras=[Camera(),DSLRCamera(),ActionCamera()]
for cams in cameras:
    cams.take_picture()






