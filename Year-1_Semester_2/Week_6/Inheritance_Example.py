## __ means that it is a private attribute (-)

class Camera:
    def __init__(self,brand,model, resolution):
        self.brand = brand
        self.model= model
        self.resolution= resolution
    
    def take_picture(self):
        print(f"{self.brand} {self.model} took a picture at {self.resolution} MP")

class DSLRCamera(Camera):
        def __init__(self,brand,model, resolution, lens_type,zoom):
                super().__init__(brand, model, resolution)
                self.lens_type=lens_type
                self.zoom_level=zoom

        def change_lens(self, new_lens):
              self.lens_type=new_lens

              print(f"Lens changed to {self.lens_type}")

        def zoom_in(self, level):
              self.zoom_level+=level
              print(f"zooming in by {level}. Zoom is now {self.zoom_level}")

        def zoom_out(self, level):
              self.zoom_level-=level
              print(f"zooming out by {level}. Zoom is now {self.zoom_level}")
              

dslr=DSLRCamera("Cannon","DSLR",24, "20mm",0)
dslr.take_picture()
dslr.change_lens("50mm")
dslr.zoom_in(3)
dslr.zoom_out(5)






