from evdd_params import Parameter
from ansys.aedt.core import Desktop



class Sim(Parameter) :

    def __init__(self, non_graphical = False) :
        # super().get_random_variable() # parameter overide
        self.project_name = "script1"
        self.flag = 1
        self.Proj = 0
        self.itr = 0
        self.freq = 105

        self.computer_name = "5950X1"
        self.create_desktop(non_graphical)
        
    def create_desktop(self,non_graphical) :
        
        # open desktop
        self.desktop = Desktop(
            version = "2024.2",
            non_graphical = non_graphical
            )
        self.desktop.disable_autosave()