import numpy as np

import math

class Parameter() :

    def __init__(self) :
        pass

    def _random_choice(self, X) :
        return round(np.random.choice( np.arange( X[0] , X[1]+X[2] , X[2]) ),X[3])

    def get_random_variable(self) :

        # ===============
        # Range setup
        # ===============

        w1_range = [20, 200, 1, 0]
        l1_leg_range = [2, 15, 0.1, 1]
        l1_top_range = [0.5, 2, 0.1, 1]
        l2_range = [5, 30, 0.1, 1] # under, upper, resolution

        h1_range = [0.1,3, 0.01, 2]

        Tx_turns_range = [5, 5, 1, 0]

        # Tx_height_range = [0.105, 0.175, 0.035, 3] 
        Tx_preg_range = [0.01, 0.2, 0.01, 2] 
        
        Rx_preg_range = [0.01, 0.2, 0.01, 2]
        Rx_height_range = [0.1, 1, 0.1, 1]

        g1_range = [0, 0, 0.01, 2]
        g2_range = [0.1, 3, 0.01, 2]

        l1_center_range = [1,25,1,0]

        Tx_space_x_range = [0.1, 5, 0.1, 1]
        Tx_space_y_range = [0.1, 5, 0.1, 1]
        Rx_space_x_range = [0.1, 5, 0.1, 1]
        Rx_space_y_range = [0.1, 5, 0.1, 1]


        Tx_layer_space_x_range = [0.2, 5, 0.1, 1]
        Tx_layer_space_y_range = [0.2, 5, 0.1, 1]
        Rx_layer_space_x_range = [0.2, 5, 0.1, 1]
        Rx_layer_space_y_range = [0.2, 5, 0.1, 1]

        # Tx_width_range = [0.5,3,0.1,1]
        Rx_width_range = [4,20,0.1,1]

        wire_diameter_range = [0.05,0.08,0.01,2]
        strand_number_range = [7,100,2,0]

        # ===============
        # Get values
        # ===============

        self.w1 = self._random_choice(w1_range)
        self.l1_leg= self._random_choice(l1_leg_range)

        self.l1_top = self._random_choice(l1_top_range)
        self.l2 = self._random_choice(l2_range)

        self.h1 = self._random_choice(h1_range)


        self.Tx_turns = self._random_choice(Tx_turns_range)

        # self.Tx_height = self._random_choice(Tx_height_range)
        self.Tx_preg = self._random_choice(Tx_preg_range)

        self.Rx_space_y = self._random_choice(Rx_space_y_range)
        self.Rx_preg = self._random_choice(Rx_preg_range)

        self.Rx_height = self._random_choice(Rx_height_range)
        self.Rx_space_x = self._random_choice(Rx_space_x_range)

        self.g1 = 0
        self.g2 = self._random_choice(g2_range)
        self.l1_center = self._random_choice(l1_center_range)


        self.Tx_space_x = self._random_choice(Tx_space_x_range)
        self.Tx_space_y = self._random_choice(Tx_space_y_range)
        self.Rx_space_x = self._random_choice(Rx_space_x_range)
        self.Rx_space_y = self._random_choice(Rx_space_y_range)


        self.Tx_layer_space_x = self._random_choice(Tx_layer_space_x_range)
        self.Tx_layer_space_y = self._random_choice(Tx_layer_space_y_range)
        self.Rx_layer_space_x = self._random_choice(Rx_layer_space_x_range)
        self.Rx_layer_space_y = self._random_choice(Rx_layer_space_y_range)

        # self.Tx_width = self._random_choice(Tx_width_range)
        self.Rx_width = self._random_choice(Rx_width_range)

        self.wire_diameter = self._random_choice(wire_diameter_range)
        self.strand_number = self._random_choice(strand_number_range)


        #######################################
        # 상단부까지가 csv 입력 
        #######################################
        self.Tx_width = round(math.sqrt(self.wire_diameter**2*self.strand_number*2),2)
        self.Tx_height = self.Tx_width

        Tx_max = (self.Tx_layer_space_y + self.Tx_width)*math.ceil(self.Tx_turns) + self.Tx_space_y - self.Tx_layer_space_y
        Rx_max = (self.Rx_width+self.Rx_space_y) 

        while(True) :
            if self.Tx_height*2 + self.Tx_preg + self.Rx_height*2 + self.Rx_preg * 2 >= self.h1:

                # self.Tx_height = self._random_choice(Tx_height_range)
                self.wire_diameter = self._random_choice(wire_diameter_range)
                self.strand_number = self._random_choice(strand_number_range)
                self.Tx_width = round(math.sqrt(self.wire_diameter**2*self.strand_number*2),2)
                self.Tx_height = self.Tx_width

                self.Tx_preg = self._random_choice(Tx_preg_range)
                self.Rx_height = self._random_choice(Rx_height_range)
                self.Rx_preg = self._random_choice(Rx_preg_range)
                self.h1 = self._random_choice(h1_range)

            elif  Tx_max + Rx_max >= self.l2 :
                self.Tx_layer_space_y = self._random_choice(Tx_layer_space_y_range)
                # self.Tx_width = self._random_choice(Tx_width_range)
                self.wire_diameter = self._random_choice(wire_diameter_range)
                self.strand_number = self._random_choice(strand_number_range)
                self.Tx_width = round(math.sqrt(self.wire_diameter**2*self.strand_number*2),2)
                self.Tx_height = self.Tx_width
                self.l2 = self._random_choice(l2_range)
                self.Rx_width = self._random_choice(Rx_width_range)
                self.Rx_space_y = self._random_choice(Rx_space_y_range)
                
                Tx_max = (self.Tx_layer_space_y + self.Tx_width)*math.ceil(self.Tx_turns) + self.Tx_space_y - self.Tx_layer_space_y
                Rx_max = (self.Rx_width+self.Rx_space_y) 

            elif  self.g2 >= self.h1 :
                self.g2 = self._random_choice(g2_range)
                self.h1 = self._random_choice(h1_range)

            else :
                break