# Basing Dynamics off of Neuro-Genetic Truck Backer Upper paper

import numpy as np
import random
from math import degrees, radians

class TruckBackerUpper:    
    def __init__(self, trailer_length=14, cab_length=6, x_bounds=[0,200], y_bounds=[-100, 100]):
        self.l_t = trailer_length
        self.l_c = cab_length
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.reset_truck()


    # Puts the Trailer in a random vertical position and orientation (from TD-paper)
    def reset_truck(self):
        # Position Variables
        self.x = 160 
        self.y = np.random.uniform(-10, 10)
        # Angle Variables
        self.theta_t = np.random.uniform(-1.5, 1.5)
        self.theta_c = 0.0


    def step(self, u):
        # Intermediate Variables for computation
        a = 3 * np.cos(u)
        b = a * np.cos(self.theta_c)
        # Updating State Variables
        self.x += -1 * b * np.cos(self.theta_t)
        self.y += -1 * b * np.sin(self.theta_t)
        self.theta_t += -1 * np.arcsin(a * np.sin(self.theta_c) / self.l_t)
        self.theta_c += np.arcsin(3 * np.sin(u) / (self.l_c + self.l_t))
        # Returning flags for termination (success, or fail)
        terminated_goal = self.at_goal()
        terminated_fail = not(self.valid())
        return terminated_goal, terminated_fail

    # Checking if truck is in a valid position
    def valid(self):
        return ((not self.is_jackknifed()) and self.valid_location() and self.valid_angles())

    # Checking if truck has reached the goal (using relaxed goal from the paper)
    def at_goal(self):
        return (np.sqrt(self.x**2 + self.y**2) <= 3 and self.theta_t <= 0.1)

    # Some utility functions
    def is_jackknifed(self):
        return(self.theta_c > np.pi/2)
    # This is the one function, I need to check  
    def valid_location(self):
        return (self.x <= self.x_bounds[1] and self.x >= self.x_bounds[0]) and \
              (self.y >= self.y_bounds[0] and self.y <= self.y_bounds[1])
    # Checking for valid angles
    def valid_angles(self):
        return self.theta_t <= 4*np.pi

if __name__ == '__main__':
    print("Hello World")
    z = TruckBackerUpper()
    for i in range(0, 100):
        f, g = z.step(-np.pi/2 + np.random.normal(0, np.pi/2))
        print(z.x, z.y, z.theta_c,z.theta_t)
        print(f,g)