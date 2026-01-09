import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from numba import njit

# use @njit (decorator) to speed up numerical algorithms 

class Body:
    """defines properties of mass in N-body system"""
    def __init__(self, name, mass, position, velocity, color):
        #object identity
        self.name = name
        self.color = color

        #physical properties
        self.mass = float(mass)

        #state vectors
        self.position = np.array(position, dtype = float)
        self.velocity = np.array(velocity, dtype = float)
        self.acceleration = np.zeros(3, dtype = float)

        self.history = []

    def update_history(self):
        """saves current position to history"""
        self.history.append(self.position.copy())


class Universe:
    
    def __init__(self, dt = 0.1, G = 6.6743e-11):
        self.bodies = []
        self.G = G
        self.dt = dt #time-step in seconds
        self.time = 0

    def add_body(self, body):
        self.bodies.append(body) 

    def compute_acceleration(self):
        """calculates the net acceleration due to gravity for every body"""
        for b in self.bodies: 
            b.acceleration = np.zeros(3) #resets acceleration to zero for each new timestep
        
        for i in range(len(self.bodies)):
            for j in range(i + 1, len(self.bodies)):
                b1 = self.bodies[i]
                b2 = self.bodies[j]
                
                r_vector = b2.position - b1.position #vector from 1 to 2
                distance = np.linalg.norm(r_vector) #distance between bodies

                epsilon = 0.001 #softening so F_mag does not go to infinity when r goes to zero
                
                #distance + epsilon prevents division by zero when distance between two bodies is zero
                F_mag = self.G * b1.mass * b2.mass / (distance + epsilon) ** 2 
                
                #b1 is pulled towards b2, which is direction positive r_vector (b2 - b1)
                #b2 is pulled towards b1, which is direction negative r_vector (b2 - b1) 
                b1.acceleration += (F_mag / b1.mass) * (r_vector / (distance + epsilon))
                b2.acceleration -= (F_mag / b2.mass) * (r_vector / (distance + epsilon))

    def step_rk4(self, dt):
        """
        4th order runge-kutta integrator
        state vector y = [x, y, z, vx, vy, vz]
        dy/dt = f(y) = [vx, vy, vz, ax, ay, az]
        
        y_n+1 = y_n + dt/6 * (k1 + 2 * k2 + 2 * k3 + k4)
        """

        # initial conditions
        initial_positions = []
        for b in self.bodies:
            initial_positions.append(b.position.copy())
        
        initial_velocities = []
        for b in self.bodies:
            initial_velocities.append(b.velocity.copy())

        #k1
        k1_r = [] #dr/dt = velocity at current state
        k1_v = [] #dv/dt = acceleration at current state
        self.compute_acceleration()

        for b in self.bodies:
            k1_r.append(b.velocity.copy())
            k1_v.append(b.acceleration.copy())
        
        #k2
        for i in range(len(self.bodies)): #updates state to midpoint (y + dt/2*k1)
            b = self.bodies[i]
            b.position = initial_positions[i] + 0.5 * dt * k1_r[i] #r + dt/2 * v
            b.velocity = initial_velocities[i] + 0.5 * dt * k1_v[i] #v + dt/2 * a
        
        self.compute_acceleration() #calculates accelerations at midpoint state
        
        k2_r = []
        k2_v = []

        for b in self.bodies: 
            k2_r.append(b.velocity.copy())
            k2_v.append(b.acceleration.copy())
        
        #k3
        for i in range(len(self.bodies)):
            b = self.bodies[i]
            b.position = initial_positions[i] + 0.5 * dt * k2_r[i] 
            b.velocity = initial_velocities[i] + 0.5 * dt * k2_v[i]

        self.compute_acceleration()

        k3_r = []
        k3_v = []

        for b in self.bodies: 
            k3_r.append(b.velocity.copy())
            k3_v.append(b.acceleration.copy())
        
        #k4
        for i in range(len(self.bodies)):
            b = self.bodies[i]
            b.position = initial_positions[i] + dt * k3_r[i]
            b.velocity = initial_velocities[i] + dt * k3_v[i]
        
        self.compute_acceleration()
        k4_r = []
        k4_v = []

        for b in self.bodies: 
            k4_r.append(b.velocity.copy())
            k4_v.append(b.acceleration.copy())
        
        #final calculation for y_n+1
        for i in range(len(self.bodies)):
            b = self.bodies[i]
            b.position = initial_positions[i] + dt/6 * (k1_r[i] + 2 * k2_r[i] + 2 * k3_r[i] + k4_r[i])
            b.velocity = initial_positions[i] + dt/6 * (k1_v[i] + 2 * k2_v[i] + 2 * k3_v[i] + k4_v[i])
        
        self.time += dt #advance time by dt

        for b in self.bodies: #saves position to history
            b.update_history()
    
    def step_leapfrog(self, dt):
        """symplectic leapfrog integrator (kick-drift-kick)"""

        #half step velocity update (kick)
        self.compute_acceleration()
        for b in self.bodies:
            b.velocity += 0.5 * dt * b.acceleration 

        #full step position update (drift)
        for b in self.bodies:
            b.position += dt * b.velocity 
        
        #half-step velocity update (kick)
        self.compute_acceleration()
        for b in self.bodies:
            b.velocity += 0.5 * dt * b.acceleration

        self.time += dt
        
        for b in self.bodies: 
            b.update_history()

@njit
def run_simulation(method, steps = 1000):
    """
    simulation function
    method parameter should be either 'leapfrog' or 'rk4'
    """
