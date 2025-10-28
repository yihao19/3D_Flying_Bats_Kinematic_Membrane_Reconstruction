import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def system(state, t, m, k, c, base_func):
    x, v = state
    xb = base_func(t)
    xbdot = (base_func(t + 1e-6) - xb) / 1e-6
    # spring extension relative to base: (x - xb)
    a = (-k*(x - xb) - c*(v - xbdot)) / m
    return [v, a]

# PARAMETERS
m = 0.000001       # mass [kg]
k = 0.0000001     # spring constant [N/m]
c = 1       # damping coefficient [N·s/m]

# Example base motion: moves at constant speed, or sinusoidal:
def base_motion(t):
    # constant velocity: vb = 0.5 m/s
    return 10 * t
    # or use sinusoidal:
    # return 0.1 * np.sin(2 * np.pi * 1.0 * t)

# INITIAL CONDITIONS
x0 = 0.0      # initial mass displacement
v0 = 0.0      # initial mass velocity
state0 = [x0, v0]

# SIMULATION TIME
t = np.linspace(0, 10, 10001)

# INTEGRATE
sol = odeint(system, state0, t, args=(m, k, c, base_motion))
x = sol[:, 0]
v = sol[:, 1]
xb = base_motion(t)

# PLOTTING
plt.figure(figsize=(10, 6))
plt.plot(t, x, label='mass displacement x(t)')
plt.plot(t, xb, '--', label='base motion xb(t)')
plt.xlabel('Time [s]')
plt.ylabel('Position [m]')
plt.legend()
plt.title('Mass–spring–damper with Base Motion')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(t, v, label='mass velocity v(t)')
plt.xlabel('Time [s]')
plt.ylabel('Velocity [m/s]')
plt.title('Mass Velocity')
plt.grid(True)
plt.show()