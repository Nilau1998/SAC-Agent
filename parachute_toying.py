from environment.integrator import Integrator
import numpy as np

"""
Example usage of the integrator to model a parachute guy jumping from h_0 and opening his parachute at h_1. Included is aerodrag and so on.
"""

if __name__ == '__main__':
    h_0 = 3000
    h_1 = 1500
    A_s = 0.5
    A_FS = 25
    m = 85
    c_w = 1.3
    p = 1.2
    g = 9.81
    t, t_max, dt = 0, 500, 0.1

    a_integrator = Integrator()
    v_integrator = Integrator(initial_value=h_0)

    total_a = 0
    tmp = 0
    while t <= t_max:
        total_a -= g
        v = a_integrator.integrate_signal(total_a)
        s = v_integrator.integrate_signal(v)
        print(v_integrator.counter, s)

        # Stop if ground reached
        if s < 0:
            break

        # Wind resistance
        if s < h_1:
            F_w = v**2 * 0.5 * p * c_w * A_FS
        else:
            F_w = v**2 * 0.5 * p * c_w * A_s
        # Turn force into acceleration
        total_a = F_w / m

        t += dt
    v_integrator.scope('s_plot')
    a_integrator.scope('v_plot')
