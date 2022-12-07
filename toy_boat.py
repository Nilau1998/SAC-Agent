from control_theory.control_blocks import Integrator, Scope
import numpy as np
import math


if __name__ == '__main__':
    # Constants
    boat_propeller_rpm = 0
    boat_rudder_angle = 0
    boat_m = 1_000

    rho = 1  # Water density

    # Timey wimey stuff
    t, t_max, dt = 0, 300, 0.1

    # Intergrators
    a_integrator = Integrator(initial_value=0.1)
    v_integrator = Integrator()

    # Scope
    scope = Scope(['a', 'v', 'F_t/10', 'F_r/10'])

    def equation_of_motions_1(v, n):
        """
        Balance of longitudinal forces
        """
        F_t = calculation_of_thrust(v, n)
        F_r = resistance(v)

        a = (F_t + F_r) / boat_m

        scope.record_signals([a, v, F_t/10, F_r/10])
        return a

    def calculation_of_thrust(v, n):
        # Thrust constants
        w = 0.3  # Wake friction (0.2 - 0.45)
        d = 1.5  # Ship propeller diameter

        # Propeller advance speed in ships wake
        v_w = v * (1 - w)

        J_v = 0
        if n != 0:
            J_v = v_w / (n * d)

        # Usually KT is a loopup table for the thrust coeficient
        KT = np.sin(J_v)

        # Thrust equation
        T = KT * rho * d**4 * n**2 * (0.7)  # Thrust

        return T

    def resistance(v):
        # Resistance constans
        c_r = 0.31  # Long cylinder
        AB = 20  # Boat area of attack

        # Resistance/Drag equation
        R = -c_r * rho * 0.5 * v**2 * AB

        return R

    # Main loop
    a = 0
    while t <= t_max:
        if boat_propeller_rpm < 20:
            boat_propeller_rpm += 0.1

        v = a_integrator.integrate_signal(a)
        s = v_integrator.integrate_signal(v)

        a = equation_of_motions_1(v, boat_propeller_rpm)
        t += dt
    scope.create_time_scope('toy_boat')
