from control_theory.control_blocks import Integrator, Scope
import numpy as np


if __name__ == '__main__':
    # Constants
    car_max_a = 10
    car_angle = 0

    # Define intergrators
    a_integrator = Integrator(upper_limit=10)
    v_x_integrator = Integrator()
    v_y_integrator = Integrator()

    # Define scope
    scope = Scope(['s_x', 's_y'])

    # Timey wimey stuff
    t, t_max, dt = 0, 500, 0.1

    total_a = car_max_a
    while t <= t_max:
        car_angle += 0.01
        v = a_integrator.integrate_signal(total_a)

        v_x = v * np.cos(car_angle)
        v_y = v * np.sin(car_angle)

        s_x = v_x_integrator.integrate_signal(v_x)
        s_y = v_y_integrator.integrate_signal(v_y)
        scope.record_signals([s_x, s_y])
        t += dt
    scope.create_time_scope('toy_car_circles')
