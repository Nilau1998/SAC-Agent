from control_theory.control_blocks import Integrator, Scope

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
    t, t_max, dt = 0, 500, 0.01

    a_integrator = Integrator()
    v_integrator = Integrator(initial_value=h_0)
    scope = Scope(labels=['Position y'])

    total_a = 0
    while t <= t_max:
        total_a -= g
        v = a_integrator.integrate_signal(total_a)
        s = v_integrator.integrate_signal(v)

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
        scope.record_signals([s])
        t += dt
    scope.create_time_scope(file_name='python_parachute_s_plot')
