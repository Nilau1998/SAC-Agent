import matplotlib.pyplot as plt


class Integrator:
    def __init__(self, initial_value=0, lower_limit=float('-inf'), upper_limit=float('inf')):
        self.dt = 0.1
        self.counter = 0
        self.time = [self.counter]
        self.initial_value = initial_value
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.input_signal = []
        self.output_signal = [initial_value]

    def integrate_signal(self, input_signal):
        """
        Integrates a given signal based on a preset dt.
        """
        self.input_signal.append(input_signal)
        if self.counter == 0:
            integrated_value = self.initial_value
        else:
            integrated_value = input_signal * self.dt + \
                self.output_signal[-1]

        if integrated_value <= self.lower_limit:
            self.output_signal.append(self.lower_limit)
        elif integrated_value >= self.upper_limit:
            self.output_signal.append(self.upper_limit)
        else:
            self.output_signal.append(integrated_value)
        self.counter += 1
        self.time.append(self.counter)

        return integrated_value

    def scope(self, file_name='scope'):
        """
        Creates a plot of all integrated input signals. If for example the integrator integrated a bunch of velocities, then calling scope on it will return all positions.
        """
        plt.plot(self.time, self.output_signal)
        plt.savefig(f"{file_name}.png")
        plt.close()


if __name__ == '__main__':
    a_integrator = Integrator()
    v_integrator = Integrator()

    t, t_max, dt = 0, 10, 0.1
    while t <= t_max:
        v = a_integrator.integrate_signal(9.81)
        v_integrator.integrate_signal(v)
        t += dt
    v_integrator.scope()
