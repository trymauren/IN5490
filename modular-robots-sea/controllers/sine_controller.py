import math

class SineController:
    def __init__(self, rng):
        self.rng = rng

        # Setup initial controller values
        self.amplitude = rng.choice([0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]) #rng.random()
        self.amplitude_scale = 1.0

        self.period = rng.choice([0.0625, 0.125, 0.25, 0.5, 1.0]) #rng.random()
        self.period_scale = 1.0/32.0

        self.phase = rng.choice([0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]) #rng.random()
        self.phase_scale = 2.0*math.pi

        self.springyness = rng.random()

        # Save previous action for modulation
        self.previous_action = 0.0

        # Not used but added for code compatability with other controllers
        self.previous_phase = 0.0

    def mutate(self, sigma, probability, rng):
        if rng.random() < probability:
            self.amplitude = rng.choice([0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0])
        if rng.random() < probability:
            self.period = rng.choice([0.125, 0.25, 0.5, 1.0])
        if rng.random() < probability:
            self.phase = rng.choice([0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0])
        if rng.random() < probability:
            self.springyness = self.gaussian_mutation(self.springyness, sigma)

    def gaussian_mutation(self, parameter, sigma):
        new_parameter = parameter
        new_parameter += self.rng.normal(0, sigma)
        while new_parameter >= 1.0 or new_parameter < 0.0:
            if new_parameter >= 1.0:
                new_parameter = 2.0 - new_parameter
            elif new_parameter < 0.0:
                new_parameter = -new_parameter
        return new_parameter

    def get_action(self, time, parent_phase_old=None, observation=None):
        true_amplitude = self.amplitude*self.amplitude_scale
        true_period = self.period*self.period_scale
        true_phase = self.phase*self.phase_scale

        controller_value = true_amplitude * math.sin(time*true_period + true_phase)
        if controller_value > self.amplitude_scale:
            controller_value = self.amplitude_scale
        elif controller_value < -self.amplitude_scale:
            controller_value = -self.amplitude_scale

        return controller_value
