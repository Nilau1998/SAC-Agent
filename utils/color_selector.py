import random


class ColorSelector:
    def __init__(self):
        self.colors = ['\x1b[34m', '\x1b[32m',
                       '\x1b[96m', '\x1b[95m', '\x1b[91m', '\x1b[97m']

    def get_color(self):
        if len(self.colors) > 1:
            color = random.choice(self.colors)
            index = self.colors.index(color)
            self.colors.pop(index)
            return color
        else:
            self.refill_colors()
            return self.colors[-1]

    def refill_colors(self):
        self.colors = ['\x1b[34m', '\x1b[32m',
                       '\x1b[96m', '\x1b[95m', '\x1b[91m', '\x1b[97m']


if __name__ == '__main__':
    color_selector = ColorSelector()
    for _ in range(20):
        print("Hey" + color_selector.get_color())
