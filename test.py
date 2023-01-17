import colorama

if __name__ == '__main__':

    colors = list(vars(colorama.Fore).values())
    print(colors)
    for color in colors:
        print('hey' + '\x1b[37m')
