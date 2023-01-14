import numpy as np
if __name__ == '__main__':
    num_models = np.array(list(range(64)))

    print(np.array_split(num_models, np.arange(5, len(num_models), 5)))
