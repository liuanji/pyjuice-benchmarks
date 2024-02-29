


class Flatten():
    def __init__(self):
        pass

    def __call__(self, x):
        return x.reshape(-1)