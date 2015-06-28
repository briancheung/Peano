class Pop(object):
    def __init__(self):
        self.params = []

    def apply(self, v):
        raise NotImplementedError(
            str(type(self)) + " does not implement apply.")

    def __call__(self, v):
        return self.apply(v)