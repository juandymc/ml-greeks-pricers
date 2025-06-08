class Seed:
    """Simple container for a random seed."""

    def __init__(self):
        self.seed = None

    def set_seed(self, seed):
        self.seed = seed

    def get_seed(self):
        return self.seed


# Global seed used when generating stock paths
path_gen_seed = Seed()
