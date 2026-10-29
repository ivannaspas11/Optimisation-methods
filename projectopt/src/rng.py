import math

MASK64 = (1 << 64) - 1
MULT   = 6364136223846793005
INCR   = 1442695040888963407


class Rng:
    def __init__(self, seed):
        self.state = seed & MASK64

    def next_u64(self):
        self.state = (self.state * MULT + INCR) & MASK64
        return self.state

    def uniform(self):
        return (self.next_u64() >> 11) * (1.0 / 9007199254740992.0)

    def normal(self):
        u1 = self.uniform()
        if u1 < 1e-300:
            u1 = 1e-300
        u2 = self.uniform()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)