import json
import math

import pandas
import matplotlib.pyplot as plt

def round_up(n, decimals=2):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier

def round_down(n, decimals=2):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier) / multiplier

def compute_latest_bounds_from_pieces(data, term):
    term_lbs = [point[3][term][0] for point in data[-1]["pieces"]]
    term_ubs = [point[3][term][1] for point in data[-1]["pieces"]]

    return round_down(min(term_lbs)), round_up(max(term_ubs))


with open("softplus_residual_greedy_input_branching.json", "r") as fp:
    data = json.load(fp)

import pdb
pdb.set_trace()

n_problems = [point["i"] for point in data]
lbs = [point["lb"] for point in data]
ubs = [point["ub"] for point in data]

n_problems = [1] + n_problems
lbs = [-2957264.3024832355] + lbs
ubs = [3018986.0696737412] + ubs

plt.plot(n_problems, [-lb for lb in lbs], c="r")
plt.plot(n_problems, len(n_problems)*[0.56], "r--")

plt.plot(n_problems, ubs, c="g")
plt.plot(n_problems, len(n_problems)*[0.42], "g--")

plt.yscale("log")
plt.ylabel("(mirrored) lower and upper bounds")
plt.xlabel("# bounds computed solved")

plt.tight_layout()
plt.show()
