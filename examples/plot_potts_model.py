import numpy as np
import matplotlib.pyplot as plt

from time import time

from pystruct.inference import inference_dispatch, compute_energy
from pystruct.inference import get_installed
from pystruct.utils import make_grid_edges

size = 20
n_states = 5

rnd = np.random.RandomState(2)
x = rnd.normal(size=(size, size, n_states))
pairwise = np.eye(n_states)
edges = make_grid_edges(x)
unaries = x.reshape(-1, n_states)

inference_methods = get_installed([('ad3', dict(branch_and_bound=True)), 'ad3',
                                   'qpbo', 'mp', 'lp', 'ogm', 'unary'])


def get_inference_method_name(method):
    dict_repr = ""
    if isinstance(method, tuple):
        method, args = method
        dict_repr = "\n" + ",".join("%s=%s" % (k, v) for k, v in args.items())
    return method + dict_repr


fig, ax = plt.subplots(1, len(inference_methods),
                       figsize=(2 * len(inference_methods), 3))
for a, inference_method in zip(ax, inference_methods):
    start = time()
    y = inference_dispatch(unaries, pairwise, edges,
                           inference_method=inference_method)
    took = time() - start
    a.matshow(y.reshape(size, size))
    energy = compute_energy(unaries, pairwise, edges, y)
    a.set_title("%s\ntime: %.2f\nenergy %.2f" %
                (get_inference_method_name(inference_method), took, energy))
    a.set_xticks(())
    a.set_yticks(())

fig.subplots_adjust(0.05, 0.01, 0.96, 0.76, 0.24, 0.23)
plt.show()
