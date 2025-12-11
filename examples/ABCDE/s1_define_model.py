import os, sys
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE)

repo_root = os.path.abspath(os.path.join(BASE, "../.."))
if repo_root not in sys.path:
    sys.path.append(repo_root)

from tbnpy import cpt, variable
import numpy as np

import c, oc, e

device = ('cuda' if os.environ.get('USE_CUDA', '0') == '1' else 'cpu')

def define_variables():
    varis = {}
    varis['A'] = variable.Variable(name='A', values=[0.0, 0.5])
    varis['B'] = variable.Variable(name='B', values=[0.5, 1.0, 1.5])
    varis['C'] = variable.Variable(name='C', values='unitless')  # Continuous
    varis['OC'] = variable.Variable(name='OC', values='unitless')  # Continuous
    varis['D'] = variable.Variable(name='D', values=[0, 1])      # Binary
    varis['E'] = variable.Variable(name='E', values='unitless')  # Continuous

    return varis

def define_probs(varis, device='cpu'):
    probs = {}

    probs['A'] = cpt.Cpt(childs=[varis['A']], C=np.array([[0], [1]]), p=np.array([0.1, 0.9]), device=device)
    probs['B'] = cpt.Cpt(childs=[varis['B']], C=np.array([[0], [1], [2]]), p=np.array([0.05, 0.15, 0.80]), device=device)

    probs['C'] = c.C(childs=[varis['C']], parents=[varis['A'], varis['B']], device=device)
    probs['OC'] = oc.OC(childs=[varis['OC']], parents=[varis['C']], device=device)

    probs['D'] = cpt.Cpt(childs=[varis['D']], C=np.array([[0], [1]]), p=np.array([0.4, 0.6]), device=device)

    probs['E'] = e.E(childs=[varis['E']], parents=[varis['C'], varis['D']], device=device)

    return probs

if __name__ == '__main__':
    varis = define_variables()
    probs = define_probs(varis, device=device)

