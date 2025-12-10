from tbnpy import variable, cpt

import numpy as np
import json

import os
BASE = os.path.dirname(os.path.abspath(__file__))

def define_variables():

    eq_sources = json.load(open(os.path.join(BASE, 'eq_sources.json')))
    m1_values = list(eq_sources['s1']['magnitude_probs'].keys())
    m2_values = list(eq_sources['s2']['magnitude_probs'].keys())

    varis = {}

    varis['m1'] = variable.Variable(name='m1', values=m1_values) # magnitude of source 1
    varis['m2'] = variable.Variable(name='m2', values=m2_values) # magnitude of source 2

    varis['e1'] = variable.Variable(name='e1', values='unit_free') # Variance in pga for edge 1
    varis['e2'] = variable.Variable(name='e2', values='unit_free') # Variance in pga for edge 2
    varis['e3'] = variable.Variable(name='e3', values='unit_free') # Variance in pga for edge 3
    varis['e4'] = variable.Variable(name='e4', values='unit_free') # Variance in pga for edge 4
    varis['e5'] = variable.Variable(name='e5', values='unit_free') # Variance in pga for edge 5
    
    varis['pga1'] = variable.Variable(name='pga1', values='g') # continuous variable (unit: g), for edge 1
    varis['pga2'] = variable.Variable(name='pga2', values='g') # continuous variable (unit: g), for edge 2
    varis['pga3'] = variable.Variable(name='pga3', values='g') # continuous variable (unit: g), for edge 3
    varis['pga4'] = variable.Variable(name='pga4', values='g') # continuous variable (unit: g), for edge 4
    varis['pga5'] = variable.Variable(name='pga5', values='g') # continuous variable (unit: g), for edge 5

    varis['o_pga1'] = variable.Variable(name='o_pga1', values='g') # continuous variable (unit: g), observed pga for pga1
    varis['o_pga2'] = variable.Variable(name='o_pga2', values='g') # continuous variable (unit: g), observed pga for pga2
    varis['o_pga3'] = variable.Variable(name='o_pga3', values='g') # continuous variable (unit: g), observed pga for pga3
    varis['o_pga4'] = variable.Variable(name='o_pga4', values='g') # continuous variable (unit: g), observed pga for pga4
    varis['o_pga5'] = variable.Variable(name='o_pga5', values='g') # continuous variable (unit: g), observed pga for pga5

    varis['x1'] = variable.Variable(name='x1', values=['fail', 'survive']) # functionality of edge 1
    varis['x2'] = variable.Variable(name='x2', values=['fail', 'survive']) # functionality of edge 2
    varis['x3'] = variable.Variable(name='x3', values=['fail', 'survive']) # functionality of edge 3
    varis['x4'] = variable.Variable(name='x4', values=['fail', 'survive']) # functionality of edge 4
    varis['x5'] = variable.Variable(name='x5', values=['fail', 'survive']) # functionality of edge 5

    varis['s2'] = variable.Variable(name='s2', values=['fail', 'survive']) # connectivity between node 1 (critical hub) to node 2
    varis['s3'] = variable.Variable(name='s3', values=['fail', 'survive']) # connectivity between node 1 (critical hub) to node 3
    varis['s4'] = variable.Variable(name='s4', values=['fail', 'survive']) # connectivity between node 1 (critical hub) to node 4

    return varis

def quantify_magnitudes(varis):

    eq_sources = json.load(open(os.path.join(BASE, 'eq_sources.json')))
    n_state = len(varis['m1'].values)

    s1_prob = eq_sources['s1']['prob_of_occurrence']
    s2_prob = eq_sources['s2']['prob_of_occurrence']

    m1_probs = [eq_sources['s1']['magnitude_probs'][str(mag)] for mag in varis['m1'].values]
    m2_probs = [eq_sources['s2']['magnitude_probs'][str(mag)] for mag in varis['m2'].values]

    probs_mag = {}

    # P(Magnitude 1)
    probs_mag['m1'] = cpt.Cpt(
        childs=[varis['m1']],
        C = np.arange(n_state+1), # including no EQ state
        p = np.array([1.0-s1_prob] + [v*s1_prob for v in m1_probs])
    )

    # P(Magnitude 2 | Source)
    s2_prob = 1e-4 # probability of EQ occuring on source 2
    probs_mag['m2'] = cpt.Cpt(
        childs=[varis['m2']],
        C = np.arange(n_state+1), # including no EQ state
        p = np.array([1.0-s2_prob] + [v*s2_prob for v in m2_probs])
    )

    return probs_mag
