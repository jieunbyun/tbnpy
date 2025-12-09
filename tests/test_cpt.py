import pytest

from tbnpy import cpt, variable
import numpy as np

import torch

@pytest.fixture
def dict_cpt():
    ''' Use instance of Variables in the variables'''
    A1 = variable.Variable(**{'name': 'A1',
                              'values': ['s', 'f']})
    A2 = variable.Variable(**{'name': 'A2',
                              'values': ['s', 'f']})
    A3 = variable.Variable(**{'name': 'A3',
                              'values': ['s', 'f']})

    return {'childs': [A1],
            'parents': [A2, A3],
            'C': np.array([[2, 2, 3], [2, 1, 2], [1, 1, 1]]) - 1,
            'p': [1, 1, 1]}

def test_init1(dict_cpt):

    a = cpt.Cpt(**dict_cpt)
    assert isinstance(a, cpt.Cpt)

def test_init2(dict_cpt):

    a = cpt.Cpt(**dict_cpt)
    assert isinstance(a, cpt.Cpt)
    assert a.childs==dict_cpt['childs']
    assert a.parents == dict_cpt['parents']
    np.testing.assert_array_equal(a.C, dict_cpt['C'])

    assert isinstance(a.C, torch.Tensor)
    assert isinstance(a.p, torch.Tensor)

def test_init3(dict_cpt):
    # using list for P
    v = dict_cpt
    a = cpt.Cpt(childs=v['childs'], parents=v['parents'], C=np.array([[0, 2, 1], [1, 2, 1]]), p=[0.1, 0.9])
    assert isinstance(a, cpt.Cpt)

def test_init4(dict_cpt):
    # no p
    v = dict_cpt
    a = cpt.Cpt(childs=[v['childs'][0]], parents=v['parents'], C=np.array([1, 2]))
    assert isinstance(a, cpt.Cpt)


def test_init5():
    # variables must be a list of Variable
    with pytest.raises(AssertionError):
        a = cpt.Cpt(childs=['1'], parents=[], C=np.array([1]), p=[0.9])

