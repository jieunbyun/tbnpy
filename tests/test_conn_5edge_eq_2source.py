from tbnpy import variable, cpt
import pytest, math
import examples.conn_5edge_eq_2source.s1_define_model as define_model

def test_define_variables1():
    varis = define_model.define_variables()

    assert 'm1' in varis
    assert 'pga3' in varis
    assert varis['x5'].values[0] == 'fail'
    assert varis['s4'].name == 's4'

def test_quantify_magnitudes1():
    varis = define_model.define_variables()
    probs_mag = define_model.quantify_magnitudes(varis)

    assert probs_mag['m1'].childs[0].name == 'm1'
    assert len(probs_mag['m2'].parents) == 0
    assert math.isclose(sum(probs_mag['m1'].p), 1.0, abs_tol=1e-6)