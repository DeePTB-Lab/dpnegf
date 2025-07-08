import pytest
from dpnegf.utils.elec_struc_cal import ElecStruCal
from dptb.nn.build import build_model


@pytest.fixture(scope='session', autouse=True)
def root_directory(request):
    """
    :return:
    """
    return str(request.config.rootdir)

def test_get_fermi(root_directory):
    ckpt = f"{root_directory}/dpnegf/tests/data/test_get_fermi/nnsk.best.pth"  #  'hopping': {'method': 'poly2exp', 'rs': 5.0, 'w': 0.6},
    stru_data = f"{root_directory}/dpnegf/tests/data/test_get_fermi/PRIMCELL.vasp"

    model = build_model(checkpoint=ckpt)
    nel_atom = {"Au":11}

    elec_cal = ElecStruCal(model=model,device='cpu')
    _, efermi =elec_cal.get_fermi_level(data=stru_data, 
                    nel_atom = nel_atom,smearing_method='FD',
                meshgrid=[30,30,30])
    assert abs(efermi  + 3.2257686853408813) < 1e-6

    _, efermi =elec_cal.get_fermi_level(data=stru_data, 
                    nel_atom = nel_atom,smearing_method='Gaussian',
                meshgrid=[30,30,30])
    assert abs(efermi  + 3.2267462015151978) < 1e-6
