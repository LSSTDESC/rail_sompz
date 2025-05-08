import os
import urllib.request
import pytest


def remove_data() -> None:

    if os.environ.get('NO_TEARDOWN', 0):
        return

    for fout in [
        'tests/romandesc_deep_data_37c_noinf.hdf5',
        'tests/romandesc_spec_data_18c_noinf.hdf5',
        'tests/romandesc_wide_data_50c_noinf.hdf5',
        'tests/romandesc_deep_data_75k_noinf.hdf5',
        'tests/romandesc_spec_data_37k_noinf.hdf5',
        'tests/romandesc_wide_data_100k_noinf.hdf5',
    ]:
        try:
            os.unlink(fout)
        except:
            pass


@pytest.fixture(name="get_data", scope="package")
def get_data(request: pytest.FixtureRequest) -> int:
    
    if not os.path.exists("tests/roman_desc_demo_data.tar.gz"):
        urllib.request.urlretrieve(
            "https://portal.nersc.gov/cfs/lsst/PZ/roman_desc_demo_data.tar.gz",
            "tests/roman_desc_demo_data.tar.gz",
        )
        if not os.path.exists("tests/roman_desc_demo_data.tar.gz"):
            return 1

    os.system('tar zxvf tests/roman_desc_demo_data.tar.gz')
    os.system('mv romandesc_*.hdf5 tests')
    
    if not os.path.exists("tests/romandesc_spec_data_18c_noinf.hdf5"):
        return 2
            
    if not os.path.exists("tests/romandesc_wide_data_50c_noinf.hdf5"):
        return 2

    if not os.path.exists("tests/romandesc_deep_data_37c_noinf.hdf5"):
        return 2

    request.addfinalizer(remove_data)
    return 0


def remove_intermediates() -> None:

    if os.environ.get('NO_TEARDOWN', 0):
        return

    os.system('\\rm -rf tests/intermediates')   
    try:
        os.unlink('tests/intermediates.tgz')
    except:
        pass


@pytest.fixture(name="get_intermediates", scope="package")
def get_intermediates(request: pytest.FixtureRequest) -> int:
    
    if not os.path.exists("tests/intermediates.tgz"):
        urllib.request.urlretrieve(
            "",
            "tests/intermediates.tgz",
        )
        if not os.path.exists("tests/intermediates.tgz"):
            return 1

    os.system('tar zxvf tests/intermediates.tgz')
    

    request.addfinalizer(remove_intermediates)
    return 0

