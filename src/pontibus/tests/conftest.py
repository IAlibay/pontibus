# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import os
from importlib import resources

import gufe
import openfe
import pytest
from gufe import SmallMoleculeComponent
from openff.toolkit import ForceField, Topology
from openff.units import unit
from rdkit import Chem

from pontibus.utils.system_creation import _proteincomp_to_topology


class SlowTests:
    """Plugin for handling fixtures that skips slow tests

    Fixtures
    --------

    Currently three fixture types are handled:
      * `gpu`:
        GPU tests that are meant to be run to truly put the code
        through a real run.

      * `slow`:
        Unit tests that just take too long to be running regularly.

      * `cpuvslow`:
        Unit tests that run very slowly and likely require high
        performance CPU runners.


    How to use the fixtures
    -----------------------

    To add these fixtures simply add a `@pytest.mark.gpu` or
    `@pytest.mark.slow` or `@pytest.mark.cpuvslow`
    decorator to the relevant function or class.


    How to run tests marked by these fixtures
    -----------------------------------------

    To run the `gpu` tests, either use the `--gpu` flag
    when invoking pytest, or set the environment variable
    `PONTIBUS_GPU_TESTS` to `true`. Note: triggering `gpu` will
    automatically also trigger tests marked as `slow`.

    To run the `slow` tests, either use the `--slow` flag when invoking
    pytest, or set the environment variable `PONTIBUS_SLOW_TESTS` to `true`.

    To run the `cpuvslow` tests, either use the `--cpuvslow` flag when
    invoking pytest, or set the environment variable `PONTIBUS_CPUVSLOW_TESTS`
    to `true`.
    """

    def __init__(self, config):
        self.config = config

    @staticmethod
    def _modify_slow(items, config):
        msg = (
            "needs --slow pytest cli option or the environment variable "
            "`PONTIBUS_SLOW_TESTS` set to `True` to run"
        )
        skip_slow = pytest.mark.skip(reason=msg)
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    @staticmethod
    def _modify_gpu(items, config):
        msg = (
            "needs --gpu pytest cli option or the environment "
            "variable `PONTIBUS_GPU_TESTS` set to `True` to run"
        )
        skip_int = pytest.mark.skip(reason=msg)
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_int)

    @staticmethod
    def _modify_cpuvslow(items, config):
        msg = (
            "needs --cpuvslow pytest cli option or the environment "
            "variable `PONTIBUS_CPUVSLOW_TESTS` set to `True` to run"
        )
        skip_int = pytest.mark.skip(reason=msg)
        for item in items:
            if "cpuvslow" in item.keywords:
                item.add_marker(skip_int)

    def pytest_collection_modifyitems(self, items, config):
        if (
            config.getoption("--gpu")
            or os.getenv("PONTIBUS_GPU_TESTS", default="false").lower() == "true"
        ):
            self._modify_cpuvslow(items, config)
        elif (
            config.getoption("--cpuvslow")
            or os.getenv("PONTIBUS_CPUVSLOW_TESTS", default="false").lower() == "true"
        ):
            self._modify_gpu(items, config)
        elif (
            config.getoption("--slow")
            or os.getenv("PONTIBUS_SLOW_TESTS", default="false").lower() == "true"
        ):
            self._modify_gpu(items, config)
            self._modify_cpuvslow(items, config)
        else:
            self._modify_gpu(items, config)
            self._modify_cpuvslow(items, config)
            self._modify_slow(items, config)


# allow for optional slow tests
# See: https://docs.pytest.org/en/latest/example/simple.html
def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true", default=False, help="run slow tests")
    parser.addoption(
        "--gpu",
        action="store_true",
        default=False,
        help="run GPU tests",
    )
    parser.addoption(
        "--cpuvslow", action="store_true", default=False, help="run very slow CPU tests"
    )


def pytest_configure(config):
    config.pluginmanager.register(SlowTests(config), "slow")
    config.addinivalue_line("markers", "slow: mark test as slow")
    config.addinivalue_line("markers", "gpu: mark test as long integration test")
    config.addinivalue_line("markers", "cpuvslow: mark tests as CPU-only and very slow")


@pytest.fixture(scope="session")
def benzene_modifications():
    files = {}
    with resources.as_file(resources.files("pontibus.tests.data")) as d:
        fn = str(d / "benzene_modifications.sdf")
        supp = Chem.SDMolSupplier(str(fn), removeHs=False)
        for rdmol in supp:
            files[rdmol.GetProp("_Name")] = SmallMoleculeComponent(rdmol)
    return files


@pytest.fixture(scope="session")
def benzene_modifications_charged():
    files = {}
    with resources.as_file(resources.files("pontibus.tests.data")) as d:
        fn = str(d / "benzene_modifications_charged.sdf")
        supp = Chem.SDMolSupplier(str(fn), removeHs=False)
        for rdmol in supp:
            files[rdmol.GetProp("_Name")] = SmallMoleculeComponent(rdmol)
    return files


@pytest.fixture(scope="session")
def benzene_modifications_net_charge():
    files = {}
    with resources.as_file(resources.files("pontibus.tests.data")) as d:
        fn = str(d / "benzenes_with_net_charge.sdf")
        supp = Chem.SDMolSupplier(str(fn), removeHs=False)
        for rdmol in supp:
            files[rdmol.GetProp("_Name")] = SmallMoleculeComponent(rdmol)
    return files


@pytest.fixture(scope="session")
def thrombin_ligands_charged():
    files = {}
    with resources.as_file(resources.files("pontibus.tests.data")) as d:
        fn = str(d / "thrombin_ligands_elf10.sdf")
        supp = Chem.SDMolSupplier(str(fn), removeHs=False)
        for rdmol in supp:
            files[rdmol.GetProp("_Name")] = SmallMoleculeComponent(rdmol)
    return files


@pytest.fixture()
def CN_molecule():
    """
    A basic CH3NH2 molecule for quick testing.
    """
    with resources.as_file(resources.files("pontibus.tests.data")) as d:
        fn = str(d / "CN.sdf")
        supp = Chem.SDMolSupplier(str(fn), removeHs=False)

        smc = [SmallMoleculeComponent(i) for i in supp][0]

    return smc


@pytest.fixture(scope="session")
def T4_protein_component():
    with resources.as_file(resources.files("pontibus.tests.data")) as d:
        fn = str(d / "181l_only.pdb")
        comp = gufe.ProteinComponent.from_pdb_file(fn, name="T4_protein")

    return comp


@pytest.fixture(scope="session")
def vsite_offxml():
    offxml = (
        resources.files("pontibus.tests.data.forcefields")
        / "vsites-v3_n2r-vdw-dimers_hmix-only-v2-torsions.offxml"
    ).as_posix()

    ff = ForceField(offxml)
    return ff.to_string()


@pytest.fixture(scope="session")
def T4_protein_offtop(T4_protein_component):
    return _proteincomp_to_topology(T4_protein_component)


@pytest.fixture(scope="session")
def thrombin_protein_component():
    with resources.as_file(resources.files("pontibus.tests.data")) as d:
        fn = str(d / "thrombin.pdb")
        comp = gufe.ProteinComponent.from_pdb_file(fn, name="thrombin_protein")

    return comp


@pytest.fixture(scope="session")
def thrombin_protein_offtop(thrombin_protein_component):
    offtop = _proteincomp_to_topology(thrombin_protein_component)
    key = str(thrombin_protein_component.key)
    mols = [m for m in offtop.molecules]
    for m in mols:
        m.properties["key"] = key

    return Topology.from_molecules(mols)


@pytest.fixture
def benzene_vacuum_system(benzene_modifications_charged):
    return openfe.ChemicalSystem(
        {"ligand": benzene_modifications_charged["benzene"]},
    )


@pytest.fixture(scope="session")
def benzene_system(benzene_modifications_charged):
    return openfe.ChemicalSystem(
        {
            "ligand": benzene_modifications_charged["benzene"],
            "solvent": openfe.SolventComponent(
                positive_ion="Na", negative_ion="Cl", ion_concentration=0.15 * unit.molar
            ),
        },
    )


@pytest.fixture
def benzene_complex_system(benzene_modifications_charged, T4_protein_component):
    return openfe.ChemicalSystem(
        {
            "ligand": benzene_modifications_charged["benzene"],
            "solvent": openfe.SolventComponent(
                positive_ion="Na", negative_ion="Cl", ion_concentration=0.15 * unit.molar
            ),
            "protein": T4_protein_component,
        }
    )


@pytest.fixture
def toluene_vacuum_system(benzene_modifications_charged):
    return openfe.ChemicalSystem(
        {"ligand": benzene_modifications_charged["toluene"]},
    )


@pytest.fixture(scope="session")
def toluene_system(benzene_modifications_charged):
    return openfe.ChemicalSystem(
        {
            "ligand": benzene_modifications_charged["toluene"],
            "solvent": openfe.SolventComponent(
                positive_ion="Na", negative_ion="Cl", ion_concentration=0.15 * unit.molar
            ),
        },
    )


@pytest.fixture
def toluene_complex_system(benzene_modifications_charged, T4_protein_component):
    return openfe.ChemicalSystem(
        {
            "ligand": benzene_modifications_charged["toluene"],
            "solvent": openfe.SolventComponent(
                positive_ion="Na", negative_ion="Cl", ion_concentration=0.15 * unit.molar
            ),
            "protein": T4_protein_component,
        }
    )


@pytest.fixture(scope="session")
def benzene_to_toluene_mapping(benzene_modifications_charged):
    mapper = openfe.setup.LomapAtomMapper(element_change=False)

    molA = benzene_modifications_charged["benzene"]
    molB = benzene_modifications_charged["toluene"]

    return next(mapper.suggest_mappings(molA, molB))


@pytest.fixture(scope="session")
def eg5_protein_pdb():  # pragma: no cover
    with resources.as_file(resources.files("pontibus.tests.data")) as d:
        yield str(d / "eg5_protein.pdb")


@pytest.fixture(scope="session")
def eg5_ligands_sdf():  # pragma: no cover
    with resources.as_file(resources.files("pontibus.tests.data")) as d:
        yield str(d / "eg5_ligands_charged.sdf")


@pytest.fixture(scope="session")
def eg5_cofactor_sdf():  # pragma: no cover
    with resources.as_file(resources.files("pontibus.tests.data")) as d:
        yield str(d / "eg5_cofactor_charged.sdf")


@pytest.fixture(scope="session")
def eg5_protein(eg5_protein_pdb) -> openfe.ProteinComponent:  # pragma: no cover
    return openfe.ProteinComponent.from_pdb_file(eg5_protein_pdb)


@pytest.fixture(scope="session")
def eg5_ligands(eg5_ligands_sdf) -> list[SmallMoleculeComponent]:  # pragma: no cover
    return [SmallMoleculeComponent(m) for m in Chem.SDMolSupplier(eg5_ligands_sdf, removeHs=False)]


@pytest.fixture(scope="session")
def eg5_cofactor(eg5_cofactor_sdf) -> SmallMoleculeComponent:  # pragma: no cover
    return SmallMoleculeComponent.from_sdf_file(eg5_cofactor_sdf)


@pytest.fixture(scope="session")
def eg5_complex_systemA(eg5_protein, eg5_ligands, eg5_cofactor):  # pragma: no cover
    mol, _ = eg5_ligands
    return openfe.ChemicalSystem(
        {
            "protein": eg5_protein,
            "ligand": mol,
            "cofactors": eg5_cofactor,
            "solvent": openfe.SolventComponent(),
        }
    )


@pytest.fixture(scope="session")
def eg5_complex_systemB(eg5_protein, eg5_ligands, eg5_cofactor):  # pragma: no cover
    _, mol = eg5_ligands
    return openfe.ChemicalSystem(
        {
            "protein": eg5_protein,
            "ligand": mol,
            "cofactors": eg5_cofactor,
            "solvent": openfe.SolventComponent(),
        }
    )


@pytest.fixture(scope="session")
def eg5_ligands_mapping(benzene_modifications_charged, eg5_ligands):  # pragma: no cover
    mapper = openfe.setup.LomapAtomMapper(element_change=False)

    molA, molB = eg5_ligands

    return next(mapper.suggest_mappings(molA, molB))
