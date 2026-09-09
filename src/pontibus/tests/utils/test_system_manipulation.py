# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import pytest
from numpy.testing import assert_allclose
from openff.interchange import Interchange
from openff.packmol import solvate_topology
from openff.toolkit import ForceField, Molecule, Topology
from openmm import CMMotionRemover, MonteCarloBarostat, System
from openmm import unit as omm_unit

from pontibus.utils.molecule_utils import (
    _get_offmol_metadata,
    _set_offmol_metadata,
)
from pontibus.utils.settings import InterchangeFFSettings
from pontibus.utils.system_manipulation import (
    adjust_system,
    copy_interchange_with_replacement,
)


def test_adjust_forces_nothing():
    """
    A smoke test, this should just pass.
    """
    system = System()
    adjust_system(system)


def test_ajdust_forces_remove_com_remover():
    system = System()
    com_force = CMMotionRemover()
    system.addForce(com_force)
    adjust_system(system, remove_force_types=CMMotionRemover)

    assert system.getNumForces() == 0


def test_adjust_forces_add_comm_and_barostat():
    system = System()
    barostat = MonteCarloBarostat(1.0 * omm_unit.bar, 298.15 * omm_unit.kelvin)
    com_force = CMMotionRemover()
    adjust_system(system, add_forces=[barostat, com_force])

    assert system.getNumForces() == 2


@pytest.fixture(scope="module")
def forcefield():
    return ForceField("openff-2.0.0.offxml", "tip3p.offxml")


@pytest.fixture(scope="module")
def insert_molecule():
    m = Molecule.from_smiles("CCCC")
    m.generate_conformers(n_conformers=1)
    return m


@pytest.fixture(scope="module")
def del_molecule():
    m = Molecule.from_smiles("CCO")
    m.generate_conformers(n_conformers=1)
    return m


def test_copy_no_conformers(forcefield):
    m1 = Molecule.from_smiles("C")
    m2 = Molecule.from_smiles("O")
    topology = Topology.from_molecules([m1])
    inter = Interchange.from_smirnoff(forcefield, topology)

    with pytest.raises(ValueError, match="molecules need conformers"):
        _ = copy_interchange_with_replacement(
            interchange=inter,
            del_mol=m1,
            insert_mol=m2,
            ffsettings=InterchangeFFSettings(),
            charged_molecules=None,
        )


def test_copy_equality_clash(forcefield, insert_molecule, del_molecule):
    topology = Topology.from_molecules([del_molecule, del_molecule])
    inter = Interchange.from_smirnoff(forcefield, topology)

    with pytest.raises(ValueError, match="equality clash"):
        _ = copy_interchange_with_replacement(
            interchange=inter,
            del_mol=del_molecule,
            insert_mol=insert_molecule,
            ffsettings=InterchangeFFSettings(),
            charged_molecules=None,
        )


def test_copy_no_del_match(forcefield, insert_molecule, del_molecule):
    fake_del_mol = Molecule.from_smiles("C")
    fake_del_mol.generate_conformers(n_conformers=1)
    topology = Topology.from_molecules([del_molecule])
    inter = Interchange.from_smirnoff(forcefield, topology)

    with pytest.raises(ValueError, match="matching del_mol in input"):
        _ = copy_interchange_with_replacement(
            interchange=inter,
            del_mol=fake_del_mol,
            insert_mol=insert_molecule,
            ffsettings=InterchangeFFSettings,
            charged_molecules=None,
        )


def test_copy_noprotein_ff14sb(forcefield, insert_molecule, del_molecule):
    topology = Topology.from_molecules([del_molecule])
    inter = Interchange.from_smirnoff(forcefield, topology)

    with pytest.raises(ValueError, match="A protein component is necessary"):
        _ = copy_interchange_with_replacement(
            interchange=inter,
            del_mol=del_molecule,
            insert_mol=insert_molecule,
            ffsettings=InterchangeFFSettings(
                forcefields=["openff-2.0.0.offxml", "ff14sb_off_impropers_0.0.4.offxml"],
            ),
            charged_molecules=None,
            protein_component=None,
        )


def test_copy_full(forcefield):
    m1 = Molecule.from_smiles("CCCC")
    m1.generate_conformers(n_conformers=1)
    m1.assign_partial_charges(partial_charge_method="gasteiger")
    _set_offmol_metadata(m1, "residue_number", 999)
    m2 = Molecule.from_smiles("CCCO")
    m2.generate_conformers(n_conformers=1)
    m2.assign_partial_charges(partial_charge_method="gasteiger")

    # Solvate m1
    solvated_top = solvate_topology(Topology.from_molecules([m1]))

    # Create interchange
    inter = Interchange.from_smirnoff(forcefield, solvated_top, charge_from_molecules=[m1])

    inter_new = copy_interchange_with_replacement(
        interchange=inter,
        del_mol=m1,
        insert_mol=m2,
        ffsettings=InterchangeFFSettings(),
        charged_molecules=[m2],
    )

    assert inter.topology.n_molecules == inter_new.topology.n_molecules
    assert inter_new.topology.n_unique_molecules == 4
    for idx in range(inter_new.topology.n_molecules - 1):
        mol_new = inter_new.topology.molecule(idx)
        mol_old = inter.topology.molecule(idx + 1)
        assert mol_new.is_isomorphic_with(mol_old)

        assert_allclose(mol_new.conformers[0], mol_old.conformers[0])

    insert_mol_new = inter_new.topology.molecule(inter_new.topology.n_molecules - 1)
    assert insert_mol_new.is_isomorphic_with(m2)
    assert_allclose(m2.conformers[0], insert_mol_new.conformers[0])
    assert_allclose(m2.partial_charges, insert_mol_new.partial_charges)
    assert _get_offmol_metadata(insert_mol_new, "residue_number") == 999
