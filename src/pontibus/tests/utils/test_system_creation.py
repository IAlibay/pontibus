# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import copy
import logging

import numpy as np
import pytest
from gufe import SmallMoleculeComponent, SolventComponent
from numpy.testing import assert_allclose, assert_equal
from openff.interchange.interop.openmm import to_openmm_positions
from openff.toolkit import ForceField, Molecule, Quantity, Topology, unit
from openff.units.openmm import from_openmm, to_openmm
from openmm import (
    HarmonicAngleForce,
    HarmonicBondForce,
    NonbondedForce,
    PeriodicTorsionForce,
)
from openmm import unit as omm_unit
from rdkit import Chem

from pontibus.components.extended_solvent_component import ExtendedSolventComponent
from pontibus.utils.molecule_utils import (
    _get_offmol_resname,
    _set_offmol_resname,
)
from pontibus.utils.molecules import WATER
from pontibus.utils.settings import (
    InterchangeFFSettings,
    InterchangeOpenMMSolvationSettings,
    PackmolSolvationSettings,
)
from pontibus.utils.system_creation import (
    _assign_comp_resnames_and_keys,
    _check_and_deduplicate_charged_mols,
    _fill_vsite_compresids,
    _get_force_field,
    _protein_split_combine_interchange,
    _proteincomp_to_topology,
    interchange_system_creation,
)
from pontibus.utils.system_manipulation import copy_interchange_with_replacement
from pontibus.utils.system_solvation import packmol_solvation


@pytest.fixture(scope="module")
def protein_ff_settings():
    return InterchangeFFSettings(
        forcefields=[
            "openff-2.0.0.offxml",
            "ff14sb_off_impropers_0.0.4.offxml",
        ],
    )


@pytest.fixture(scope="module")
def smc_components_benzene_unnamed(benzene_modifications):
    benzene_off = benzene_modifications["benzene"].to_openff()
    benzene_off.assign_partial_charges(partial_charge_method="gasteiger")
    return {benzene_modifications["benzene"]: benzene_off}


@pytest.fixture(scope="module")
def smc_components_benzene_named(benzene_modifications):
    benzene_off = benzene_modifications["benzene"].to_openff()
    _set_offmol_resname(benzene_off, "BNZ")
    benzene_off.assign_partial_charges(partial_charge_method="gasteiger")
    return {benzene_modifications["benzene"]: benzene_off}


@pytest.fixture(scope="module")
def smc_components_acetic_acid():
    mol = Molecule.from_smiles("CC(=O)[O-]")
    mol.generate_conformers(n_conformers=1)
    mol.assign_partial_charges(partial_charge_method="gasteiger")
    return {SmallMoleculeComponent.from_openff(mol): mol}


@pytest.fixture()
def methanol():
    m = Molecule.from_smiles("CO")
    m.generate_conformers()
    m.assign_partial_charges(partial_charge_method="gasteiger")
    return m


@pytest.fixture(scope="module")
def water_off():
    m = WATER.to_openff()
    return m


@pytest.fixture(scope="module")
def water_off_named_charged():
    water = WATER.to_openff()
    _set_offmol_resname(water, "HOH")
    water.assign_partial_charges(partial_charge_method="gasteiger")
    return water


@pytest.fixture(scope="module")
def water_off_am1bcc():
    water = WATER.to_openff()
    water.assign_partial_charges(partial_charge_method="am1bcc")
    return water


def test_convert_proteincomp(T4_protein_component):
    # Get an OpenFF Topology by going through rdkit
    rdmols = Chem.GetMolFrags(T4_protein_component.to_rdkit(), asMols=True, sanitizeFrags=False)
    ofe_mol = Molecule.from_rdkit(
        rdmols[0], allow_undefined_stereo=True, hydrogens_are_explicit=True
    )
    ofe_top = Topology.from_molecules([ofe_mol])
    # The the OpenFF Topology with the tooling
    off_top = _proteincomp_to_topology(T4_protein_component)

    # light isormophic check
    assert off_top.molecule(0).is_isomorphic_with(
        ofe_top.molecule(0), atom_stereochemistry_matching=False
    )


def test_get_and_set_offmol_resname(CN_molecule, caplog):
    CN_off = CN_molecule.to_openff()

    # No residue name to begin with
    assert _get_offmol_resname(CN_off) is None

    # Boop the floof
    _set_offmol_resname(CN_off, "BOOP")

    # Does the floof accept the boop?
    assert "BOOP" == _get_offmol_resname(CN_off)

    # Oh no, one of the atoms didn't like the boop!
    atom3 = list(CN_off.atoms)[2]
    atom3.metadata["residue_name"] = "NOBOOP"

    with caplog.at_level(logging.WARNING):
        assert _get_offmol_resname(CN_off) is None
    assert "Inconsistent metadata residue_name" in caplog.text


def test_check_charged_mols_pass(methanol):
    _check_and_deduplicate_charged_mols([methanol])


def test_check_deduplicated_charged_mols(smc_components_benzene_unnamed):
    """
    Base test case for deduplication. Same molecule, same partial charges,
    different conformer.
    """
    benzene1 = list(smc_components_benzene_unnamed.values())[0]
    benzene1.assign_partial_charges(partial_charge_method="gasteiger")
    benzene2 = Molecule.from_smiles("c1ccccc1")
    benzene2.generate_conformers(n_conformers=1)
    benzene2.assign_partial_charges(partial_charge_method="gasteiger")

    assert all(benzene1.partial_charges == benzene2.partial_charges)
    assert np.any(benzene1.conformers[0] != benzene2.conformers[0])
    assert benzene1.is_isomorphic_with(benzene2)

    uniques = _check_and_deduplicate_charged_mols([benzene1, benzene2])

    assert len(uniques) == 1
    assert uniques[0] == benzene1


def test_check_charged_mols_nocharge(water_off, methanol):
    with pytest.raises(ValueError, match="One or more"):
        _check_and_deduplicate_charged_mols([water_off, methanol])


def test_check_charged_mols(water_off_am1bcc, water_off_named_charged):
    with pytest.raises(ValueError, match="different charges"):
        _check_and_deduplicate_charged_mols([water_off_am1bcc, water_off_named_charged])


def test_protein_component_nosolv_fail(smc_components_benzene_named, T4_protein_component):
    errmsg = "Must have solvent to have a protein"
    with pytest.raises(ValueError, match=errmsg):
        interchange_system_creation(
            ffsettings=InterchangeFFSettings(),
            solvation_settings=PackmolSolvationSettings(),
            smc_components=smc_components_benzene_named,
            protein_component=T4_protein_component,
            solvent_component=None,
            solvent_offmol=None,
        )


@pytest.mark.parametrize(
    "neutralize, ion_conc",
    [
        [True, 0.0 * unit.molar],
        [False, 0.1 * unit.molar],
        [True, 0.1 * unit.molar],
    ],
)
def test_wrong_solventcomp_settings_nonwater(
    neutralize, ion_conc, smc_components_benzene_named, methanol
):
    with pytest.raises(ValueError, match="Counterions are currently not"):
        interchange_system_creation(
            ffsettings=InterchangeFFSettings(),
            solvation_settings=PackmolSolvationSettings(),
            smc_components=smc_components_benzene_named,
            protein_component=None,
            solvent_component=ExtendedSolventComponent(
                neutralize=neutralize,
                ion_concentration=ion_conc,
                solvent_molecule=SmallMoleculeComponent.from_openff(methanol),
            ),
            solvent_offmol=methanol,
        )


def test_not_neutralize_but_ion_conc(
    smc_components_benzene_named,
    water_off,
):
    with pytest.raises(ValueError, match="Cannot add ions without"):
        interchange_system_creation(
            ffsettings=InterchangeFFSettings(),
            solvation_settings=PackmolSolvationSettings(),
            smc_components=smc_components_benzene_named,
            protein_component=None,
            solvent_component=ExtendedSolventComponent(
                neutralize=False,
                ion_concentration=0.1 * unit.molar,
            ),
            solvent_offmol=water_off,
        )


@pytest.mark.parametrize("pos, neg", [["Na+", "F-"], ["K+", "Cl-"], ["K+", "F-"]])
def test_bad_ions(
    smc_components_benzene_named,
    water_off,
    pos,
    neg,
):
    with pytest.raises(ValueError, match="Can only neutralize with NaCl"):
        interchange_system_creation(
            ffsettings=InterchangeFFSettings(),
            solvation_settings=PackmolSolvationSettings(),
            smc_components=smc_components_benzene_named,
            protein_component=None,
            solvent_component=ExtendedSolventComponent(
                neutralize=True,
                ion_concentration=0 * unit.molar,
                positive_ion=pos,
                negative_ion=neg,
            ),
            solvent_offmol=water_off,
        )


@pytest.mark.parametrize("resname", ["NA+", "CL-"])
def test_resname_solvent_ion_clash(smc_components_benzene_named, resname):
    solv_off = WATER.to_openff()
    _set_offmol_resname(solv_off, resname)

    errmsg = "Solvent resname is set to"
    with pytest.raises(ValueError, match=errmsg):
        _assign_comp_resnames_and_keys(
            smc_components_benzene_named,
            ExtendedSolventComponent(neutralize=True),
            solv_off,
            None,
            None,
        )


def test_comp_resnames_and_keys(
    smc_components_benzene_named, water_off, T4_protein_component, T4_protein_offtop
):
    protmol = copy.deepcopy(T4_protein_offtop.molecule(0))

    _assign_comp_resnames_and_keys(
        smc_components=smc_components_benzene_named,
        solvent_component=SolventComponent(),
        solvent_offmol=water_off,
        protein_component=T4_protein_component,
        protein_molecules=[protmol],
    )

    assert water_off.properties["key"] == SolventComponent().key
    assert protmol.properties["key"] == T4_protein_component.key
    for key, val in smc_components_benzene_named.items():
        assert val.properties["key"] == key.key


def test_comp_resnames_and_keys_missing_prot_residueinfo(
    smc_components_benzene_named, water_off, T4_protein_component, T4_protein_offtop
):
    protmol = copy.deepcopy(T4_protein_offtop.molecule(0))
    protmol.atoms[0].metadata.pop("residue_name")

    with pytest.raises(ValueError, match="missing residue info"):
        _assign_comp_resnames_and_keys(
            smc_components=smc_components_benzene_named,
            solvent_component=SolventComponent(),
            solvent_offmol=water_off,
            protein_component=T4_protein_component,
            protein_molecules=[protmol],
        )


def test_packmol_solvation_neutralize_nonwater(methanol):
    msg = "Cannot neutralize a system with non-water solvent"
    with pytest.raises(ValueError, match=msg):
        packmol_solvation(
            solute_topology=Topology.from_molecules(methanol),
            solvent_offmol=methanol,
            solvation_settings=PackmolSolvationSettings(),
            neutralize=True,
            ion_concentration=0.1 * unit.molar,
        )


def test_packmol_solvation_neutralize_bad_conc(water_off, methanol):
    msg = "is not compatible with mole / liter"
    with pytest.raises(ValueError, match=msg):
        packmol_solvation(
            solute_topology=Topology.from_molecules(methanol),
            solvent_offmol=water_off,
            solvation_settings=PackmolSolvationSettings(),
            neutralize=True,
            ion_concentration=0.1 * unit.nm,
        )


def test_solv_but_no_solv_offmol(
    smc_components_benzene_named,
):
    with pytest.raises(ValueError, match="A solvent offmol"):
        interchange_system_creation(
            ffsettings=InterchangeFFSettings(),
            solvation_settings=PackmolSolvationSettings(),
            smc_components=smc_components_benzene_named,
            protein_component=None,
            solvent_component=ExtendedSolventComponent(),
            solvent_offmol=None,
        )


def test_solv_mismatch(
    smc_components_benzene_named,
    methanol,
):
    assert ExtendedSolventComponent().smiles == "[H][O][H]"
    with pytest.raises(ValueError, match="does not match"):
        interchange_system_creation(
            ffsettings=InterchangeFFSettings(),
            solvation_settings=PackmolSolvationSettings(),
            smc_components=smc_components_benzene_named,
            protein_component=None,
            solvent_component=ExtendedSolventComponent(),
            solvent_offmol=methanol,
        )


def test_no_solvent_conformers(
    smc_components_benzene_named,
):
    solmol = Molecule.from_smiles("C")
    solmol.assign_partial_charges(partial_charge_method="gasteiger")
    solmol.generate_conformers()
    solvent = ExtendedSolventComponent(solvent_molecule=SmallMoleculeComponent.from_openff(solmol))
    solmol._conformers = []

    with pytest.raises(ValueError, match="single conformer"):
        interchange_system_creation(
            ffsettings=InterchangeFFSettings(),
            solvation_settings=PackmolSolvationSettings(assign_solvent_charges=True),
            smc_components=smc_components_benzene_named,
            protein_component=None,
            solvent_component=solvent,
            solvent_offmol=solmol,
        )


def test_get_force_field():
    ffsettings = InterchangeFFSettings(
        nonbonded_cutoff=1.0 * unit.nanometer,
        switch_width=0.2 * unit.nanometer,
    )

    ff = _get_force_field(ffsettings, exclude_ff14sb=True)

    assert ff["vdW"].cutoff == ff["Electrostatics"].cutoff == 1.0 * unit.nanometer
    assert ff["vdW"].switch_width == 0.2 * unit.nanometer


def test_get_force_field_custom():
    """
    Re-implementation of the test Josh implemented in OMMForcefields
    """
    sage = ForceField("openff-2.2.1.offxml")
    ethane = Molecule.from_smiles("C")
    bond_parameter = sage.label_molecules(ethane.to_topology())[0]["Bonds"][(0, 1)]
    bonds = sage.get_parameter_handler("Bonds")
    new_parameter = bonds[bond_parameter.smirks]
    new_parameter.length = 2 * unit.angstrom

    ffsettings = InterchangeFFSettings(forcefields=[sage.to_string(), "opc.offxml"])

    # Setting exclusion to False purely to check that it still works
    ff = _get_force_field(ffsettings, exclude_ff14sb=False)

    bonds = ff.get_parameter_handler("Bonds")
    bond_param = bonds[bond_parameter.smirks]
    assert bond_param.length == 2 * unit.angstrom


def test_multiple_solvent_conformers(
    smc_components_benzene_named,
):
    solmol = Molecule.from_smiles("CCCCCCCCCCCCCCCCCCCCCCCCCCC")
    solmol.generate_conformers()
    solmol.assign_partial_charges(partial_charge_method="gasteiger")
    solvent = ExtendedSolventComponent(solvent_molecule=SmallMoleculeComponent.from_openff(solmol))

    with pytest.raises(ValueError, match="single conformer"):
        interchange_system_creation(
            ffsettings=InterchangeFFSettings(),
            solvation_settings=PackmolSolvationSettings(assign_solvent_charges=True),
            smc_components=smc_components_benzene_named,
            protein_component=None,
            solvent_component=solvent,
            solvent_offmol=solmol,
        )


@pytest.mark.parametrize(
    "assign_charges, errmsg",
    [
        (True, "do not have partial charges"),
        (False, "No library charges"),
    ],
)
def test_charge_assignment_errors(smc_components_benzene_named, assign_charges, errmsg):
    """
    True case: passing a Molecule without partial charges to Interchange
    and asking to get charges from it will fail.
    False case: not having any partial charges will try to see if it's
    in the LibraryCharges if not, it will fail (which it does here).
    """
    solvent_offmol = Molecule.from_smiles("COC")
    solvent_offmol.generate_conformers(n_conformers=1)

    with pytest.raises(ValueError, match=errmsg):
        _, _ = interchange_system_creation(
            ffsettings=InterchangeFFSettings(
                forcefields=[
                    "openff-2.0.0.offxml",
                ]
            ),
            solvation_settings=PackmolSolvationSettings(
                assign_solvent_charges=assign_charges,
            ),
            smc_components=smc_components_benzene_named,
            protein_component=None,
            solvent_component=SolventComponent(
                smiles="COC",
                neutralize=False,
                ion_concentration=0 * unit.molar,
            ),
            solvent_offmol=solvent_offmol,
        )


def test_assign_duplicate_resnames(caplog):
    """
    Pass two smcs named the same and expect one to be renamed
    """
    a = Molecule.from_smiles("C")
    b = Molecule.from_smiles("CCC")
    a.generate_conformers()
    b.generate_conformers()
    a.assign_partial_charges(partial_charge_method="gasteiger")
    b.assign_partial_charges(partial_charge_method="gasteiger")
    _set_offmol_resname(a, "FOO")
    _set_offmol_resname(b, "FOO")
    smc_a = SmallMoleculeComponent.from_openff(a)
    smc_b = SmallMoleculeComponent.from_openff(b)

    smcs = {smc_a: a, smc_b: b}

    with caplog.at_level(logging.WARNING):
        _, smc_comps = interchange_system_creation(
            ffsettings=InterchangeFFSettings(
                forcefields=[
                    "openff-2.0.0.offxml",
                ]
            ),
            solvation_settings=None,
            smc_components=smcs,
            protein_component=None,
            solvent_component=None,
            solvent_offmol=None,
        )
    for match in ["Duplicate", "residue name to AAA"]:
        assert match in caplog.text

    assert len(smc_comps) == 2
    assert smc_comps[smc_a][0] == 0
    assert smc_comps[smc_b][0] == 1


@pytest.mark.parametrize(
    "smiles",
    [
        "C1CCCCC1",
        "C1CCOC1",
        "CC(=O)N(C)C",
        "CC(=O)c1ccccc1",
        "CC(Br)Br",
        "CC(C)CO",
        "CC(C)O",
        "CC(C)OC(C)C",
        "CC(C)c1ccccc1",
        "CC(Cl)Cl",
        "CCBr",
        "CCCCC",
        "CCCCCC",
        "CCCCCC(C)C",
        "CCCCCCC",
        "CCCCCCCC",
        "CCCCCCCCBr",
        "CCCCCCCl",
        "CCCCCCO",
        "CCCCCO",
        "CCCCO",
        "CCCCOCCCC",
        "CCCCOP(=O)(OCCCC)OCCCC",
        "CCCCc1ccccc1",
        "CCCO",
        "CCN(CC)CC",
        "CCO",
        "CCOCC",
        "CCOc1ccccc1",
        "CCc1ccccc1",
        "CN(C)C=O",
        "CNC=O",
        "COc1ccccc1",
        "Cc1ccccc1",
        "Cc1ccccc1C",
        "Cc1ccccc1C(C)C",
        "Cc1ccccn1",
        "Cc1cccnc1C",
        "c1ccc2c(c1)CCCC2",
        "c1ccccc1",
        "c1ccncc1",
    ],
)
def test_nonwater_solvent_short(smc_components_benzene_named, smiles):
    solvent_offmol = Molecule.from_smiles(smiles)
    solvent_offmol.assign_partial_charges(partial_charge_method="gasteiger")
    solvent_offmol.generate_conformers(n_conformers=1)

    if smiles == "c1ccccc1":
        ligand = list(smc_components_benzene_named.values())[0]
        assert solvent_offmol.is_isomorphic_with(ligand)
        assert_allclose(solvent_offmol.partial_charges, ligand.partial_charges)
        assert all(solvent_offmol.partial_charges == ligand.partial_charges)

    interchange, _ = interchange_system_creation(
        ffsettings=InterchangeFFSettings(
            forcefields=[
                "openff-2.0.0.offxml",
            ]
        ),
        solvation_settings=PackmolSolvationSettings(
            solvent_padding=None,
            number_of_solvent_molecules=100,
            assign_solvent_charges=True,
            target_density=0.715 * unit.grams / unit.mL,
        ),
        smc_components=smc_components_benzene_named,
        protein_component=None,
        solvent_component=SolventComponent(
            smiles=smiles,
            neutralize=False,
            ion_concentration=0 * unit.molar,
        ),
        solvent_offmol=solvent_offmol,
    )

    if smiles == "c1ccccc1":
        # solvent == ligand
        assert solvent_offmol.is_isomorphic_with(list(smc_components_benzene_named.values())[0])
        assert interchange.topology.n_unique_molecules == 1
    else:
        assert interchange.topology.n_unique_molecules == 2
        assert solvent_offmol.is_isomorphic_with(list(interchange.topology.unique_molecules)[1])
    assert interchange.topology.n_molecules == 101


@pytest.mark.cpuvslow  # pragma: no cover
@pytest.mark.parametrize(
    "solvent_smiles, solute_smiles",
    [
        ("Cc1ccccn1", "CCCCCCCC"),
        ("Cc1ccccn1", "Cc1ccccc1"),
        ("Cc1ccccn1", "CCO"),
        ("Cc1ccccn1", "C1COCCO1"),
        ("Cc1ccccn1", "CCC(C)=O"),
        ("CC(=O)c1ccccc1", "CCCCCCCC"),
        ("CC(=O)c1ccccc1", "Cc1ccccc1"),
        ("CC(=O)c1ccccc1", "CCO"),
        ("CC(=O)c1ccccc1", "C1COCCO1"),
        ("CC(=O)c1ccccc1", "CCC(C)=O"),
        ("COc1ccccc1", "CCCCCCCC"),
        ("COc1ccccc1", "Cc1ccccc1"),
        ("COc1ccccc1", "CCO"),
        ("COc1ccccc1", "C1COCCO1"),
        ("COc1ccccc1", "CCC(C)=O"),
        ("COc1ccccc1", "CCCCN"),
        ("c1ccccc1", "CCCCCCCC"),
        ("c1ccccc1", "C1CCCCC1"),
        ("c1ccccc1", "CC(C)O"),
        ("c1ccccc1", "CC(C)(C)O"),
        ("c1ccccc1", "C1COCCO1"),
        ("c1ccccc1", "CC(C)=O"),
        ("c1ccccc1", "CCC(C)=O"),
        ("c1ccccc1", "COC(C)=O"),
        ("c1ccccc1", "CNC"),
        ("c1ccccc1", "CN(C)C"),
        ("c1ccccc1", "c1ccncc1"),
        ("c1ccccc1", "Cc1cccc(C)n1"),
        ("c1ccccc1", "O=Cc1cccc(O)c1"),
        ("c1ccccc1", "O=Cc1ccc(O)cc1"),
        ("c1ccccc1", "Oc1ccc(Br)cc1"),
        ("c1ccccc1", "CCOP(=O)(OCC)OCC"),
        ("c1ccccc1", "C1CCNCC1"),
        ("c1ccccc1", "CN"),
        ("c1ccccc1", "COC(=O)c1ccccc1"),
        ("c1ccccc1", "COP(=O)(OC)OC=C(Cl)Cl"),
        ("c1ccccc1", "Cc1cncc(C)c1"),
        ("c1ccccc1", "NC(=O)c1ccccc1"),
        ("c1ccccc1", "Cc1ccccc1N"),
        ("CCBr", "CCCCCCCC"),
        ("CCBr", "Cc1ccccc1"),
        ("CCBr", "C1COCCO1"),
        ("CCBr", "CCC(C)=O"),
        ("CCCCCCCCBr", "COC(C)=O"),
        ("CCCCCCCCBr", "CCCCCOC(C)=O"),
        ("CCCCO", "CCCCCCCC"),
        ("CCCCO", "c1ccccc1"),
        ("CCCCO", "CO"),
        ("CCCCO", "CCO"),
        ("CCCCO", "C=O"),
        ("CCCCO", "CCC(C)=O"),
        ("CCCCO", "CCN"),
        ("CCCCO", "CCCN"),
        ("CCCCO", "CCNCC"),
        ("CCCCO", "O=c1cc[nH]c(=O)[nH]1"),
        ("CCCCc1ccccc1", "Oc1ccccc1"),
        ("CCCCc1ccccc1", "CC(=O)C(C)(C)C"),
        ("CCCCc1ccccc1", "CCCCCC(C)=O"),
        ("CCCCc1ccccc1", "CCC(=O)OC"),
        ("CCCCc1ccccc1", "CCCCOC(C)=O"),
        ("CCCCCCCl", "CC(C)=O"),
        ("CCCCCCCl", "CC(=O)C(C)(C)C"),
        ("CCCCCCCl", "CCC(=O)OC"),
        ("CCCCCCCl", "CCCCOC(C)=O"),
        ("C1CCCCC1", "CCC"),
        ("C1CCCCC1", "CCCCCCCC"),
        ("C1CCCCC1", "CC(C)O"),
        ("C1CCCCC1", "CC(C)(C)O"),
        ("C1CCCCC1", "C1COCCO1"),
        ("C1CCCCC1", "COc1ccccc1"),
        ("C1CCCCC1", "CCC(=O)CC"),
        ("C1CCCCC1", "CC(=O)C(C)(C)C"),
        ("C1CCCCC1", "COC(C)=O"),
        ("C1CCCCC1", "CCN"),
        ("C1CCCCC1", "CN(C)C"),
        ("C1CCCCC1", "CCNCC"),
        ("C1CCCCC1", "c1ccncc1"),
        ("C1CCCCC1", "CNc1ccccc1"),
        ("C1CCCCC1", "CCCS"),
        ("C1CCCCC1", "CSc1ccccc1"),
        ("C1CCCCC1", "O=Cc1cccc(O)c1"),
        ("C1CCCCC1", "O=Cc1ccc(O)cc1"),
        ("C1CCCCC1", "Clc1ccccc1"),
        ("C1CCCCC1", "Clc1ccc(Cl)cc1"),
        ("C1CCCCC1", "Brc1ccccc1"),
        ("C1CCCCC1", "Oc1ccc(Br)cc1"),
        ("C1CCCCC1", "COP(=O)(OC)OC"),
        ("C1CCCCC1", "CCOP(=O)(OCC)OCC"),
        ("C1CCCCC1", "COC(=O)c1ccccc1"),
        ("C1CCCCC1", "C1CCOCC1"),
        ("C1CCCCC1", "N#Cc1cc(Br)c(O)c(Br)c1"),
        ("C1CCCCC1", "CN(C)C=O"),
        ("C1CCCCC1", "c1cc[nH]c1"),
        ("C1CCCCC1", "NC(=O)c1ccccc1"),
        ("C1CCCCC1", "Cc1ccccc1N"),
        ("CC(Br)Br", "CO"),
        ("CC(Br)Br", "CCCCO"),
        ("CC(Br)Br", "Oc1ccccc1"),
        ("CC(Br)Br", "Oc1ccc(Br)cc1"),
        ("CCCCOCCCC", "CCCCCCCC"),
        ("CCCCOCCCC", "CCO"),
        ("CCCCOCCCC", "C1COCCO1"),
        ("CCCCOCCCC", "CCC(C)=O"),
        ("CCCCOCCCC", "CCNCC"),
        ("CCCCOCCCC", "c1ccncc1"),
        ("CCCCOCCCC", "Cc1cnccn1"),
        ("CCCCOCCCC", "Cc1ccccn1"),
        ("CCCCOCCCC", "CCc1cnccn1"),
        ("CC(Cl)Cl", "CO"),
        ("CC(Cl)Cl", "CCCCO"),
        ("CC(Cl)Cl", "Cc1ccccc1O"),
        ("CC(Cl)Cl", "O=Cc1ccccc1"),
        ("CC(Cl)Cl", "CC(=O)c1ccccc1"),
        ("CC(Cl)Cl", "COC(C)=O"),
        ("CC(Cl)Cl", "CCCCCC(=O)OC"),
        ("CC(Cl)Cl", "CCN"),
        ("CC(Cl)Cl", "CCCN"),
        ("CC(Cl)Cl", "CCNCC"),
        ("CC(Cl)Cl", "c1ccncc1"),
        ("CC(Cl)Cl", "O=Cc1cccc(O)c1"),
        ("CC(Cl)Cl", "Oc1ccc(Br)cc1"),
        ("CC(Cl)Cl", "COP(=O)(OC)OC"),
        ("CC(Cl)Cl", "CCOP(=O)(OCC)OCC"),
        ("CC(Cl)Cl", "NC(=O)c1ccccc1"),
        ("CCOCC", "CCCCCCCC"),
        ("CCOCC", "CC(C)O"),
        ("CCOCC", "CC(C)(C)O"),
        ("CCOCC", "C1COCCO1"),
        ("CCOCC", "COc1ccccc1"),
        ("CCOCC", "CC=O"),
        ("CCOCC", "CCC(C)=O"),
        ("CCOCC", "CC(=O)c1ccccc1"),
        ("CCOCC", "CNC"),
        ("CCOCC", "CN(C)C"),
        ("CCOCC", "c1ccncc1"),
        ("CCOCC", "Nc1ccccc1"),
        ("CCOCC", "C=CCO"),
        ("CCOCC", "COCCO"),
        ("CCOCC", "O=Cc1cccc(O)c1"),
        ("CCOCC", "Clc1ccccc1"),
        ("CCOCC", "Clc1ccc(Cl)cc1"),
        ("CCOCC", "Brc1ccccc1"),
        ("CCOCC", "S"),
        ("CCOCC", "C1CCNCC1"),
        ("CCOCC", "CN"),
        ("CCOCC", "CC(N)=O"),
        ("CCOCC", "CN(C)C=O"),
        ("CCOCC", "NC=O"),
        ("CCOCC", "O=c1cc[nH]c(=O)[nH]1"),
        ("CCOCC", "O=c1[nH]cc(Br)c(=O)[nH]1"),
        ("CC(C)OC(C)C", "CCCCCCCC"),
        ("CC(C)OC(C)C", "c1ccc2ccccc2c1"),
        ("CC(C)OC(C)C", "CCO"),
        ("CC(C)OC(C)C", "C1COCCO1"),
        ("CC(C)OC(C)C", "C=O"),
        ("CC(C)OC(C)C", "CCC(C)=O"),
        ("CC(C)OC(C)C", "CN(C)C"),
        ("CC(C)OC(C)C", "CCNCC"),
        ("CC(C)OC(C)C", "c1ccncc1"),
        ("CC(C)OC(C)C", "Nc1ccccc1"),
        ("CC(C)OC(C)C", "O=Cc1ccc(O)cc1"),
        ("CC(=O)N(C)C", "CCCCCCCC"),
        ("CC(=O)N(C)C", "Cc1ccccc1"),
        ("CC(=O)N(C)C", "CCO"),
        ("CC(=O)N(C)C", "C1COCCO1"),
        ("CC(=O)N(C)C", "CCC(C)=O"),
        ("CN(C)C=O", "CCCCCCCC"),
        ("CN(C)C=O", "Cc1ccccc1"),
        ("CN(C)C=O", "CCO"),
        ("CN(C)C=O", "C1COCCO1"),
        ("CN(C)C=O", "CCC(C)=O"),
        ("Cc1cccnc1C", "CCCCCCCC"),
        ("Cc1cccnc1C", "Cc1ccccc1"),
        ("Cc1cccnc1C", "CCO"),
        ("Cc1cccnc1C", "C1COCCO1"),
        ("Cc1cccnc1C", "CCC(C)=O"),
        ("Cc1cccnc1C", "Cc1cccc(C)n1"),
        ("CCO", "CCCCCCCC"),
        ("CCO", "Cc1ccccc1"),
        ("CCO", "C1COCCO1"),
        ("CCO", "CCC(C)=O"),
        ("CCO", "Clc1ccccc1"),
        ("CCO", "O=C1CCCO1"),
        ("CCOc1ccccc1", "CCCCCCCC"),
        ("CCOc1ccccc1", "Cc1ccccc1"),
        ("CCOc1ccccc1", "CCO"),
        ("CCOc1ccccc1", "C1COCCO1"),
        ("CCOc1ccccc1", "CCC(C)=O"),
        ("CCc1ccccc1", "CO"),
        ("CCc1ccccc1", "CCCCCO"),
        ("CCc1ccccc1", "Cc1ccccc1O"),
        ("CCc1ccccc1", "CCC(C)=O"),
        ("CCc1ccccc1", "CC(=O)C(C)(C)C"),
        ("CCc1ccccc1", "COC(C)=O"),
        ("CCc1ccccc1", "CCCCC(=O)OC"),
        ("CCc1ccccc1", "CCN"),
        ("CCc1ccccc1", "CCCN"),
        ("CCc1ccccc1", "CN(C)C"),
        ("CCc1ccccc1", "Oc1ccc(Br)cc1"),
        ("CCCCCCC", "CO"),
        ("CCCCCCC", "CCCCCO"),
        ("CCCCCCC", "COc1ccccc1"),
        ("CCCCCCC", "O=Cc1ccccc1"),
        ("CCCCCCC", "CCC(C)=O"),
        ("CCCCCCC", "CC(=O)C(C)(C)C"),
        ("CCCCCCC", "COC(C)=O"),
        ("CCCCCCC", "CCCCC(=O)OC"),
        ("CCCCCCC", "CCN"),
        ("CCCCCCC", "c1ccncc1"),
        ("CCCCCCC", "Clc1ccc(Cl)cc1"),
        ("CCCCCCC", "Brc1ccccc1"),
        ("CCCCCCC", "Brc1ccc(Br)cc1"),
        ("CCCCCCC", "COP(=O)(OC)OC"),
        ("CCCCCCC", "CCOP(=O)(OCC)OCC"),
        ("CCCCCCC", "c1ccsc1"),
        ("CCCCCCC", "Clc1ccccc1-c1ccccc1Cl"),
        ("CCCCCCC", "CC(=O)N(C)C"),
        ("CCCCCCC", "NC(=O)c1ccccc1"),
        ("CCCCCCC", "Cc1ccccc1N"),
        ("CCCCCCC", "CN1CCCC1=O"),
        ("CCCCCCC", "O=c1[nH]cc(Br)c(=O)[nH]1"),
        ("CCCCCC", "CCCCCCCC"),
        ("CCCCCC", "CO"),
        ("CCCCCC", "CCO"),
        ("CCCCCC", "C1COCCO1"),
        ("CCCCCC", "O=Cc1ccccc1"),
        ("CCCCCC", "CC(=O)C(C)(C)C"),
        ("CCCCCC", "CC(=O)c1ccccc1"),
        ("CCCCCC", "COC(C)=O"),
        ("CCCCCC", "CCCCC(=O)OC"),
        ("CCCCCC", "CCN"),
        ("CCCCCC", "c1ccncc1"),
        ("CCCCCC", "O=Cc1ccc(O)cc1"),
        ("CCCCCC", "Clc1ccccc1"),
        ("CCCCCC", "Clc1ccc(Cl)cc1"),
        ("CCCCCC", "Brc1ccccc1"),
        ("CCCCCC", "Oc1ccc(Br)cc1"),
        ("CCCCCC", "COP(=O)(OC)OC"),
        ("CCCCCC", "CCOP(=O)(OCC)OCC"),
        ("CCCCCC", "N#Cc1cc(Br)c(O)c(Br)c1"),
        ("CCCCCC", "NC(=O)c1ccccc1"),
        ("CCCCCC", "Cc1ccc(N)cc1"),
        ("CCCCCCO", "c1ccccc1"),
        ("CCCCCCO", "Oc1ccccc1"),
        ("CCCCCCO", "C=O"),
        ("CCCCCCO", "CCN"),
        ("CCCCCCO", "CCCCN"),
        ("CCCCCCO", "Oc1ccc(Br)cc1"),
        ("CC(C)CO", "CCCC=O"),
        ("CC(C)CO", "CCOC(C)=O"),
        ("CC(C)CO", "CNC"),
        ("CC(C)CO", "CN(C)C"),
        ("CC(C)CO", "C1CNCCN1"),
        ("CC(C)CO", "c1ccncc1"),
        ("CC(C)CO", "C1CCNCC1"),
        ("CC(C)CO", "CN"),
        ("CCCCCC(C)C", "CCCCC"),
        ("CCCCCC(C)C", "CCCCCC"),
        ("CCCCCC(C)C", "C=CC"),
        ("CCCCCC(C)C", "C=CCC"),
        ("CCCCCC(C)C", "c1ccccc1"),
        ("CCCCCC(C)C", "CCO"),
        ("CCCCCC(C)C", "CCCCCCO"),
        ("CCCCCC(C)C", "Cc1ccccc1O"),
        ("CCCCCC(C)C", "C1COCCO1"),
        ("CCCCCC(C)C", "CCCC=O"),
        ("CCCCCC(C)C", "CCCCC=O"),
        ("CCCCCC(C)C", "CC(C)=O"),
        ("CCCCCC(C)C", "CCC(C)=O"),
        ("CCCCCC(C)C", "CCCCN"),
        ("CCCCCC(C)C", "Nc1ccccc1"),
        ("CCCCCC(C)C", "CCS"),
        ("CCCCCC(C)C", "CCCS"),
        ("CCCCCC(C)C", "COC(=O)c1ccccc1"),
        ("CC(C)O", "CCCCCCCC"),
        ("CC(C)O", "Cc1ccccc1"),
        ("CC(C)O", "CCO"),
        ("CC(C)O", "C1COCCO1"),
        ("CC(C)O", "CCC(C)=O"),
        ("CC(C)c1ccccc1", "CCO"),
        ("CC(C)c1ccccc1", "CC(C)=O"),
        ("CC(C)c1ccccc1", "CC(=O)C(C)(C)C"),
        ("CC(C)c1ccccc1", "CCOC(C)=O"),
        ("CC(C)c1ccccc1", "CCCCCC(=O)OC"),
        ("CC(C)c1ccccc1", "CCCCN"),
        ("Cc1ccccc1C(C)C", "CCC(=O)OC"),
        ("Cc1ccccc1C(C)C", "CCCCCOC(C)=O"),
        ("Cc1ccccc1C(C)C", "CCCCN"),
        ("CNC=O", "CCCCCCCC"),
        ("CNC=O", "Cc1ccccc1"),
        ("CNC=O", "CCO"),
        ("CNC=O", "C1COCCO1"),
        ("CNC=O", "CCC(C)=O"),
        ("CCCCCCCC", "CO"),
        ("CCCCCCCC", "CCCO"),
        ("CCCCCCCC", "CCC(C)=O"),
        ("CCCCCCCC", "CC(=O)C(C)(C)C"),
        ("CCCCCCCC", "COC(C)=O"),
        ("CCCCCCCC", "CCCCCC(=O)OC"),
        ("CCCCCCCC", "CCN"),
        ("CCCCCCCC", "CCNCC"),
        ("CCCCCCCC", "Cc1cnccn1"),
        ("CCCCCCCC", "Nc1ccccc1"),
        ("CCCCCCCC", "CCc1cnccn1"),
        ("CCCCCCCC", "Cc1ccccc1N"),
        ("CCCCC", "CO"),
        ("CCCCC", "CCCO"),
        ("CCCCC", "Oc1ccccc1"),
        ("CCCCC", "CCCC(C)=O"),
        ("CCCCC", "CC(=O)C(C)(C)C"),
        ("CCCCC", "COC(C)=O"),
        ("CCCCC", "CCCCCC(=O)OC"),
        ("CCCCC", "CCN"),
        ("CCCCC", "CCCN"),
        ("CCCCC", "Nc1ccccc1"),
        ("CCCCCO", "c1ccccc1"),
        ("CCCCCO", "Cc1ccccc1O"),
        ("CCCCCO", "C=O"),
        ("CCCCCO", "CCCCN"),
        ("CCCCCO", "CCNCC"),
        ("CCCCCO", "Oc1ccc(Br)cc1"),
        ("CCCCCO", "CN"),
        ("CCCO", "CCCCCCCC"),
        ("CCCO", "Cc1ccccc1"),
        ("CCCO", "CCO"),
        ("CCCO", "C1COCCO1"),
        ("CCCO", "CCC(C)=O"),
        ("c1ccncc1", "CCCCCCCC"),
        ("c1ccncc1", "Cc1ccccc1"),
        ("c1ccncc1", "CCO"),
        ("c1ccncc1", "C1COCCO1"),
        ("c1ccncc1", "CCC(C)=O"),
        ("C1CCOC1", "CCCCCCCC"),
        ("C1CCOC1", "Cc1ccccc1"),
        ("C1CCOC1", "CCO"),
        ("C1CCOC1", "C1COCCO1"),
        ("C1CCOC1", "CCC(C)=O"),
        ("c1ccc2c(c1)CCCC2", "CCO"),
        ("c1ccc2c(c1)CCCC2", "CC(=O)C(C)(C)C"),
        ("c1ccc2c(c1)CCCC2", "CCCCCC(C)=O"),
        ("Cc1ccccc1", "CCCCCCCC"),
        ("Cc1ccccc1", "CO"),
        ("Cc1ccccc1", "CCO"),
        ("Cc1ccccc1", "C1COCCO1"),
        ("Cc1ccccc1", "CC(C)=O"),
        ("Cc1ccccc1", "CC(=O)C(C)(C)C"),
        ("Cc1ccccc1", "COC(C)=O"),
        ("Cc1ccccc1", "CNC"),
        ("Cc1ccccc1", "CN(C)C"),
        ("Cc1ccccc1", "CCCNCCC"),
        ("Cc1ccccc1", "c1ccncc1"),
        ("Cc1ccccc1", "Nc1ccccc1"),
        ("Cc1ccccc1", "Oc1ccc(Br)cc1"),
        ("Cc1ccccc1", "CN"),
        ("Cc1ccccc1", "O=C1CCCO1"),
        ("CCCCOP(=O)(OCCCC)OCCCC", "CO"),
        ("CCCCOP(=O)(OCCCC)OCCCC", "CCCCCO"),
        ("CCCCOP(=O)(OCCCC)OCCCC", "CCN"),
        ("CCCCOP(=O)(OCCCC)OCCCC", "CCCN"),
        ("CCCCOP(=O)(OCCCC)OCCCC", "Nc1ccccc1"),
        ("CCCCOP(=O)(OCCCC)OCCCC", "COCCO"),
        ("CCN(CC)CC", "CCCCCCCC"),
        ("CCN(CC)CC", "Cc1ccccc1"),
        ("CCN(CC)CC", "CCO"),
        ("CCN(CC)CC", "C1COCCO1"),
        ("CCN(CC)CC", "CCC(C)=O"),
        ("Cc1ccccc1C", "CCCCCCCC"),
        ("Cc1ccccc1C", "CO"),
        ("Cc1ccccc1C", "CCO"),
        ("Cc1ccccc1C", "Cc1ccc(O)cc1"),
        ("Cc1ccccc1C", "C1COCCO1"),
        ("Cc1ccccc1C", "CC(C)=O"),
        ("Cc1ccccc1C", "CC(=O)C(C)(C)C"),
        ("Cc1ccccc1C", "COC(C)=O"),
        ("Cc1ccccc1C", "CCCCC(=O)OC"),
        ("Cc1ccccc1C", "CNC"),
        ("Cc1ccccc1C", "CN(C)C"),
        ("Cc1ccccc1C", "c1ccncc1"),
        ("Cc1ccccc1C", "Nc1ccccc1"),
        ("Cc1ccccc1C", "Oc1ccc(Br)cc1"),
        ("Cc1ccccc1C", "C1CCNCC1"),
        ("Cc1ccccc1C", "CN"),
    ],
)
def test_nonwater_solvent_long(solvent_smiles, solute_smiles):
    solvent_offmol = Molecule.from_smiles(solvent_smiles)
    solvent_offmol.generate_conformers(n_conformers=1)
    solvent_offmol.assign_partial_charges(partial_charge_method="gasteiger")
    solvent_smc = SmallMoleculeComponent.from_openff(solvent_offmol)

    ligand_offmol = Molecule.from_smiles(solute_smiles)
    ligand_offmol.generate_conformers(n_conformers=1)
    ligand_offmol.assign_partial_charges(partial_charge_method="gasteiger")
    ligand_smc = SmallMoleculeComponent.from_openff(ligand_offmol)

    interchange, _ = interchange_system_creation(
        ffsettings=InterchangeFFSettings(
            forcefields=[
                "openff-2.0.0.offxml",
            ]
        ),
        solvation_settings=PackmolSolvationSettings(
            solvent_padding=None,
            number_of_solvent_molecules=1000,
            assign_solvent_charges=True,
            target_density=0.715 * unit.grams / unit.mL,
        ),
        smc_components={ligand_smc: ligand_offmol},
        protein_component=None,
        solvent_component=ExtendedSolventComponent(solvent_molecule=solvent_smc),
        solvent_offmol=solvent_offmol,
    )

    if solvent_smiles == solute_smiles:
        # solvent == ligand
        assert solvent_offmol.is_isomorphic_with(ligand_offmol)
        assert interchange.topology.n_unique_molecules == 1
    else:
        assert interchange.topology.n_unique_molecules == 2
        assert solvent_offmol.is_isomorphic_with(list(interchange.topology.unique_molecules)[1])
    assert interchange.topology.n_molecules == 1001


def test_split_combine_noprotein_error(water_off, protein_ff_settings):
    with pytest.raises(ValueError, match="without a protein is a bad idea"):
        _ = _protein_split_combine_interchange(
            input_topology=Topology.from_molecules(water_off),
            charge_from_molecules=None,
            protein_component=None,
            ffsettings=protein_ff_settings,
        )


def test_split_combine_parameters(
    thrombin_protein_component, thrombin_protein_offtop, thrombin_ligands_charged
):
    # Prep ligand 6a
    l_6a = thrombin_ligands_charged["6a"]
    l_6a_off = l_6a.to_openff()
    l_6a_off.properties["key"] = str(l_6a.key)

    # Prep ligand 6b
    l_6b = thrombin_ligands_charged["6b"]
    l_6b_off = l_6b.to_openff()
    l_6b_off.properties["key"] = str(l_6b.key)

    # A water molecule
    water = Molecule.from_smiles("O")
    water.generate_conformers(n_conformers=1)
    water.properties["key"] = "foo"

    # Protein
    # Note: don't have to assign key to thrombin_protein_offtop
    # that's already done in the fixture
    protein_mols = [m for m in thrombin_protein_offtop.molecules]

    # stateA topology
    stateA_top = Topology.from_molecules(protein_mols + [l_6a_off, water])

    # ffsettings
    ffsettings = InterchangeFFSettings(
        forcefields=[
            "openff-2.0.0.offxml",
            "ff14sb_off_impropers_0.0.4.offxml",
            "opc3.offxml",
        ],
    )

    # force fields
    proteinff = ForceField("ff14sb_off_impropers_0.0.4.offxml", "opc3.offxml")
    proteinff_smirks = [parameter.smirks for parameter in proteinff["vdW"].parameters]
    ligandff = ForceField("openff-2.0.0.offxml")
    ligandff_smirks = [parameter.smirks for parameter in ligandff["vdW"].parameters]
    waterff = ForceField("opc3.offxml")
    waterff_smirks = [parameter.smirks for parameter in waterff["vdW"].parameters]

    interA = _protein_split_combine_interchange(
        stateA_top,
        charge_from_molecules=[l_6a_off],
        protein_component=thrombin_protein_component,
        ffsettings=ffsettings,
    )

    # Work around https://github.com/openforcefield/openff-interchange/issues/1337
    # this is just dummy parameters which should only match water and do nothing physically
    proteinff["Bonds"].add_parameter(
        {
            "smirks": "[#1:1]-[#8X2H2+0:2]-[#1]",
            "id": "dummy_water_bond",
            "length": Quantity(0.9572, "angstrom"),
            "k": Quantity(0.0, "kilocalorie / angstrom ** 2 / mole"),  # dummy
        }
    )
    proteinff["Angles"].add_parameter(
        {
            "smirks": "[#1:1]-[#8X2H2+0:2]-[#1:3]",
            "id": "dummy_water_angle",
            "angle": Quantity(104.52, "degree"),
            "k": Quantity(0.0, "kilocalorie / mol /radian ** 2"),  # dummy
        }
    )

    protein_vdw = proteinff.create_interchange(thrombin_protein_offtop)[
        "vdW"
    ].get_system_parameters()
    ligand_vdw = ligandff.create_interchange(
        l_6a_off.to_topology(), charge_from_molecules=[l_6a_off]
    )["vdW"].get_system_parameters()

    found_vdw = interA["vdW"].get_system_parameters()

    protein_atoms = thrombin_protein_offtop.n_atoms
    np.testing.assert_equal(found_vdw[:protein_atoms, :], protein_vdw)

    np.testing.assert_equal(
        found_vdw[protein_atoms : (protein_atoms + l_6a_off.n_atoms), :],
        ligand_vdw,
    )

    # Probe the vdw parameters and make sure they belong to the right force field
    for key, val in interA["vdW"].key_map.items():
        if key.atom_indices[0] < thrombin_protein_offtop.n_atoms:
            assert val.id in proteinff_smirks
            if val.id not in ["[#1]-[#8X2H2+0:1]-[#1]", "[#1:1]-[#8X2H2+0]-[#1]"]:
                assert val.id not in ligandff_smirks
        elif key.atom_indices[0] < (thrombin_protein_offtop.n_atoms + l_6a_off.n_atoms):
            assert val.id not in proteinff_smirks
            assert val.id not in waterff_smirks
            assert val.id in ligandff_smirks
        else:
            assert val.id.strip("_DUPLICATE") in waterff_smirks

    # Now get a stateB interchange and do the same check
    interB = copy_interchange_with_replacement(
        interchange=interA,
        del_mol=l_6a_off,
        insert_mol=l_6b_off,
        ffsettings=ffsettings,
        charged_molecules=[l_6b_off],
        protein_component=thrombin_protein_component,
    )

    # Again probe the contents to make sure we applied the right force field
    for key, val in interB["vdW"].key_map.items():
        if key.atom_indices[0] < thrombin_protein_offtop.n_atoms:
            assert val.id in proteinff_smirks
            if val.id not in ["[#1]-[#8X2H2+0:1]-[#1]", "[#1:1]-[#8X2H2+0]-[#1]"]:
                assert val.id not in ligandff_smirks
        elif key.atom_indices[0] < (thrombin_protein_offtop.n_atoms + water.n_atoms):
            assert val.id.strip("_DUPLICATE") in waterff_smirks
        else:
            assert val.id not in proteinff_smirks
            assert val.id not in waterff_smirks
            assert val.id in ligandff_smirks


class BaseSystemTests:
    @pytest.fixture(scope="class")
    def omm_system(self, interchange_system):
        interchange, _ = interchange_system

        return interchange.to_openmm_system()

    @pytest.fixture(scope="class")
    def omm_topology(self, interchange_system):
        interchange, _ = interchange_system

        return interchange.to_openmm_topology(collate=True)

    @pytest.fixture(scope="class")
    def nonbonds(self, omm_system):
        return [f for f in omm_system.getForces() if isinstance(f, NonbondedForce)]


class TestVacuumUnamedBenzene(BaseSystemTests):
    """
    TODO: add test checking we get a warning on renaming
    """

    smc_comps = "smc_components_benzene_unnamed"
    resname = "AAA"
    nonbond_index = 0

    @pytest.fixture(scope="class")
    def interchange_system(self, request):
        smc_components = request.getfixturevalue(self.smc_comps)
        interchange, comp_resids = interchange_system_creation(
            ffsettings=InterchangeFFSettings(),
            solvation_settings=PackmolSolvationSettings(),
            smc_components=smc_components,
            protein_component=None,
            solvent_component=None,
            solvent_offmol=None,
        )

        return interchange, comp_resids

    @pytest.fixture(scope="class")
    def num_particles(self):
        return 12

    @pytest.fixture(scope="class")
    def num_residues(self):
        return 1

    @pytest.fixture(scope="class")
    def num_constraints(self):
        return 6

    @pytest.fixture(scope="class")
    def num_bonds(self):
        return 6

    @pytest.fixture(scope="class")
    def num_angles(self):
        return 18

    @pytest.fixture(scope="class")
    def num_dih(self):
        return 42

    def test_comp_resids(self, interchange_system, request):
        _, comp_resids = interchange_system

        assert len(comp_resids) == 1
        assert next(iter(comp_resids.values())) == 0
        assert next(iter(request.getfixturevalue(self.smc_comps))) in comp_resids

    def test_topology(self, omm_topology, num_residues, num_particles):
        residues = list(omm_topology.residues())
        assert len(residues) == num_residues
        assert len(list(omm_topology.atoms())) == num_particles
        assert residues[0].name == self.resname  # Expect auto-naming to AAA
        assert residues[0].index == residues[0].id == 0

    def test_system_basics(self, omm_system, num_particles, num_constraints):
        # Expected number of atoms
        assert omm_system.getNumParticles() == num_particles

        # 5 forces expected
        # Note: CMMotionRemover is there and needs to be removed in Protocol
        assert omm_system.getNumForces() == 5

        # 6 constraints, one for each hydrogen
        assert omm_system.getNumConstraints() == num_constraints

    def test_positions(self, interchange_system, num_particles):
        inter, _ = interchange_system
        assert len(to_openmm_positions(inter)) == num_particles

    def test_system_nonbonded(self, nonbonds):
        # One nonbonded force
        assert len(nonbonds) == 1

        # Gas phase should be nonbonded
        assert nonbonds[0].getNonbondedMethod() == self.nonbond_index
        assert nonbonds[0].getCutoffDistance() == to_openmm(0.9 * unit.nanometer)
        assert nonbonds[0].getSwitchingDistance() == to_openmm(0.8 * unit.nanometer)

    def test_system_bonds(self, omm_system, num_bonds):
        bond = [f for f in omm_system.getForces() if isinstance(f, HarmonicBondForce)]

        # One bond forces
        assert len(bond) == 1

        # 6 bonds
        assert bond[0].getNumBonds() == num_bonds

    def test_system_angles(self, omm_system, num_angles):
        angle = [f for f in omm_system.getForces() if isinstance(f, HarmonicAngleForce)]

        # One bond forces
        assert len(angle) == 1

        # 18 angles
        assert angle[0].getNumAngles() == num_angles

    def test_system_dihedrals(self, omm_system, num_dih):
        dih = [f for f in omm_system.getForces() if isinstance(f, PeriodicTorsionForce)]

        # One bond forces
        assert len(dih) == 1

        # 42 angles
        assert dih[0].getNumTorsions() == num_dih


class TestVacuumNamedBenzene(TestVacuumUnamedBenzene):
    smc_comps = "smc_components_benzene_named"
    resname = "BNZ"


class TestSolventOPC3UnamedBenzene(TestVacuumUnamedBenzene):
    smc_comps = "smc_components_benzene_unnamed"
    resname = "AAA"
    nonbond_index = 4
    solvent_resname = "SOL"

    @pytest.fixture(scope="class")
    def interchange_system(self, water_off, request):
        smc_components = request.getfixturevalue(self.smc_comps)
        interchange, comp_resids = interchange_system_creation(
            ffsettings=InterchangeFFSettings(
                forcefields=["openff-2.0.0.offxml", "opc3.offxml"],
            ),
            solvation_settings=PackmolSolvationSettings(),
            smc_components=smc_components,
            protein_component=None,
            solvent_component=ExtendedSolventComponent(),
            solvent_offmol=water_off,
        )

        return interchange, comp_resids

    @pytest.fixture(scope="class")
    def num_residues(self, omm_topology):
        return omm_topology.getNumResidues()

    @pytest.fixture(scope="class")
    def num_waters(self, num_residues):
        return num_residues - 1

    @pytest.fixture(scope="class")
    def num_particles(self, num_waters):
        return 12 + (3 * num_waters)

    @pytest.fixture(scope="class")
    def num_constraints(self, num_waters):
        return 6 + (3 * num_waters)

    def test_comp_resids(self, interchange_system, request, num_residues):
        _, comp_resids = interchange_system

        assert len(comp_resids) == 2
        assert list(comp_resids)[0] == next(iter(request.getfixturevalue(self.smc_comps)))
        assert list(comp_resids)[1] == ExtendedSolventComponent()
        assert_equal(list(comp_resids.values())[0], [0])
        assert_equal(list(comp_resids.values())[1], [i for i in range(1, num_residues)])

    def test_solvent_resnames(self, omm_topology):
        for i, res in enumerate(list(omm_topology.residues())[1:]):
            assert res.index == res.id == i + 1
            assert res.name == self.solvent_resname

    def test_solvent_nonbond_parameters(self, nonbonds, num_particles, num_waters):
        for index in range(12, num_particles - num_waters, 3):
            # oxygen
            c, s, e = nonbonds[0].getParticleParameters(index)
            assert c.value_in_unit(omm_unit.elementary_charge) == pytest.approx(-0.89517)
            assert e.value_in_unit(omm_unit.kilojoule_per_mole) == pytest.approx(0.683690704)
            assert s.value_in_unit(omm_unit.angstrom) == pytest.approx(3.1742703509365926)

            # hydrogens
            c1, s1, e1 = nonbonds[0].getParticleParameters(index + 1)
            c2, s2, e2 = nonbonds[0].getParticleParameters(index + 2)
            assert c1.value_in_unit(omm_unit.elementary_charge) == pytest.approx(0.447585)
            assert e1.value_in_unit(omm_unit.kilocalorie_per_mole) == 0
            assert s1.value_in_unit(omm_unit.angstrom) == pytest.approx(1.7817974)
            assert c1 == c2
            assert s1 == s2
            assert e2 == e2


class TestSolventOPC3NamedChargedButUnAssignedBenzene(TestSolventOPC3UnamedBenzene):
    """
    OPC3 model
    Charged water offmol
    Charges not assigned
    """

    solvent_resname = "HOH"

    @pytest.fixture(scope="class")
    def interchange_system(self, water_off_named_charged, request):
        smc_components = request.getfixturevalue(self.smc_comps)
        interchange, comp_resids = interchange_system_creation(
            ffsettings=InterchangeFFSettings(
                forcefields=["openff-2.0.0.offxml", "opc3.offxml"],
            ),
            solvation_settings=PackmolSolvationSettings(),
            smc_components=smc_components,
            protein_component=None,
            solvent_component=ExtendedSolventComponent(),
            solvent_offmol=water_off_named_charged,
        )

        return interchange, comp_resids


class TestSolventOPC3NamedChargedAssignedBenzene(TestSolventOPC3UnamedBenzene):
    """
    OPC3 model
    Charged water offmol
    Charges assigned
    """

    solvent_resname = "HOH"

    @pytest.fixture(scope="class")
    def interchange_system(self, water_off_named_charged, request):
        smc_components = request.getfixturevalue(self.smc_comps)
        interchange, comp_resids = interchange_system_creation(
            ffsettings=InterchangeFFSettings(
                forcefields=["openff-2.0.0.offxml", "opc3.offxml"],
            ),
            solvation_settings=PackmolSolvationSettings(
                assign_solvent_charges=True,
            ),
            smc_components=smc_components,
            protein_component=None,
            solvent_component=ExtendedSolventComponent(),
            solvent_offmol=water_off_named_charged,
        )

        return interchange, comp_resids

    def test_solvent_nonbond_parameters(
        self, nonbonds, num_particles, num_waters, water_off_named_charged
    ):
        for index in range(12, num_particles - num_waters, 3):
            # oxygen
            c, s, e = nonbonds[0].getParticleParameters(index)
            assert from_openmm(c) == water_off_named_charged.partial_charges[0]
            assert e.value_in_unit(omm_unit.kilojoule_per_mole) == pytest.approx(0.683690704)
            assert s.value_in_unit(omm_unit.angstrom) == pytest.approx(3.1742703509365926)

            # hydrogens
            c1, s1, e1 = nonbonds[0].getParticleParameters(index + 1)
            c2, s2, e2 = nonbonds[0].getParticleParameters(index + 2)
            assert from_openmm(c1) == water_off_named_charged.partial_charges[1]
            assert e1.value_in_unit(omm_unit.kilocalorie_per_mole) == 0
            assert s1.value_in_unit(omm_unit.angstrom) == pytest.approx(1.7817974)
            assert c1 == c2
            assert s1 == s2
            assert e2 == e2


class TestSolventOPC3AceticAcidNeutralize(TestSolventOPC3UnamedBenzene):
    smc_comps = "smc_components_acetic_acid"

    @pytest.fixture(scope="class")
    def interchange_system(self, water_off, request):
        smc_components = request.getfixturevalue(self.smc_comps)
        interchange, comp_resids = interchange_system_creation(
            ffsettings=InterchangeFFSettings(
                forcefields=["openff-2.0.0.offxml", "opc3.offxml"],
            ),
            solvation_settings=PackmolSolvationSettings(
                solvent_padding=2 * unit.nm,
            ),
            smc_components=smc_components,
            protein_component=None,
            solvent_component=ExtendedSolventComponent(
                neutralize=True,
                ion_concentration=0.15 * unit.molar,
            ),
            solvent_offmol=water_off,
        )

        return interchange, comp_resids

    @pytest.fixture(scope="class")
    def num_bonds(self):
        return 3

    @pytest.fixture(scope="class")
    def num_angles(self):
        return 9

    @pytest.fixture(scope="class")
    def num_dih(self):
        return 15

    @pytest.fixture(scope="class")
    def num_pos_ions(self):
        return 7

    @pytest.fixture(scope="class")
    def num_neg_ions(self):
        return 6

    @pytest.fixture(scope="class")
    def num_waters(self, num_residues, num_pos_ions, num_neg_ions):
        return num_residues - (1 + num_neg_ions + num_pos_ions)

    @pytest.fixture(scope="class")
    def num_particles(self, num_waters, num_neg_ions, num_pos_ions):
        return 7 + (3 * num_waters) + num_neg_ions + num_pos_ions

    @pytest.fixture(scope="class")
    def num_constraints(self, num_waters):
        return 3 + (3 * num_waters)

    def test_comp_resids(self, interchange_system, request, num_residues):
        _, comp_resids = interchange_system

        assert len(comp_resids) == 2
        assert list(comp_resids)[0] == next(iter(request.getfixturevalue(self.smc_comps)))
        assert list(comp_resids)[1] == ExtendedSolventComponent(
            neutralize=True,
            ion_concentration=0.15 * unit.molar,
        )
        assert_equal(list(comp_resids.values())[0], [0])
        assert_equal(list(comp_resids.values())[1], [i for i in range(1, num_residues)])

    def test_solvent_resnames(self, omm_topology):
        for i, res in enumerate(list(omm_topology.residues())[1:]):
            assert res.index == res.id == i + 1
            assert res.name in [self.solvent_resname, "NA+", "CL-"]

    def test_solvent_nonbond_parameters(self, nonbonds, num_particles, num_waters):
        for index in range(7, 7 + num_waters, 3):
            # oxygen
            c, s, e = nonbonds[0].getParticleParameters(index)
            assert c.value_in_unit(omm_unit.elementary_charge) == pytest.approx(-0.89517)
            assert e.value_in_unit(omm_unit.kilojoule_per_mole) == pytest.approx(0.683690704)
            assert s.value_in_unit(omm_unit.angstrom) == pytest.approx(3.1742703509365926)

            # hydrogens
            c1, s1, e1 = nonbonds[0].getParticleParameters(index + 1)
            c2, s2, e2 = nonbonds[0].getParticleParameters(index + 2)
            assert c1.value_in_unit(omm_unit.elementary_charge) == pytest.approx(0.447585)
            assert e1.value_in_unit(omm_unit.kilocalorie_per_mole) == 0
            assert s1.value_in_unit(omm_unit.angstrom) == pytest.approx(1.7817974)
            assert c1 == c2
            assert s1 == s2
            assert e2 == e2

        for index in range(7 + (num_waters * 3), num_particles):
            c, s, e = nonbonds[0].getParticleParameters(index)

            charge = from_openmm(c)
            assert abs(charge.m) == 1

            if charge.m == 1:
                assert e.value_in_unit(omm_unit.kilojoule_per_mole) == pytest.approx(0.1260287744)
                assert s.value_in_unit(omm_unit.angstroms) == pytest.approx(2.617460434)

            if charge.m == -1:
                assert e.value_in_unit(omm_unit.kilojoule_per_mole) == pytest.approx(2.68724395648)
                assert s.value_in_unit(omm_unit.angstroms) == pytest.approx(4.108824888)

    def test_system_total_charge(self, nonbonds, omm_system):
        total_charge = 0.0
        for i in range(omm_system.getNumParticles()):
            c, s, e = nonbonds[0].getParticleParameters(i)
            total_charge += from_openmm(c).m

        assert total_charge == pytest.approx(0)


class TestSolventOPCNamedBenzene(TestSolventOPC3UnamedBenzene):
    smc_comps = "smc_components_benzene_named"
    resname = "BNZ"
    nonbond_index = 4

    @pytest.fixture(scope="class")
    def interchange_system(self, water_off, request):
        smc_components = request.getfixturevalue(self.smc_comps)
        interchange, comp_resids = interchange_system_creation(
            ffsettings=InterchangeFFSettings(
                forcefields=["openff-2.0.0.offxml", "opc.offxml"],
            ),
            solvation_settings=PackmolSolvationSettings(),
            smc_components=smc_components,
            protein_component=None,
            solvent_component=ExtendedSolventComponent(),
            solvent_offmol=water_off,
        )

        return interchange, comp_resids

    @pytest.fixture(scope="class")
    def num_particles(self, num_waters):
        return 12 + (4 * num_waters)

    def test_solvent_nonbond_parameters(self, nonbonds, num_particles, num_waters):
        for index in range(12, num_particles - num_waters, 3):
            # oxygen
            c, s, e = nonbonds[0].getParticleParameters(index)
            assert c.value_in_unit(omm_unit.elementary_charge) == 0
            assert e.value_in_unit(omm_unit.kilojoule_per_mole) == pytest.approx(0.890358601)
            assert s.value_in_unit(omm_unit.angstrom) == pytest.approx(3.16655208)

            # hydrogens
            c1, s1, e1 = nonbonds[0].getParticleParameters(index + 1)
            c2, s2, e2 = nonbonds[0].getParticleParameters(index + 2)
            assert c1.value_in_unit(omm_unit.elementary_charge) == 0.679142
            assert e1.value_in_unit(omm_unit.kilocalorie_per_mole) == 0
            assert s1.value_in_unit(omm_unit.angstrom) == pytest.approx(1.7817974)
            assert c1 == c2
            assert s1 == s2
            assert e2 == e2

    def test_comp_resids(self, interchange_system, request, num_residues, num_waters):
        """
        Must redefine to account for virtual site residues in comp_resids
        """
        _, comp_resids = interchange_system

        assert len(comp_resids) == 2
        assert list(comp_resids)[0] == next(iter(request.getfixturevalue(self.smc_comps)))
        assert list(comp_resids)[1] == ExtendedSolventComponent()
        assert_equal(list(comp_resids.values())[0], [0])
        # We have 2 residues per water instead of 1 (1 extra for the vsite)
        assert_equal(list(comp_resids.values())[1], [i for i in range(1, (num_waters * 2) + 1)])

    def test_virtual_sites(self, omm_system, num_waters, num_particles, nonbonds):
        for index in range(num_particles, num_particles - num_waters, -1):
            assert omm_system.isVirtualSite(index - 1)
            c, s, e = nonbonds[0].getParticleParameters(index - 1)
            assert from_openmm(c) == -0.679142 * 2 * unit.elementary_charge
            assert from_openmm(e) == 0 * unit.kilocalorie_per_mole
            assert from_openmm(s) * 2 ** (1 / 6) / 2.0 == 1 * unit.angstrom


class TestComplexOPC3(TestSolventOPC3UnamedBenzene):
    smc_comps = "smc_components_benzene_named"
    protein_comp = "T4_protein_component"
    resname = "BNZ"
    nonbond_index = 4

    @pytest.fixture(scope="class")
    def interchange_system(self, water_off, request):
        smc_components = request.getfixturevalue(self.smc_comps)
        protein_component = request.getfixturevalue(self.protein_comp)
        interchange, comp_resids = interchange_system_creation(
            ffsettings=InterchangeFFSettings(
                forcefields=[
                    "openff-2.0.0.offxml",
                    "ff14sb_off_impropers_0.0.4.offxml",
                    "opc3.offxml",
                ],
            ),
            solvation_settings=PackmolSolvationSettings(target_density=0.3 * unit.grams / unit.mL),
            smc_components=smc_components,
            protein_component=protein_component,
            solvent_component=SolventComponent(),
            solvent_offmol=water_off,
        )

        return interchange, comp_resids

    @pytest.fixture(scope="class")
    def num_bonds(self):
        return 1320

    @pytest.fixture(scope="class")
    def num_angles(self):
        return 4769

    @pytest.fixture(scope="class")
    def num_dih(self):
        return 12701

    @pytest.fixture(scope="class")
    def num_pos_ions(self):
        return 9

    @pytest.fixture(scope="class")
    def num_neg_ions(self):
        return 18

    @pytest.fixture(scope="class")
    def num_waters(self, num_residues, num_pos_ions, num_neg_ions):
        return num_residues - (1 + 164 + num_neg_ions + num_pos_ions)

    @pytest.fixture(scope="class")
    def num_protein_atoms(self):
        return 2613

    @pytest.fixture(scope="class")
    def num_particles(self, num_waters, num_neg_ions, num_pos_ions):
        return 12 + 2613 + (3 * num_waters) + num_neg_ions + num_pos_ions

    @pytest.fixture(scope="class")
    def num_constraints(self, num_waters):
        return 6 + 1319 + (3 * num_waters)

    def test_topology(self, omm_topology, num_residues, num_particles):
        residues = list(omm_topology.residues())
        assert len(residues) == num_residues
        assert len(list(omm_topology.atoms())) == num_particles
        # protein
        assert residues[0].name == "ACE"  # Expect first protein residue
        assert residues[0].index == int(residues[0].id) - 1 == 0
        assert all([r.chain.id == "A" for r in residues[:164]])
        assert all([r.chain.index == 0 for r in residues[:164]])
        # ligand
        assert residues[164].name == self.resname  # Expect auto-named to AAA
        assert residues[164].index == 164
        assert residues[164].id == 0
        assert residues[164].chain.id == "B"
        assert residues[164].chain.index == 1
        # solvent
        assert residues[165].name == "SOL"
        assert residues[165].index == 165
        assert residues[165].id == 2
        assert all([r.chain.id == "C" for r in residues[165:]])
        assert all([r.chain.index == i + 2 for i, r in enumerate(residues[165:])])

    def test_system_total_charge(self, nonbonds, omm_system):
        total_charge = 0.0
        for i in range(omm_system.getNumParticles()):
            c, s, e = nonbonds[0].getParticleParameters(i)
            total_charge += from_openmm(c).m

        assert total_charge == pytest.approx(0, abs=1e-6)

    def test_comp_resids(self, interchange_system, request, num_residues):
        _, comp_resids = interchange_system

        assert len(comp_resids) == 3
        assert list(comp_resids)[0] == request.getfixturevalue(self.protein_comp)
        assert list(comp_resids)[1] == next(iter(request.getfixturevalue(self.smc_comps)))
        assert list(comp_resids)[2] == SolventComponent()

        # 164 residues in T4 lysozyme
        assert_equal(list(comp_resids.values())[0], [i for i in range(164)])
        # ligand
        assert_equal(list(comp_resids.values())[1], [164])
        # solvent
        assert_equal(list(comp_resids.values())[2], [i for i in range(165, num_residues)])

    def test_solvent_resnames(self, omm_topology, num_pos_ions, num_neg_ions, num_waters):
        counts = {
            self.solvent_resname: 0,
            "NA+": 0,
            "CL-": 0,
        }
        for i, res in enumerate(list(omm_topology.residues())[165:]):
            assert res.index == i + 165
            assert res.id == i + 2  # 2 because solute + protein first
            counts[res.name] += 1

        assert counts[self.solvent_resname] == num_waters
        assert counts["NA+"] == num_pos_ions
        assert counts["CL-"] == num_neg_ions

    def test_solvent_nonbond_parameters(self, nonbonds, num_protein_atoms, num_waters):
        solute_count = 12 + num_protein_atoms
        for index in range(solute_count, solute_count + num_waters, 3):
            # oxygen
            c, s, e = nonbonds[0].getParticleParameters(index)
            assert c.value_in_unit(omm_unit.elementary_charge) == pytest.approx(-0.89517)
            assert e.value_in_unit(omm_unit.kilojoule_per_mole) == pytest.approx(0.683690704)
            assert s.value_in_unit(omm_unit.angstrom) == pytest.approx(3.1742703509365926)

            # hydrogens
            c1, s1, e1 = nonbonds[0].getParticleParameters(index + 1)
            c2, s2, e2 = nonbonds[0].getParticleParameters(index + 2)
            assert c1.value_in_unit(omm_unit.elementary_charge) == pytest.approx(0.447585)
            assert e1.value_in_unit(omm_unit.kilocalorie_per_mole) == 0
            assert s1.value_in_unit(omm_unit.angstrom) == pytest.approx(1.7817974)
            assert c1 == c2
            assert s1 == s2
            assert e2 == e2


@pytest.mark.slow
class TestOpenMMSolvationComplexOPC3(TestComplexOPC3):
    smc_comps = "smc_components_benzene_named"
    protein_comp = "T4_protein_component"
    resname = "BNZ"
    nonbond_index = 4

    @pytest.fixture(scope="class")
    def interchange_system(self, water_off, request):
        smc_components = request.getfixturevalue(self.smc_comps)
        protein_component = request.getfixturevalue(self.protein_comp)
        interchange, comp_resids = interchange_system_creation(
            ffsettings=InterchangeFFSettings(
                forcefields=[
                    "openff-2.0.0.offxml",
                    "ff14sb_off_impropers_0.0.4.offxml",
                    "opc3.offxml",
                ],
            ),
            solvation_settings=InterchangeOpenMMSolvationSettings(
                target_density=0.1 * unit.grams / unit.mL
            ),
            smc_components=smc_components,
            protein_component=protein_component,
            solvent_component=SolventComponent(),
            solvent_offmol=water_off,
        )

        return interchange, comp_resids

    @pytest.fixture(scope="class")
    def num_pos_ions(self):
        return 2

    @pytest.fixture(scope="class")
    def num_neg_ions(self):
        return 11

    @pytest.fixture(scope="class")
    def num_waters(self, num_residues, num_pos_ions, num_neg_ions):
        return num_residues - (1 + 164 + num_neg_ions + num_pos_ions)

    @pytest.fixture(scope="class")
    def num_protein_atoms(self):
        return 2613

    @pytest.fixture(scope="class")
    def num_particles(self, num_waters, num_neg_ions, num_pos_ions):
        return 12 + 2613 + (3 * num_waters) + num_neg_ions + num_pos_ions

    def test_topology(self, omm_topology, num_residues, num_waters, num_particles):
        residues = list(omm_topology.residues())
        assert len(residues) == num_residues
        assert len(list(omm_topology.atoms())) == num_particles
        # protein
        assert residues[0].name == "ACE"  # Expect first protein residue
        assert residues[0].index == int(residues[0].id) - 1 == 0
        assert all([r.chain.id == "A" for r in residues[:164]])
        assert all([r.chain.index == 0 for r in residues[:164]])
        # ligand
        assert residues[164].name == self.resname  # Expect auto-named to AAA
        assert residues[164].index == 164
        assert residues[164].id == 0
        assert residues[164].chain.id == "B"
        assert residues[164].chain.index == 1
        # solvent
        assert residues[165].name == "SOL"
        assert residues[165].index == 165
        assert residues[165].id == "166"  # TODO - check why this hasn't reset
        assert all([r.chain.id == "3" for r in residues[165 : 165 + num_waters]])
        assert all([r.chain.index == i + 2 for i, r in enumerate(residues[165:])])

    def test_solvent_resnames(self, omm_topology, num_pos_ions, num_neg_ions, num_waters):
        counts = {
            self.solvent_resname: 0,
            "NA+": 0,
            "CL-": 0,
        }
        for i, res in enumerate(list(omm_topology.residues())[165:]):
            assert res.index == i + 165
            # TODO: for some reason res.id isn't continuous, likeelihood is that
            # waters that get chosen to be turned into ions take the rersidue id
            # assert res.id == str(i + 165 + 1)  # starts at 1
            counts[res.name] += 1

        assert counts[self.solvent_resname] == num_waters
        assert counts["NA+"] == num_pos_ions
        assert counts["CL-"] == num_neg_ions


class TestComplexOPC3NumWaters(TestComplexOPC3):
    smc_comps = "smc_components_benzene_named"
    protein_comp = "T4_protein_component"
    resname = "BNZ"
    nonbond_index = 4

    @pytest.fixture(scope="class")
    def interchange_system(self, water_off, request):
        smc_components = request.getfixturevalue(self.smc_comps)
        protein_component = request.getfixturevalue(self.protein_comp)
        interchange, comp_resids = interchange_system_creation(
            ffsettings=InterchangeFFSettings(
                forcefields=[
                    "openff-2.0.0.offxml",
                    "ff14sb_off_impropers_0.0.4.offxml",
                    "opc3.offxml",
                ],
            ),
            solvation_settings=PackmolSolvationSettings(
                number_of_solvent_molecules=500,
                solvent_padding=None,
                target_density=0.715 * unit.grams / unit.mL,
            ),
            smc_components=smc_components,
            protein_component=protein_component,
            solvent_component=SolventComponent(),
            solvent_offmol=water_off,
        )

        return interchange, comp_resids

    @pytest.fixture(scope="class")
    def num_bonds(self):
        return 1320

    @pytest.fixture(scope="class")
    def num_angles(self):
        return 4769

    @pytest.fixture(scope="class")
    def num_dih(self):
        return 12701

    @pytest.fixture(scope="class")
    def num_pos_ions(self):
        return 0

    @pytest.fixture(scope="class")
    def num_neg_ions(self):
        return 9

    @pytest.fixture(scope="class")
    def num_waters(self):
        return 500

    @pytest.fixture(scope="class")
    def num_protein_atoms(self):
        return 2613

    @pytest.fixture(scope="class")
    def num_particles(self, num_waters, num_neg_ions, num_pos_ions):
        return 12 + 2613 + (3 * num_waters) + num_neg_ions + num_pos_ions

    @pytest.fixture(scope="class")
    def num_constraints(self, num_waters):
        return 6 + 1319 + (3 * num_waters)

    def test_system_total_charge(self, nonbonds, omm_system):
        total_charge = 0.0
        for i in range(omm_system.getNumParticles()):
            c, s, e = nonbonds[0].getParticleParameters(i)
            total_charge += from_openmm(c).m

        assert total_charge == pytest.approx(0, abs=1e-6)

    def test_solvent_resnames(self, omm_topology, num_pos_ions, num_neg_ions, num_waters):
        counts = {
            self.solvent_resname: 0,
            "NA+": 0,
            "CL-": 0,
        }
        for i, res in enumerate(list(omm_topology.residues())[165:]):
            assert res.index == i + 165
            assert res.id == i + 2  # 2 because solute + protein first
            counts[res.name] += 1

        assert counts[self.solvent_resname] == num_waters
        assert counts["NA+"] == num_pos_ions
        assert counts["CL-"] == num_neg_ions

    def test_solvent_nonbond_parameters(self, nonbonds, num_protein_atoms, num_waters):
        solute_count = 12 + num_protein_atoms
        for index in range(solute_count, solute_count + num_waters, 3):
            # oxygen
            c, s, e = nonbonds[0].getParticleParameters(index)
            assert c.value_in_unit(omm_unit.elementary_charge) == pytest.approx(-0.89517)
            assert e.value_in_unit(omm_unit.kilojoule_per_mole) == pytest.approx(0.683690704)
            assert s.value_in_unit(omm_unit.angstrom) == pytest.approx(3.1742703509365926)

            # hydrogens
            c1, s1, e1 = nonbonds[0].getParticleParameters(index + 1)
            c2, s2, e2 = nonbonds[0].getParticleParameters(index + 2)
            assert c1.value_in_unit(omm_unit.elementary_charge) == pytest.approx(0.447585)
            assert e1.value_in_unit(omm_unit.kilocalorie_per_mole) == 0
            assert s1.value_in_unit(omm_unit.angstrom) == pytest.approx(1.7817974)
            assert c1 == c2
            assert s1 == s2
            assert e2 == e2


class TestOpenMMSolvationCofactorOPC3NumWaters(TestOpenMMSolvationComplexOPC3):
    protein_comp = "eg5_protein"
    nonbond_index = 4

    @pytest.fixture(scope="class")
    def interchange_system(self, water_off, request, eg5_ligands, eg5_cofactor):
        ligand, _ = eg5_ligands
        smc_components = {
            ligand: ligand.to_openff(),
            eg5_cofactor: eg5_cofactor.to_openff(),
        }
        protein_component = request.getfixturevalue(self.protein_comp)
        interchange, comp_resids = interchange_system_creation(
            ffsettings=InterchangeFFSettings(
                forcefields=[
                    "openff-2.0.0.offxml",
                    "ff14sb_off_impropers_0.0.4.offxml",
                    "opc3.offxml",
                ],
            ),
            solvation_settings=InterchangeOpenMMSolvationSettings(
                number_of_solvent_molecules=100,
                solvent_padding=None,
                target_density=0.715 * unit.grams / unit.mL,
            ),
            smc_components=smc_components,
            protein_component=protein_component,
            solvent_component=SolventComponent(),
            solvent_offmol=water_off,
        )

        return interchange, comp_resids

    @pytest.fixture(scope="class")
    def num_bonds(self):
        return 2815

    @pytest.fixture(scope="class")
    def num_angles(self):
        return 10148

    @pytest.fixture(scope="class")
    def num_dih(self):
        return 27134

    @pytest.fixture(scope="class")
    def num_pos_ions(self):
        return 3

    @pytest.fixture(scope="class")
    def num_neg_ions(self):
        return 0

    @pytest.fixture(scope="class")
    def num_waters(self):
        return 97

    @pytest.fixture(scope="class")
    def num_protein_atoms(self):
        return 5495

    @pytest.fixture(scope="class")
    def num_particles(self, num_protein_atoms, num_waters, num_neg_ions, num_pos_ions):
        return 52 + 39 + num_protein_atoms + (3 * num_waters) + num_neg_ions + num_pos_ions

    @pytest.fixture(scope="class")
    def num_constraints(self):
        return 3106

    def test_topology(self, omm_topology, num_residues, num_waters, num_particles):
        residues = list(omm_topology.residues())
        assert len(residues) == num_residues
        assert len(list(omm_topology.atoms())) == num_particles
        # protein
        assert residues[0].name == "ACE"  # Expect first protein residue
        assert residues[0].index == int(residues[0].id) - 1 == 0
        assert all([r.chain.id == "A" for r in residues[:349]])
        assert all([r.chain.id == "B" for r in residues[349:356]])  # waters
        assert all([r.chain.index == 0 for r in residues[:349]])
        assert all([r.chain.index == i + 1 for i, r in enumerate(residues[349:356])])  # waters
        # ligand
        assert residues[356].name == "AAA"  # Expect auto-named to AAA
        assert residues[356].index == 356
        assert residues[356].id == 0
        assert residues[356].chain.id == "C"
        assert residues[356].chain.index == 8
        # cofactor
        assert residues[357].name == "AAB"  # Expect auto-named to AAB
        assert residues[357].index == 357
        assert residues[357].id == 1  # TODO: work out why this didn't reset
        assert residues[357].chain.id == "D"
        assert residues[357].chain.index == 9
        # solvent
        assert residues[358].name == "SOL"
        assert residues[358].index == 358
        assert residues[358].id == "359"  # TODO - check why this hasn't reset
        assert all([r.chain.id == "11" for r in residues[358 : 358 + num_waters]])
        assert all([r.chain.index == i + 10 for i, r in enumerate(residues[358:])])

    def test_solvent_resnames(self, omm_topology, num_pos_ions, num_neg_ions, num_waters):
        counts = {
            self.solvent_resname: 0,
            "NA+": 0,
            "CL-": 0,
        }
        for i, res in enumerate(list(omm_topology.residues())[358:]):
            assert res.index == i + 358
            counts[res.name] += 1

        assert counts[self.solvent_resname] == num_waters
        assert counts["NA+"] == num_pos_ions
        assert counts["CL-"] == num_neg_ions

    def test_solvent_nonbond_parameters(self, nonbonds, num_protein_atoms, num_waters):
        solute_count = 52 + 39 + num_protein_atoms
        for index in range(solute_count, solute_count + num_waters, 3):
            # oxygen
            c, s, e = nonbonds[0].getParticleParameters(index)
            assert c.value_in_unit(omm_unit.elementary_charge) == pytest.approx(-0.89517)
            assert e.value_in_unit(omm_unit.kilojoule_per_mole) == pytest.approx(0.683690704)
            assert s.value_in_unit(omm_unit.angstrom) == pytest.approx(3.1742703509365926)

            # hydrogens
            c1, s1, e1 = nonbonds[0].getParticleParameters(index + 1)
            c2, s2, e2 = nonbonds[0].getParticleParameters(index + 2)
            assert c1.value_in_unit(omm_unit.elementary_charge) == pytest.approx(0.447585)
            assert e1.value_in_unit(omm_unit.kilocalorie_per_mole) == 0
            assert s1.value_in_unit(omm_unit.angstrom) == pytest.approx(1.7817974)
            assert c1 == c2
            assert s1 == s2
            assert e2 == e2

    def test_comp_resids(
        self, interchange_system, request, num_residues, eg5_ligands, eg5_cofactor
    ):
        _, comp_resids = interchange_system

        ligand, _ = eg5_ligands
        assert len(comp_resids) == 4
        assert list(comp_resids)[0] == request.getfixturevalue(self.protein_comp)
        assert list(comp_resids)[1] == ligand
        assert list(comp_resids)[2] == eg5_cofactor
        assert list(comp_resids)[3] == SolventComponent()

        # 356 residues in EG5
        assert_equal(list(comp_resids.values())[0], [i for i in range(356)])
        # ligand
        assert_equal(list(comp_resids.values())[1], [356])
        # cofactor
        assert_equal(list(comp_resids.values())[2], [357])
        # solvent
        assert_equal(list(comp_resids.values())[3], [i for i in range(358, num_residues)])


@pytest.mark.parametrize("box_shape", ["dodecahedron", "cube"])
def test_box_setting_dodecahedron(
    box_shape,
    smc_components_benzene_named,
    T4_protein_component,
    water_off,
    protein_ff_settings,
):
    import openmm.unit

    solvation_settings = PackmolSolvationSettings(
        target_density=0.3 * unit.grams / unit.mL,
        box_shape=box_shape,
    )
    assert solvation_settings.box_shape == box_shape

    interchange, _ = interchange_system_creation(
        ffsettings=protein_ff_settings,
        solvation_settings=solvation_settings,
        smc_components=smc_components_benzene_named,
        protein_component=T4_protein_component,
        solvent_component=SolventComponent(),
        solvent_offmol=water_off,
    )

    openmm_system = interchange.to_openmm_system()

    # have to massage list[Vec3[Quantity]] into a more convenient array
    a, b, c = [
        row.value_in_unit(openmm.unit.nanometer)
        for row in openmm_system.getDefaultPeriodicBoxVectors()
    ]

    openmm_box = np.asarray([a, b, c])

    DODECAHEDRON_BOX = np.array([[1, 0, 0], [0, 1, 0], [0.5, 0.5, 0.5**0.5]])

    match box_shape:
        case "dodecahedron":
            np.testing.assert_allclose(
                interchange.box.m / np.linalg.norm(interchange.box.m, axis=1),
                DODECAHEDRON_BOX,
            )

            np.testing.assert_allclose(
                openmm_box / np.linalg.norm(openmm_box, axis=1),
                DODECAHEDRON_BOX,
            )

        case "cube":
            np.testing.assert_allclose(
                interchange.box.m / np.linalg.norm(interchange.box.m, axis=1),
                np.eye(3),
            )

            np.testing.assert_allclose(
                openmm_box / np.linalg.norm(openmm_box, axis=1),
                np.eye(3),
            )


class TestVsitesCompresids:
    @pytest.fixture(scope="class")
    def solute(self):
        offmol = Molecule.from_smiles("c1cc(ccc1O)Br")
        offmol.generate_conformers(n_conformers=1)
        offmol.assign_partial_charges(partial_charge_method="gasteiger")

        return SmallMoleculeComponent.from_openff(offmol)

    @pytest.fixture(scope="class")
    def interchange_system(self, solute, water_off, vsite_offxml):
        smc_components = {solute: solute.to_openff()}
        interchange, comp_resids = interchange_system_creation(
            ffsettings=InterchangeFFSettings(
                forcefields=[vsite_offxml, "opc.offxml"],
            ),
            solvation_settings=PackmolSolvationSettings(
                number_of_solvent_molecules=1000, solvent_padding=None
            ),
            smc_components=smc_components,
            protein_component=None,
            solvent_component=ExtendedSolventComponent(),
            solvent_offmol=water_off,
        )

        return interchange, comp_resids

    def test_num_particles(self, interchange_system):

        interchange, comp_resids = interchange_system
        openmm_system = interchange.to_openmm_system()
        # 1000 * 4 sites + 13 atoms + 1 site
        assert openmm_system.getNumParticles() == (4000 + 14)

    def test_num_missing_error(self, interchange_system):

        interchange, comp_resids = interchange_system

        n_residues = interchange.to_openmm_topology(collate=False).getNumResidues()

        # Claim more residues than the topology has, so num_missing goes negative
        broken_comp_resids = {key: val.copy() for key, val in comp_resids.items()}
        first_key = next(iter(broken_comp_resids))
        broken_comp_resids[first_key] = np.concatenate(
            [broken_comp_resids[first_key], np.arange(n_residues, 2 * n_residues)]
        ).astype(int)

        errmsg = "There are fewer OpenMM Topology residues"
        with pytest.raises(ValueError, match=errmsg):
            _fill_vsite_compresids(interchange, broken_comp_resids)

    def test_non_unique_resids(self, interchange_system):

        interchange, comp_resids = interchange_system

        new_comp_resids = {key: np.append(val, val) for key, val in comp_resids.items()}

        with pytest.raises(ValueError, match="Non unique residue indexes"):
            _fill_vsite_compresids(interchange, new_comp_resids)

    def test_fill_compresids(self, interchange_system, solute):
        interchange, comp_resids = interchange_system

        omm_top = interchange.to_openmm_topology(collate=False)
        assert omm_top.getNumResidues() == 2002

        # 2 alchemical residues (1 normal atoms, 1 vsite), rest is water
        assert len(comp_resids[ExtendedSolventComponent()]) == 2000
        assert_equal(
            comp_resids[ExtendedSolventComponent()],
            np.array([i for i in range(1, 1001)] + [i for i in range(1002, 2002)]),
        )
        assert len(comp_resids[solute]) == 2
        assert comp_resids[solute][0] == 0
        assert comp_resids[solute][1] == 1001
