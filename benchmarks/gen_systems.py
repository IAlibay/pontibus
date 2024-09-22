import click
import json
import gzip
import pathlib
from rdkit import Chem
from gufe.tokenization import JSON_HANDLER
from gufe import SmallMoleculeComponent, ChemicalSystem
from openff.toolkit import Molecule
from pontibus.components import ExtendedSolventComponent


solvents: dict[str, SmallMoleculeComponent] = {}
systems: list[ChemicalSystem] = []


def add_chemical_systems(
    sdffile: str,
    dataset_name: str,
    solvents: dict[str, SmallMoleculeComponent],
    systems: list[ChemicalSystem]
) -> None:
    """
    Add Solute + Solvent ChemicalSystems to running list.

    Parameters
    ----------
    sdffile : str
      The SDF file to read entries from.
    dataset_name : str
      The name of the dataset.
    solvents: dict[str, SmallMoleculeComponent]
      Running dictionary of solvents to draw & store prepared solvent
      molecules from/to.
    systems: list[ChemicalSystem]
      Runing list of ChemicalSystems we are appending to.
    """
    for i, rdmol in enumerate(Chem.SDMolSupplier(sdffile, removeHs=False)):
        offmol = Molecule.from_rdkit(rdmol)
        offmol.assign_partial_charges(partial_charge_method='am1bccelf10')
        solvent_smi = rdmol.GetProp('solvent')
        if solvent_smi not in solvents.keys():
            solvent_offmol = Molecule.from_smiles(solvent_smi)
            solvent_offmol.generate_conformers()
            solvent_offmol.assign_partial_charges(partial_charge_method='am1bccelf10')
            solvents[solvent_smi] = SmallMoleculeComponent.from_openff(solvent_offmol)

        systems.append(ChemicalSystem(
            {
                'solute': SmallMoleculeComponent.from_openff(offmol),
                'solvent': ExtendedSolventComponent(solvent_molecule=solvents[solvent_smi]),
            },
            name=f"molecule{i}_{dataset_name}"
        ))


def store_chemical_systems(systems: list[ChemicalSystem]):
    """
    Store ChemicalSystems to gzip file.

    Parameters
    ----------
    systems: list[ChemicalSystem]
      List of ChemicalSystems to store to file.
    """
    for system in systems:
        with gzip.open(f"{system.name}_chemicalsystem.gz", 'wt') as zipfile:
            json.dump(system.to_dict(), zipfile, cls=JSON_HANDLER.encoder)


@click.command
@click.option(
    '--sdfs',
    type=click.Path(dir_okay=False, file_okay=True, path_type=pathlib.Path),
    required=True,
    help="Path to the prepared PDB file to validate",
)