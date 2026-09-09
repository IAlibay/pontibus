# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import logging

import numpy as np
import numpy.typing as npt
from openff.packmol import (
    RHOMBIC_DODECAHEDRON,
    UNIT_CUBE,
    pack_box,
)
from openff.packmol._packmol import (
    _check_add_positive_mass,
    _check_box_shape_shape,
    _max_dist_between_points,
    _scale_box,
)
from openff.toolkit import Molecule as OFFMolecule
from openff.toolkit import Topology
from openff.units import Quantity

from pontibus.utils.molecule_utils import (
    _get_offmol_resname,
    _get_unique_mols,
    _set_offmol_resname,
)
from pontibus.utils.molecules import offmol_water
from pontibus.utils.settings import (
    InterchangeOpenMMSolvationSettings,
    PackmolSolvationSettings,
)

logger = logging.getLogger(__name__)


def _n_solvent_and_box_from_density(
    solute_topology: Topology,
    solvent: OFFMolecule,
    box_shape: npt.NDArray,
    padding: Quantity,
    target_density: Quantity,
) -> tuple[int, Quantity]:
    """
    Get the number of solvent molecules based on the expected box density.

    Parameters
    ----------
    solute_topology : openff.toolkit.Topology
      A Topology of the solutes being solvated.
    solvent: openff.toolkit.Molecule
      A Molecule representing the solvent to use.
    box_shape : npt.NDArray
      A numpy array defining the box shape.
    padding : openff.units.Quantity
      The minimum distance from the solute and the edge of the box.
    target_density : openff.units.Quantity
      The target density for the solvated system.

    Returns
    -------
    n_solvent : int
      The number of solvent molecules to add.
    box_vectors : openff.units.Quantity
      The unit cell box vectors. Array with shape (3, 3).

    Raises
    ------
    PACKMOLValueError
      If the `box_shape` is not an array of shape (3, 3)
    ValueError
      If `box_shape`, `padding`, or `target_density` are `None`.

    Acknowledgements
    ----------------
    This is mostly vendored code from Interchange.components._packmol.py
    """
    # Validate inputs
    _check_box_shape_shape(box_shape)

    # compute box vectors from the solulte length and requested padding
    solute_length = _max_dist_between_points(solute_topology.get_positions())
    image_distance = solute_length + padding * 2  # type: ignore[operator]
    box_vectors = box_shape * image_distance

    # compute target masses of solvent
    box_volume = np.linalg.det(box_vectors.m) * box_vectors.u**3
    target_mass = box_volume * target_density
    solute_mass = sum(
        sum([atom.mass for atom in molecule.atoms]) for molecule in solute_topology.molecules
    )
    solvent_mass = sum([atom.mass for atom in solvent.atoms])
    solvent_mass_to_add = target_mass - solute_mass

    _check_add_positive_mass(solvent_mass_to_add)

    n_solvent = int((solvent_mass_to_add / solvent_mass).m_as("dimensionless").round())

    return n_solvent, box_vectors


def _box_density_from_mols(
    molecules: list[OFFMolecule],
    n_copies: list[int],
    solute_topology: Topology,
    target_density: Quantity,
    box_shape: npt.NDArray,
) -> Quantity:
    """
    Approximate box size with known number and type of molecules.

    Generate an approximate box size based on the number and molecular
    weight of the molecules present, and a target density for the final
    solvated mixture.

    Parameters
    ----------
    molecules : list[openff.toolkit.Molecule]
      The molecules in the system.
    n_copies : list of int
      The number of copies of each molecule.
    solute_topology : openff.toolkit.Topology
      The solute topology.
    target_density : openff.units.Quantity
      The target mass density for final system. It should have units
      compatible with g / mL.
    box_shape: NDArray
      The shape of the simulation box, used in conjunction with the
      `target_density` parameter. Should have shape (3, 3) with all positive
      elements.

    Returns
    -------
    box_vectors : openff.units.Quantity
      The unit cell box vectors. Array with shape (3, 3).

    Acknowledgements
    ----------------
    Adapted from openff.interchange, with a fix to account for solute mass.
    """
    # Get the desired volume in cubic working units
    molecules_total_mass = sum(
        sum([atom.mass for atom in molecule.atoms]) * n for molecule, n in zip(molecules, n_copies)
    )
    solute_total_mass = sum(
        sum([atom.mass for atom in molecule.atoms]) for molecule in solute_topology.molecules
    )
    total_mass = molecules_total_mass + solute_total_mass
    volume = total_mass / target_density

    return _scale_box(box_shape, volume, box_scaleup_factor=1.0)


def _neutralize_and_pack_box(
    solute_topology: Topology,
    solvent: OFFMolecule,
    init_n_solvent_mols: int,
    ion_concentration: Quantity,
    packing_tolerance: Quantity,
    box_vectors: Quantity,
    solvent_molarity: Quantity = Quantity(55.4, "mole / liter"),
) -> Topology:
    """
    Parameters
    ----------
    solute_topology : openff.toolkit.Topology
      The solute topology.
    solvent : openff.toolkit.Molecule
      An OpenFF Molecule representing the solvent.
    init_n_solvent_mols : int
      The "initial" number of solvent molecules to be placed.
    ion_concentration : openff.units.Quantity
      The salt concentration to add.
    packing_tolerance : openff.units.Quantity
      The packmol packing tolerance.
    box_vectors : openff.units.Quantity
      The box vectors of the final box to be written.
    solvent_molarity : openff.units.Quantity
      The pure solvent molarity, default to water at 55.4 mole/liter.

    Returns
    -------
    Topology
      The solvated topology.

    Raises
    ------
    ValueError
      If the total charge after adding the ions, is not close to zero.

    Notes
    -----
    For now this only really works for water.

    Acknowledgements
    ----------------
    Adapted from openff.interchange
    """

    solute_charge = sum([molecule.total_charge.m for molecule in solute_topology.molecules])
    solute_charge_magnitude = np.abs(solute_charge)

    na = OFFMolecule.from_smiles("[Na+]")
    cl = OFFMolecule.from_smiles("[Cl-]")

    # Set ion metadata
    solvent_key = solvent.properties["key"]
    _set_offmol_resname(na, "NA+")
    _set_offmol_resname(cl, "CL-")
    na.properties["key"] = solvent_key
    cl.properties["key"] = solvent_key

    # Get the individual solvent mass and the total mass of all the solvent molecules
    solvent_mol_mass = sum([atom.mass for atom in solvent.atoms])
    solvent_total_mass = init_n_solvent_mols * solvent_mol_mass
    nacl_mass = sum([atom.mass for atom in na.atoms]) + sum([atom.mass for atom in cl.atoms])

    # Compute the number of salt ions to add for bulk salt
    # If ionic strength is zero, then the solute_ion_ratio for SLTCAP is undefined
    if ion_concentration == Quantity(0, "mole / liter"):
        nacl_to_add = 0
        solvent_mass_to_add = solvent_total_mass
    else:
        # Compute the number of salt "molecules" to add from the mass
        # and concentration for a neutral solute
        neutral_nacl_mass_fraction = (ion_concentration * nacl_mass) / (
            solvent_molarity * solvent_mol_mass
        )
        neutral_nacl_mass_to_add = solvent_total_mass * neutral_nacl_mass_fraction
        neutral_nacl_to_add = np.round(neutral_nacl_mass_to_add / nacl_mass).m_as("dimensionless")

        if neutral_nacl_to_add == 0:
            nacl_to_add = 0
            solvent_mass_to_add = solvent_total_mass
        else:
            # Compute the number of salt "molecules" to add using the SLTCAP method
            solute_ion_ratio = solute_charge_magnitude / (2 * neutral_nacl_to_add)
            sltcap_effective_ionic_strength = ion_concentration * (
                np.sqrt(1 + solute_ion_ratio * solute_ion_ratio) - solute_ion_ratio
            )

            nacl_mass_fraction = (sltcap_effective_ionic_strength * nacl_mass) / (
                solvent_molarity * solvent_mol_mass
            )
            nacl_mass_to_add = solvent_total_mass * nacl_mass_fraction
            nacl_to_add = np.round(nacl_mass_to_add / nacl_mass).m_as("dimensionless")

            # Compute the number of water molecules to add to make up the remaining mass
            solvent_mass_to_add = solvent_total_mass - nacl_mass_to_add

    solvent_to_add = int(np.round(solvent_mass_to_add / solvent_mol_mass).m_as("dimensionless"))

    # Neutralise the system by adding and removing salt
    na_to_add = int(
        np.round(
            nacl_to_add + (solute_charge_magnitude - solute_charge) / 2.0,
        )
    )
    cl_to_add = int(
        np.round(
            nacl_to_add + (solute_charge_magnitude + solute_charge) / 2.0,
        )
    )

    if abs(solute_charge + na_to_add - cl_to_add) > 1e-6:
        raise ValueError(f"Failed to neutralise solute with charge {solute_charge.m}")

    return pack_box(
        molecules=[solvent, na, cl],
        number_of_copies=[solvent_to_add, na_to_add, cl_to_add],
        solute=solute_topology,
        tolerance=packing_tolerance,
        box_vectors=box_vectors,
        center_solute=False,
        working_directory=None,
        retain_working_files=False,
    )


def _process_solvation_inputs(
    solute_topology: Topology,
    solvent_offmol: OFFMolecule,
    solvation_settings: PackmolSolvationSettings | InterchangeOpenMMSolvationSettings,
) -> tuple[npt.NDArray | None, int, Quantity]:
    """
    Helper method for solvation methods.

    Parameters
    ----------
    solute_topology : openff.toolkit.Topology
      The solute Topology to solvate.
    solvent_offmol : openff.toolkit.Molecule
      The OpenFF Molecule representing the solvent.
    solvation_settings : PackmolSolvationSettings | InterchangeOpenMMSolvationSettings

    Returns
    -------
    box_shape : npt.NDArray | None
      NumPy array defining the box shape, if defined.
    n_solvent : int
      The number of solvent molecules to be added.
    box_vectors : Quantity
      The solvated box vectors.
    """
    # Pick up the user selected box shape
    # todo: switch to calling `get` once we normalize settings
    if solvation_settings.box_shape is not None:
        box_shape = {
            "cube": UNIT_CUBE,
            "dodecahedron": RHOMBIC_DODECAHEDRON,
        }[solvation_settings.box_shape.lower()]
    else:
        box_shape = None

    # Get the number of solvent molecules and the box vectors
    if solvation_settings.number_of_solvent_molecules is not None:
        n_solvent = solvation_settings.number_of_solvent_molecules
        if solvation_settings.box_vectors is not None:
            box_vectors = solvation_settings.box_vectors
        else:
            box_vectors = _box_density_from_mols(
                molecules=[solvent_offmol],
                n_copies=[solvation_settings.number_of_solvent_molecules],
                solute_topology=solute_topology,
                target_density=solvation_settings.target_density,  # type: ignore[arg-type]
                box_shape=box_shape,  # type: ignore[arg-type]
            )
    else:
        # In this case box vectors cannot be defined
        n_solvent, box_vectors = _n_solvent_and_box_from_density(
            solute_topology=solute_topology,
            solvent=solvent_offmol,
            box_shape=box_shape,  # type: ignore[arg-type]
            padding=solvation_settings.solvent_padding,  # type: ignore[arg-type]
            target_density=solvation_settings.target_density,  # type: ignore[arg-type]
        )

    return box_shape, n_solvent, box_vectors


def packmol_solvation(
    solute_topology: Topology,
    solvent_offmol: OFFMolecule,
    solvation_settings: PackmolSolvationSettings,
    neutralize: bool,
    ion_concentration: Quantity,
) -> Topology:
    """
    Solvate solute Topology using the Interchange packmol interface.

    Parameters
    ----------
    solute_topology : Topology
      The solute Topology to solvate.
    solvent_offmol : OFFMolecule
      An OpenFF Molecule representing the solvent.
    solvation_settings : PackmolSolvationSettings
      Settings for how to solvate the system using
      the packmol interface.
    neutralize : bool
      Whether or not the system should be neutralized.
    ion_concentration : Quantity
      The concentration of NaCl to add to the system when
      neutralizing. Note that this is ignored if neutralize
      if False. Must be compatible with mole / liter. Addition
      of ions is done through the SLTCAP method.

    Returns
    -------
    Topology
      The solvated Topology.

    Raises
    ------
    ValueError
      If ``neutralize`` is ``True`` and ``solvent_offmol`` is not water.
      if ``ion_concentration`` is not compatible with mole / liter.
    """
    box_shape, n_solvent, box_vectors = _process_solvation_inputs(
        solute_topology,
        solvent_offmol,
        solvation_settings,
    )

    if neutralize:
        if not solvent_offmol.is_isomorphic_with(offmol_water):
            errmsg = "Cannot neutralize a system with non-water solvent"
            raise ValueError(errmsg)
        if not ion_concentration.is_compatible_with("mole / liter"):
            errmsg = f"{ion_concentration} is not compatible with mole / liter"
            raise ValueError(errmsg)

        return _neutralize_and_pack_box(
            solute_topology=solute_topology,
            solvent=solvent_offmol,
            init_n_solvent_mols=n_solvent,
            ion_concentration=ion_concentration.to("mole / liter"),
            packing_tolerance=solvation_settings.packing_tolerance,
            box_vectors=box_vectors,
            solvent_molarity=Quantity(55.4, "mole / liter"),
        )
    else:
        return pack_box(
            molecules=[solvent_offmol],
            number_of_copies=[n_solvent],
            solute=solute_topology,
            tolerance=solvation_settings.packing_tolerance,
            box_vectors=box_vectors,
            target_density=None,
            box_shape=box_shape,
            center_solute=False,
            working_directory=None,
            retain_working_files=False,
        )


def openmm_solvation(
    solute_topology: Topology,
    solvent_offmol: OFFMolecule,
    solvation_settings: InterchangeOpenMMSolvationSettings,
    neutralize: bool,
    ion_concentration: Quantity,
) -> Topology:
    """
    Solvate solute Topology using OpenMM's Modeller.

    Parameters
    ----------
    solute_topology : openff.toolkit.Topology
      The Topology of the solute to solvate.
    solvent_offmol : openff.toolkit.Molecule
      An OpenFF Molecule representing the solvent species.
    solvation_settings : InterchangeOpenMMSolvationSettings
      Settings defining how the system will be solvated.
    neutralize : bool
      Whether or not the system should be neutralized.
    ion_concentration : openff.unit.Quantity
      The concentration of NaCl to add to the system
      when neutralizing. Note that this is ignored if
      ``neutralize`` is ``False``. Must be compatible
      with mole / liter. Note that this method, unlike the
      packmol interchange, uses the add neutralize method rather
      than SLTCAP.

    Raises
    ------
    ValueError
      * If neutralizing with a non-water solvent.
      * If ``ion_concentration`` is not compatible with ``mole / liter``
      * If ``neutralize`` is ``False`` and the system has a net charge.
    """
    import openmm
    import openmm.app

    ions = [OFFMolecule.from_smiles("[Na+]"), OFFMolecule.from_smiles("[Cl-]")]

    def make_vec3(positions: Quantity) -> openmm.Vec3:
        return [
            openmm.Vec3(float(row[0]), float(row[1]), float(row[2]))
            for row in positions.m_as("nanometer")  # type: ignore[union-attr]
        ]

    _, n_solvent, box_vectors = _process_solvation_inputs(
        solute_topology,
        solvent_offmol,
        solvation_settings,
    )

    if not solvent_offmol.is_isomorphic_with(offmol_water):
        errmsg = "Cannot neutralize a system with non-water solvent"
        raise ValueError(errmsg)

    # MAYBE TODO: Note we don't use SLTCAP here (Issue #148)
    if neutralize:
        if not ion_concentration.is_compatible_with("mole / liter"):
            errmsg = f"{ion_concentration} is not compatible with mole / liter"
            raise ValueError(errmsg)

        modeller = openmm.app.Modeller(
            topology=solute_topology.to_openmm(),
            positions=make_vec3(solute_topology.get_positions()),
        )

        from openmmforcefields.generators import SMIRNOFFTemplateGenerator

        forcefield = openmm.app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")

        forcefield.registerTemplateGenerator(
            SMIRNOFFTemplateGenerator(
                molecules=_get_unique_mols([m for m in solute_topology.molecules])
            ).generator
        )

        # To avoid a clash, set the box vectors to None if a shape is already
        # defined. Otherwise, use the vectors
        if solvation_settings.box_shape is not None:
            box_vectors = None  # type: ignore[assignment]
        else:
            box_vectors = make_vec3(box_vectors)

        modeller.addSolvent(
            forcefield=forcefield,
            model="tip3p",
            numAdded=n_solvent,
            boxVectors=box_vectors,
            boxShape=solvation_settings.box_shape,
            positiveIon="Na+",
            negativeIon="Cl-",
            ionicStrength=ion_concentration.to_openmm(),  # type: ignore[attr-defined]
            neutralize=True,
        )

        topology = Topology.from_openmm(
            modeller.topology,
            unique_molecules=_get_unique_mols(
                [*solute_topology.molecules] + [solvent_offmol] + ions
            ),
            positions=modeller.getPositions(),
        )

        solvent_key = solvent_offmol.properties["key"]
        solvated_molecules = []

        for molecule_index, molecule in enumerate(topology.molecules):
            if molecule_index < solute_topology.n_molecules:
                og_molecule = solute_topology.molecule(molecule_index)
                if np.allclose(molecule.conformers[0], og_molecule.conformers[0]):
                    solvated_molecules.append(og_molecule)
                else:
                    raise ValueError("Solute molecule positions have changed")

                continue

            # all "solvents"  get this key, even if they're ions
            molecule.properties["key"] = solvent_key
            solvent_residue_name = _get_offmol_resname(solvent_offmol)
            match molecule.n_atoms:
                case 3:
                    _set_offmol_resname(molecule, solvent_residue_name)

                case 1:
                    match molecule.atom(0).atomic_number:
                        case 11:
                            _set_offmol_resname(molecule, "NA+")
                        case 17:
                            _set_offmol_resname(molecule, "CL-")
                        case _:
                            raise ValueError(
                                f"Unrecognized ion with atomic number {molecule.atom(0).atomic_number}"
                            )

                case _:
                    raise ValueError(
                        f"Unrecognized 'solvent' molecule with {molecule.to_smiles()} SMILES"
                    )

            solvated_molecules.append(molecule)

    else:
        raise NotImplementedError("Non-neutralized systems with net charge are not supported")

    final_topology = Topology.from_molecules(solvated_molecules)
    final_topology.box_vectors = topology.box_vectors

    return final_topology
