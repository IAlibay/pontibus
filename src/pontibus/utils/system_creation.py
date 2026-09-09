# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import logging
import tempfile
from itertools import product
from string import ascii_uppercase

import numpy as np
import numpy.typing as npt
import openmm
from gufe import Component, ProteinComponent, SmallMoleculeComponent, SolventComponent
from openff.interchange import Interchange
from openff.toolkit import ForceField, Topology
from openff.toolkit import Molecule as OFFMolecule
from openff.units import unit

from pontibus.utils.molecule_utils import (
    _check_and_deduplicate_charged_mols,
    _check_library_charges,
    _get_num_residues,
    _get_offmol_resname,
    _set_offmol_metadata,
    _set_offmol_resname,
)
from pontibus.utils.molecules import offmol_water
from pontibus.utils.settings import (
    InterchangeFFSettings,
    InterchangeOpenMMSolvationSettings,
    PackmolSolvationSettings,
)
from pontibus.utils.system_solvation import openmm_solvation, packmol_solvation

logger = logging.getLogger(__name__)


def _get_virtual_site_components(
    omm_system: openmm.System,
    omm_topology: openmm.app.Topology,
    nonvsite_comp_resids: dict[Component, npt.NDArray],
    nonvsite_resids: set[int],
) -> dict[Component, list[int]]:
    """
    Helper method to get a comp_resids dictionary for the virtual sites in the system.

    Parameters
    ----------
    omm_system : openmm.System
      The OpenMM System to look through.
    omm_topology : openmm.Topology
      The OpenMM Topology to look through.
    nonvsite_comp_resids : dict[Component, list[int]]
      The dictionary of components to residue indexes
      for the non virtual site atoms.
    nonvsite_resids : set[int]
      A set of residue indexes for the non virtual site atoms.

    Returns
    -------
    viste_comp_resids : dict[Component, list[int]]
      A dictionary with lists of residue indexes keyed by Component
      for all virtual sites in the System.
    """
    # Create a temporary comp_resids dictionary for virtual sites
    vsites_comp_resids: dict[Component, list[int]] = {comp: [] for comp in nonvsite_comp_resids}

    # Invert comp_resids to be able to know which residue is which Component
    resids_to_comp: dict[int, Component] = {
        int(resid): comp for comp, resids in nonvsite_comp_resids.items() for resid in resids
    }

    # Create a list of atoms for later use
    omm_atoms = list(omm_topology.atoms())

    # Now loop over the residues and find parents
    for residue in omm_topology.residues():
        # Skip if we already know you
        if residue.index in nonvsite_resids:
            continue

        residue_component: set[Component] = set()
        for atom in residue.atoms():
            if not omm_system.isVirtualSite(atom.index):
                errmsg = (
                    f"OpenMM Topology residue {residue.index} was expected to "
                    "contain only virtual sites, but particle {atom.index} is "
                    "not a virtual site."
                )
                raise ValueError(errmsg)

            # We can safely assume that the parent will always be Component
            parent_atom_index = omm_system.getVirtualSite(atom.index).getParticle(0)
            parent_res_index = omm_atoms[parent_atom_index].residue.index
            residue_component.add(resids_to_comp[parent_res_index])

        if len(residue_component) != 1:
            errmsg = (
                f"Virtual site residue {residue.index} has parent particles "
                "spanning more than one Component"
            )
            raise ValueError(errmsg)

        vsites_comp_resids[residue_component.pop()].append(residue.index)

    return vsites_comp_resids


def _fill_vsite_compresids(
    interchange: Interchange,
    comp_resids: dict[Component, npt.NDArray],
) -> None:
    """
    In-place fill in the comp_resids dictionary with the
    OpenMM Topology (note: assuming ``collate=False``)
    residue indexes with virtual sites.

    Parameters
    ----------
    interchange : Interchange
      The Interchange object which defines the OpenMM
      System / Topology we will be creating.
    comp_resids : dict[Component, npt.NDArray]
      The comp_resids dictionary to backfill with virtual
      site residues. This dictionary should already have
      non-virtual site residues assigned.

    Notes
    -----
    This processes relies on:
    * The OpenMM Topology being created via Interchange
      with ``collate=False``.
    * Each Molecule in the system having been assigned a
      **unique** residue number **prior** to Interchange creation.
    * The ``comp_resids`` dictionary has already been filled
      with non-virtual site residues.
    """
    # Validate the number of known & missing residue indexes
    known_resids = np.concatenate(list(comp_resids.values()), dtype=int)
    known_resids_set = set(known_resids.tolist())

    # There should only be unique indexes
    if len(known_resids_set) != len(known_resids):
        errmsg = "Non unique residue indexes found in comp_resids!"
        raise ValueError(errmsg)

    # Number of missing residues is the different between OpenMM Topology's
    # reporter count and the set of known residue indexes
    omm_top = interchange.to_openmm_topology(collate=False)
    num_missing = omm_top.getNumResidues() - len(known_resids_set)

    # If zero early return
    if num_missing == 0:
        return

    # If less than zero, something went wrong
    if num_missing < 0:
        errmsg = (
            "There are fewer OpenMM Topology residues than expected "
            "molecules. Something went wrong!"
        )
        raise ValueError(errmsg)

    # Number of missing residues has been validated, we now move ahead with
    # backfilling the dictionary
    logger.info(f"{num_missing} virtual site residues found, adding them to comp_resids")

    # Get the vsite comp_resids dictionary
    vsites_comp_resids = _get_virtual_site_components(
        omm_system=interchange.to_openmm_system(),
        omm_topology=omm_top,
        nonvsite_comp_resids=comp_resids,
        nonvsite_resids=known_resids_set,
    )

    # Check that we have assigned the same number of residues as were originally missing
    n_assigned = sum(len(v) for v in vsites_comp_resids.values())
    if n_assigned != num_missing:
        errmsg = (
            f"{n_assigned} virtual site residues were assigned, but "
            f"{num_missing} residues needed to be assigned. "
        )
        raise ValueError(errmsg)

    # Now we merge everything back and that's it!
    for comp in comp_resids:
        # force it back into ints since an empty list would cause floats
        comp_resids[comp] = np.concatenate([comp_resids[comp], vsites_comp_resids[comp]]).astype(
            int
        )


def _proteincomp_to_topology(protein_component: ProteinComponent) -> Topology:
    """
    Convert a ProteinComponent to an OpenFF Topology via PDB serialization.

    Parameters
    ----------
    protein_component : ProteinComponent
      The ProteinComponent to convert.

    Returns
    -------
    off_top : openff.toolkit.Topology
      A Topology containing the protein.
    """
    # TODO: maybe switch to NamedTemporaryFile eventually?
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = f"{tmpdir}/tmp.pdb"
        protein_component.to_pdb_file(filepath)
        off_top = Topology.from_pdb(filepath)

    return off_top


def _validate_components(
    protein_component: ProteinComponent | None,
    solvent_component: SolventComponent | None,
    solvent_offmol: OFFMolecule | None,
) -> None:
    """
    Validates input components to ``interchange_packmol_creation``.


    Parameters
    ----------
    protein_component : Optional[ProteinComponent]
      Protein component of the system, if any.
    solvent_component : Optional[SolventComponent]
      Solvent component of the system, if any.
    solvent_offmol : Optional[openff.toolkit.Molecule]
      OpenFF Molecule defining the solvent, if necessary


    Raises
    ------
    ValueError
      If the solvent_component requests counterions for non-water solvent.
      If the solvent component requests counterions without neutralizing.
      If the counterions are not Na+ and Cl-.
      If we have a solvent_component but no solvent_offmol.
      If the solvent_component and solvent_offmol are not isomorphic.
      If the solvent_offmol doesn't have one conformer.
    """
    if (solvent_component is None) and (protein_component is not None):
        # If there's a protein without solvent then we're in trouble
        errmsg = "Must have solvent to have a protein"
        raise ValueError(errmsg)

    if solvent_component is not None:
        if solvent_offmol is None:
            errmsg = "A solvent offmol must be passed to solvate a system!"
            raise ValueError(errmsg)

        # Check we're not trying to neutralize with non-water solvent
        if not solvent_offmol.is_isomorphic_with(offmol_water):
            if solvent_component.neutralize or solvent_component.ion_concentration > 0 * unit.molar:
                errmsg = "Counterions are currently not supported for non-water solvent"
                raise ValueError(errmsg)

        # We can't add ions without neutralizing but we can neutralize without ion conc
        if not solvent_component.neutralize:
            if solvent_component.ion_concentration > 0 * unit.molar:
                errmsg = "Cannot add ions without neutralizing"
                raise ValueError(errmsg)

        # Can't neutralize with anything but Na Cl
        if solvent_component.neutralize:
            pos = solvent_component.positive_ion.upper()
            neg = solvent_component.negative_ion.upper()

            if pos != "NA+" or neg != "CL-":
                errmsg = f"Can only neutralize with NaCl, got {pos} / {neg}"
                raise ValueError(errmsg)

        # Check that the component matches the offmol
        if not solvent_offmol.is_isomorphic_with(OFFMolecule.from_smiles(solvent_component.smiles)):
            errmsg = (
                f"Passed molecule: {solvent_offmol} does not match the "
                f"the solvent component: {solvent_component.smiles}"
            )
            raise ValueError(errmsg)

        if solvent_offmol.n_conformers != 1:
            errmsg = (
                "Solvent OpenFF Molecule should have a single conformer. "
                f"Number of conformers found: {solvent_offmol.n_conformers}"
            )
            raise ValueError(errmsg)


def _get_force_field(ffsettings: InterchangeFFSettings, exclude_ff14sb: bool) -> ForceField:
    """
    Get a ForceField object based on an input InterchangeFFSettings object.

    Parameters
    ----------
    ffsettings : InterchangeFFSettings
      Settings defining how the force field is applied.
    exclude_ff14sb : bool
      Whether or not to exclude ff14sb

    Returns
    -------
    force_field : ForceField
      An OpenFF toolkit ForceField object.
    """
    # forcefields is a list so we unpack it
    if exclude_ff14sb:
        ffnames = [name for name in ffsettings.forcefields if "ff14sb" not in name]
        force_field = ForceField(*ffnames)
    else:
        force_field = ForceField(*ffsettings.forcefields)

    # We also set nonbonded cutoffs whilst we are here
    # TODO: double check what this means for nocutoff simulations
    force_field["Electrostatics"].cutoff = ffsettings.nonbonded_cutoff
    force_field["vdW"].cutoff = ffsettings.nonbonded_cutoff
    force_field["vdW"].switch_width = ffsettings.switch_width

    return force_field


def _assign_comp_resnames_and_keys(
    smc_components: dict[SmallMoleculeComponent, OFFMolecule],
    solvent_component: SolventComponent | None,
    solvent_offmol: OFFMolecule | None,
    protein_component: ProteinComponent | None,
    protein_molecules: list[OFFMolecule] | None,
) -> None:
    """
    Assign residue names to Small and Solvent Components.

    Parameters
    ----------
    smc_components : dict[SmallMoleculeComponent, openff.toolkit.Molecule]
      Solute SmallMoleculeComponents.
    solvent_component : Optional[SolventComponent]
      Solvent component of the system, if any.
    solvent_offmol : Optional[openff.toolkit.Molecule]
      OpenFF Molecule defining the solvent, if necessary.
    """
    # List of unique resnames to track
    unique_resnames = []

    # If we have solvent, we set its residue name
    if solvent_component is not None:
        # Assign the key
        solvent_offmol.properties["key"] = str(solvent_component.key)  # type: ignore[union-attr]

        # Assign the resname if necessary
        offmol_resname = _get_offmol_resname(solvent_offmol)
        if offmol_resname is None:
            offmol_resname = "SOL"
            _set_offmol_resname(solvent_offmol, offmol_resname)

        if solvent_component.neutralize:
            if offmol_resname in ["NA+", "CL-"]:
                errmsg = "Solvent resname is set to NA+ or CL-"
                raise ValueError(errmsg)

        unique_resnames.append(offmol_resname)

    # A store of residue names to replace residue names if they aren't unique
    resnames_store = ["".join(i) for i in product(ascii_uppercase, repeat=3)]

    for comp, offmol in smc_components.items():
        # Assign the key
        offmol.properties["key"] = str(comp.key)

        # Assign the resname if necessary
        off_resname = _get_offmol_resname(offmol)
        if off_resname is None or off_resname in unique_resnames:
            # warn that we are overriding clashing molecule resnames
            if off_resname in unique_resnames:
                wmsg = f"Duplicate residue name {off_resname}, duplicate will be renamed"
                logger.warning(wmsg)

            # just loop through and pick up a name that doesn't exist
            while (off_resname in unique_resnames) or (off_resname is None):
                off_resname = resnames_store.pop(0)

        wmsg = f"Setting component {comp} residue name to {off_resname}"
        logger.warning(wmsg)
        _set_offmol_resname(offmol, off_resname)
        unique_resnames.append(off_resname)

    if protein_component is not None:
        # Assign the key and check that everything has a resname
        for mol in protein_molecules:  # type: ignore[union-attr]
            mol.properties["key"] = str(protein_component.key)

            for at in mol.atoms:
                if "residue_name" not in at.metadata:
                    errmsg = "protein molecule missing residue info"
                    raise ValueError(errmsg)


def _post_process_topology(
    pre_topology, smc_components, solvent_component, protein_component
) -> Topology:
    """
    Helper method to post-process a Topology

    Specifically we:
      1. Assign molecule chains based on their components.
      2. Ensure all molecules have a residue number.
      3. Get the resindex range for each molecule and add it to comp_resids.

    Parameters
    ----------
    pre_topology : openff.toolkit.Topology
      The Topology to post-process
    smc_components : dict[SmallMoleculeComponent, openff.toolkit.Molecule]
      Solute SmallMoleculeComponents.
    solvent_component : Optional[SolventComponent]
      Solvent component of the system, if any.
    protein_component : Optional[ProteinComponent]
      Protein component of the system, if any.

    Returns
    -------
    post_topology : openff.toolkit.Topology
      The post-processed Topology
    """
    mols = [m for m in pre_topology.molecules]

    # Create a list of components
    comps = [*smc_components.keys()]

    for extra_comp in [solvent_component, protein_component]:
        if extra_comp is not None:
            comps.append(extra_comp)

    # Do some checks and get a list of all existing chains
    known_chains = set()
    for mol in mols:
        chain_truth = ["chain_id" in at.metadata for at in mol.atoms]
        resnum_truth = ["residue_number" in at.metadata for at in mol.atoms]

        if any(chain_truth):
            if not all(chain_truth):
                errmsg = f"All atoms in {mol} must have chain ID defined"
                raise ValueError(errmsg)

            chain_ids = set([at.metadata["chain_id"] for at in mol.atoms])
            known_chains.update(chain_ids)

        if any(resnum_truth):
            if not all(resnum_truth):
                errmsg = f"All atoms in {mol} must have a residue number if any defined"
                raise ValueError(errmsg)

    # Get a list of available chain IDs
    available_ids = [a for a in ascii_uppercase if a not in known_chains]

    if len(available_ids) < len(comps):
        errmsg = "Too few chain IDs are available"
        raise ValueError(errmsg)

    # Add in a chain for each Component
    chains = {comp.key: chain_id for comp, chain_id in zip(comps, available_ids)}

    # Add in the chain and residue numbers if needed
    for mol in mols:
        mol_index = pre_topology.molecule_index(mol)

        # set the chain if no chain is specified
        # already checked that all atoms must be specified if any
        if "chain_id" not in mol.atoms[0].metadata:
            _set_offmol_metadata(mol, "chain_id", chains[mol.properties["key"]])

        # set the residue number if it's not been set on the first atom
        if "residue_number" not in mol.atoms[0].metadata:
            _set_offmol_metadata(mol, "residue_number", mol_index)

    # create the new Topology
    post_topology = Topology.from_molecules(mols)
    post_topology.box_vectors = pre_topology.box_vectors

    return post_topology


def _protein_split_combine_interchange(
    input_topology: Topology,
    charge_from_molecules: list[OFFMolecule] | None,
    protein_component: ProteinComponent | None,
    ffsettings: InterchangeFFSettings,
) -> Interchange:
    """
    Create an interchange as the combination of the protein
    and non-protein components.

    Parameters
    ----------
    input_topology : openff.toolkit.Topology
      The input topology to split and combine into an interchange.
    charge_from_molecules : list[OFFMolecule] | None
      A list of charged molecules to pass on Interchange creation.
    protein_component : ProteinComponent | None
      The ProteinComponent, if there is one.
    ffsettings : InterchangeFFSettings
      The force field settings.

    Returns
    -------
    Interchange
      The combined Interchange, with the protein going first.

    Raises
    ------
    ValueError
      If ``protein_component`` is ``None``.
    """
    if protein_component is None:
        raise ValueError("Using ff14SB without a protein is a bad idea")

    protein_ff = _get_force_field(ffsettings=ffsettings, exclude_ff14sb=False)
    nonprotein_ff = _get_force_field(ffsettings=ffsettings, exclude_ff14sb=True)

    # Get a list of all the protein molecules
    protein_key = str(protein_component.key)
    protein_mols = []
    nonprotein_mols = []

    for mol in input_topology.molecules:
        if mol.properties["key"] == protein_key:
            protein_mols.append(mol)
        else:
            nonprotein_mols.append(mol)

    # Create the individual topologies and make sure we copy the box vectors
    protein_top = Topology.from_molecules(protein_mols)
    protein_top.box_vectors = input_topology.box_vectors
    nonprotein_top = Topology.from_molecules(nonprotein_mols)
    nonprotein_top.box_vectors = input_topology.box_vectors

    # We assume proteins will never have input charge
    protein_interchange = protein_ff.create_interchange(
        topology=protein_top,
    )

    non_protein_interchange = nonprotein_ff.create_interchange(
        topology=nonprotein_top, charge_from_molecules=charge_from_molecules
    )

    # Return the combination of the two
    return protein_interchange.combine(non_protein_interchange)


def _get_comp_resids(
    interchange: Interchange,
    smc_components: dict[SmallMoleculeComponent, OFFMolecule],
    solvent_component: SolventComponent | None,
    protein_component: ProteinComponent | None,
) -> dict[Component, npt.NDArray]:
    """
    interchange : openff.interchange.Interchange
      Interchange object for the created system.
    smc_components : dict[SmallMoleculeComponent, openff.toolkit.Molecule]
      Solute SmallMoleculeComponents.
    solvent_component: SolventComponent | None
      SolventComponent of the system, if any.
    protein_component : ProteinComponent | None
      ProteinComponent of the system, if any.

    Returns
    -------
    comp_resids : dict[Component, npt.NDArray]
      A dictionary defining the residue indices matching
      various components in the system.
    """
    comps = [*smc_components.keys()]

    for extra_comp in [solvent_component, protein_component]:
        if extra_comp is not None:
            comps.append(extra_comp)

    key_to_comp: dict[str, Component] = {comp.key: comp for comp in comps}

    # Temporary container to feed comp_resids
    compkey_residx = {}

    # Keep track of the current residx
    residx = 0
    for mol in interchange.topology.molecules:
        key = mol.properties["key"]
        num_residx = _get_num_residues(mol)
        residx_range = [r for r in range(residx, residx + num_residx)]
        if key not in compkey_residx:
            compkey_residx[key] = residx_range
        else:
            compkey_residx[key].extend(residx_range)

        # Update the residx tracker
        residx += num_residx

    # Turn compkey_resids to comp_resids
    comp_resids = {
        key_to_comp[key]: np.array(val, dtype=int) for key, val in compkey_residx.items()
    }

    return comp_resids


def interchange_system_creation(
    ffsettings: InterchangeFFSettings,
    solvation_settings: PackmolSolvationSettings | InterchangeOpenMMSolvationSettings,
    smc_components: dict[SmallMoleculeComponent, OFFMolecule],
    protein_component: ProteinComponent | None,
    solvent_component: SolventComponent | None,
    solvent_offmol: OFFMolecule | None,
) -> tuple[Interchange, dict[Component, npt.NDArray]]:
    """
    Create an Interchange object for a given system, solvating with
    packmol where necessary.

    Parameters
    ----------
    ffsettings : InterchangeFFSettings
      Settings defining how the force field is applied.
    solvation_settings : PackmolSolvationSettings | InterchangeOpenMMSolvationSettings
      Settings defining how the system will be solvated.
    smc_components : dict[SmallMoleculeComponent, openff.toolkit.Molecule]
      Solute SmallMoleculeComponents.
    protein_component : Optional[ProteinComponent]
      Protein component of the system, if any.
    solvent_component : Optional[SolventComponent]
      Solvent component of the system, if any.
    solvent_offmol : Optional[openff.toolkit.Molecule]
      OpenFF Molecule defining the solvent, if necessary

    Returns
    -------
    Interchange : openff.interchange.Interchange
      Interchange object for the created system.
    comp_resids : dict[Component, npt.NDArray]
      A dictionary defining the "canonical" (i.e. 0-indexed)
      OpenMM Topology residue indices matching the various
      components in the system.
    """

    # Component validations
    _validate_components(protein_component, solvent_component, solvent_offmol)

    # Get protein molecules if needed
    if protein_component is not None:
        protein_molecules = [m for m in _proteincomp_to_topology(protein_component).molecules]
    else:
        protein_molecules = None

    # Assign residue names to component keys
    _assign_comp_resnames_and_keys(
        smc_components=smc_components,
        solvent_component=solvent_component,
        solvent_offmol=solvent_offmol,
        protein_component=protein_component,
        protein_molecules=protein_molecules,
    )

    # Create a list of Molecules and solvate if necessary
    # Add in the ligands, this is the base "no solvent" case!
    topology_molecules = [*smc_components.values()]

    # Also create a list of charged molecules for later use
    charged_mols = [*smc_components.values()]

    if solvent_component is not None:  # solvent case
        # Append to charged molecule to charged_mols if we want to
        # otherwise we rely on library charges
        if solvation_settings.assign_solvent_charges:
            charged_mols.append(solvent_offmol)
        else:
            # Make sure we have library charges for the molecule
            _check_library_charges(
                _get_force_field(ffsettings=ffsettings, exclude_ff14sb=True), solvent_offmol
            )

        # Add protein mols if they exist
        if protein_molecules is not None:
            topology_molecules += protein_molecules

        solute_topology = Topology.from_molecules(topology_molecules)

        if isinstance(solvation_settings, PackmolSolvationSettings):
            topology = packmol_solvation(
                solute_topology=solute_topology,
                solvent_offmol=solvent_offmol,
                solvation_settings=solvation_settings,
                neutralize=solvent_component.neutralize,
                ion_concentration=solvent_component.ion_concentration,
            )
        elif isinstance(solvation_settings, InterchangeOpenMMSolvationSettings):
            topology = openmm_solvation(
                solute_topology=solute_topology,
                solvent_offmol=solvent_offmol,
                solvation_settings=solvation_settings,
                neutralize=solvent_component.neutralize,
                ion_concentration=solvent_component.ion_concentration,
            )
        else:
            raise ValueError("Unknown solvation method")

    else:  # no solvent case
        topology = Topology.from_molecules(topology_molecules)

    topology = _post_process_topology(
        pre_topology=topology,
        smc_components=smc_components,
        solvent_component=solvent_component,
        protein_component=protein_component,
    )

    # Run validation checks on inputs to Interchange
    # Examples: https://github.com/openforcefield/openff-interchange/issues/1058
    unique_charged_mols = _check_and_deduplicate_charged_mols(charged_mols)

    # ff14sb can end up with overlapping parameters, so split things
    # if necessary
    if any(["ff14sb" in name for name in ffsettings.forcefields]):
        interchange = _protein_split_combine_interchange(
            input_topology=topology,
            charge_from_molecules=unique_charged_mols,
            protein_component=protein_component,
            ffsettings=ffsettings,
        )
    else:
        force_field = _get_force_field(ffsettings=ffsettings, exclude_ff14sb=True)
        interchange = force_field.create_interchange(
            topology=topology,
            charge_from_molecules=unique_charged_mols,
        )

    # get the comp_resids dict
    comp_resids = _get_comp_resids(
        interchange=interchange,
        smc_components=smc_components,
        solvent_component=solvent_component,
        protein_component=protein_component,
    )

    # Finally we fill in comp_resids for any virtual sites we have
    _fill_vsite_compresids(interchange, comp_resids)

    return interchange, comp_resids
