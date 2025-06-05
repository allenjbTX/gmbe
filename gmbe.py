#!/usr/bin/env python3
"""
Generalized Many-Body Expansion (GMBE) to arbitrary order for proteins,
using ORCA for electronic structure calculations and the Liu & Herbert
(2016) partition scheme (P1/P2) for fragmenting residues.

This script:
 1. Reads an input PDB of a protein.
 2. Fragments the protein into residue-based groups (P1 or P2 splitting).
 3. Generates all k-fold intersections of fragment groups up to order N.
 4. For each intersection (i.e., each "term" in the GMBE), builds a capped geometry,
    writes an ORCA input file, runs ORCA, and parses the single-point energy.
 5. Accumulates the GMBE energy using inclusion-exclusion coefficients:
       E_GMBE = sum_{k=1..N} [ (-1)^(k+1) * sum_{all intersections of size k} E(intersection) ]

Charges:
 - Base fragment (whole residue, or P2-main) uses the standard residue net charge:
     ASP, GLU: -1; LYS, ARG: +1; HIS: +0 (unless protonation variant).
 - For P2 partition, side-chain fragments are assigned zero charge (backbone/main carries full). 
 - Intersection charges = sum of base fragment charges in the combination.

Usage:
    python3 gmbe.py --pdb protein.pdb --partition P1 --order 3 \
          --method MP2 --basis def2-SVP --orca-path /path/to/orca \
          --workdir gmbe_out

Example:
    python3 gmbe.py \
      --pdb myprotein.pdb \
      --partition P2 \
      --order 2 \
      --method GFN2-xTB \
      --basis xtb \
      --orca-path /usr/local/orca/orca \
      --workdir run_gmbe

"""
import os
import sys
import argparse
import itertools
import subprocess
from pathlib import Path
from Bio.PDB import PDBParser
import numpy as np

# -------------------- Residue Charge Lookup --------------------
RESIDUE_CHARGES = {
    'ASP': -1,
    'GLU': -1,
    'LYS': +1,
    'ARG': +1,
    'HIS': 0,
    # assume HIS neutral by default; adjust if HIP present
}

# ------------------------- Utility Functions -------------------------

def vdw_radius(element):
    """Return van der Waals radius (Å) for common elements."""
    radii = {
        'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80,
        'P': 1.80, 'F': 1.47, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98
    }
    return radii.get(element, 1.75)


def define_residue_groups(structure, partition='P1'):
    """
    Define base fragment groups per Liu & Herbert (2016):
      - P1: one group per residue, cutting C(=O)-Cα.
      - P2: if residue has >15 heavy atoms, split into main-chain and side-chain.
    Also assign each base fragment an integer net charge based on residue name.

    Returns: dict { group_id: {'atoms': [Atom,...], 'charge': int} }.
    """
    groups = {}
    idx = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != ' ':  # skip hetero/water
                    continue
                resname = residue.get_resname()
                z_res = RESIDUE_CHARGES.get(resname, 0)
                heavy_atoms = [atom for atom in residue if atom.element != 'H']
                if partition == 'P2' and len(heavy_atoms) > 15:
                    # split into backbone (N, CA, C, O, CB) and sidechain
                    backbone_names = set(['N', 'CA', 'C', 'O', 'CB'])
                    backbone_atoms = [a for a in heavy_atoms if a.get_name() in backbone_names]
                    sidechain_atoms = [a for a in heavy_atoms if a.get_name() not in backbone_names]
                    main_id = f"res{idx:04d}_main"
                    # assign full residue charge to main; side gets 0
                    groups[main_id] = {'atoms': backbone_atoms, 'charge': z_res}
                    idx += 1
                    side_id = f"res{idx:04d}_side"
                    groups[side_id] = {'atoms': sidechain_atoms, 'charge': 0}
                    idx += 1
                else:
                    gid = f"res{idx:04d}"
                    # whole residue carries z_res
                    groups[gid] = {'atoms': heavy_atoms, 'charge': z_res}
                    idx += 1
    return groups


def find_intersection_atoms(frag_atoms_list):
    """
    Given a list of fragment atom-lists (each is a list of Atom objects),
    return the list of atoms present in all fragments (intersection by serial number).
    """
    # Convert each fragment's atoms to dict {serial: Atom}
    sets = [ {atom.get_serial_number(): atom for atom in frag} for frag in frag_atoms_list ]
    # Compute common serial numbers
    common_serials = set(sets[0].keys())
    for s in sets[1:]:
        common_serials &= set(s.keys())
    # Return the Atom objects corresponding to common_serials (from first dict)
    return [sets[0][ser] for ser in sorted(common_serials)]


def add_caps(atom_list, all_atoms):
    """
    Given a list of Atom objects (the intersection), identify severed bonds to atoms
    not in the list and cap each severed bond by placing a hydrogen.
    Returns: list of (element, x, y, z) including original atoms + caps.
    """
    inter_serials = {atom.get_serial_number() for atom in atom_list}
    capped = []
    all_nonH = [atom for atom in all_atoms if atom.element != 'H']
    for atom in atom_list:
        elem = atom.element
        coord = atom.get_coord()
        capped.append((elem, coord[0], coord[1], coord[2]))
        for neigh in all_nonH:
            if neigh.get_serial_number() in inter_serials:
                continue
            dist = np.linalg.norm(atom.get_coord() - neigh.get_coord())
            if dist < 1.8:
                # severed bond → place H cap
                R1 = vdw_radius(atom.element)
                R2 = vdw_radius(neigh.element)
                RH = vdw_radius('H')
                r1 = atom.get_coord()
                r2 = neigh.get_coord()
                direction = (r1 - r2) / np.linalg.norm(r1 - r2)
                r_cap = r1 + direction * (R1 + RH)
                capped.append(('H', r_cap[0], r_cap[1], r_cap[2]))
    return capped


def write_xyz(atom_coords, filename):
    """Write list of (elem, x, y, z) to an XYZ file."""
    with open(filename, 'w') as f:
        f.write(f"{len(atom_coords)}")
        f.write(f"{filename}")
        for elem, x, y, z in atom_coords:
            f.write(f"{elem} {x:.6f} {y:.6f} {z:.6f}")


def generate_orca_input(workdir, frag_id, atom_coords, method, basis, charge, mult, embedding_charges=None):
    """
    Create ORCA input file for a given fragment or intersection.
    """
    filename = f"{frag_id}.inp"
    filepath = workdir / filename
    with open(filepath, 'w') as f:
        if basis:
            f.write(f"! {method} {basis} TightSCF\n")
        else:
            f.write(f"! {method} TightSCF\n")
        f.write("%pal nprocs 1 end\n")
        if embedding_charges:
            f.write("%pointcharges\n")
            for (x, y, z), q in embedding_charges.items():
                f.write(f"   {q:>10.5f} {x:>10.5f} {y:>10.5f} {z:>10.5f}\n")
            f.write("end\n")
        f.write(f"* xyz {charge} {mult}\n")
        for elem, x, y, z in atom_coords:
            f.write(f"  {elem} {x:>10.5f} {y:>10.5f} {z:>10.5f}\n")
        f.write("*\n")
    return filepath


def run_orca(orcapath, input_file):
    """Run ORCA point calculation; return output filename."""
    # ensure absolute paths so cwd changes don't break file locations
    input_path = os.path.abspath(input_file)
    output_file = input_path.replace('.inp', '.out')
    # make sure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as out:
        subprocess.run([orcapath, input_path], stdout=out, stderr=subprocess.STDOUT)
    return output_file


def parse_orca_energy(output_file):
    """Extract FINAL SINGLE POINT ENERGY from ORCA output."""
    energy = None
    with open(output_file) as f:
        for line in f:
            if 'FINAL SINGLE POINT ENERGY' in line:
                energy = float(line.split()[-1])
    return energy

# ----------------------- Main GMBE Workflow -------------------------

def main():
    parser = argparse.ArgumentParser(
        description='GMBE to arbitrary order for proteins with ORCA')
    parser.add_argument('--pdb', required=True, help='Protein PDB file')
    parser.add_argument('--partition', choices=['P1', 'P2'], default='P1', help='Partition scheme')
    parser.add_argument('--order', type=int, required=True, help='Max GMBE order (1 = monomer, 2 = pairs, etc)')
    parser.add_argument('--method', required=True, help='QM method for ORCA (e.g., MP2, B3LYP)')
    parser.add_argument('--basis', required=False, help='Basis set for ORCA (e.g., def2-SVP, cc-pVDZ)')
    parser.add_argument('--mult', type=int, default=1, help='Multiplicity for each fragment (default: 1)')
    parser.add_argument('--orca-path', default='orca', help='Path to ORCA executable')
    parser.add_argument('--workdir', default='gmbe', help='Working directory')
    args = parser.parse_args()

    # Create workdir and copy PDB
    wd = Path(args.workdir)
    wd.mkdir(parents=True, exist_ok=True)
    pdb_target = wd / 'protein.pdb'
    from shutil import copyfile
    copyfile(args.pdb, pdb_target)

    # Parse structure once
    parser_pdb = PDBParser(QUIET=True)
    structure = parser_pdb.get_structure('protein', str(pdb_target))
    all_atoms = [atom for atom in structure.get_atoms() if atom.element != 'H']

    # 1) Fragment via P1/P2, with charges
    base_groups = define_residue_groups(structure, partition=args.partition)
    frag_ids = sorted(base_groups.keys())
    num_frags = len(frag_ids)
    print(f"Defined {num_frags} base fragments (order-1 terms). Charges assigned per residue.")

    # 2) For k = 1..order, generate all combinations
    energies = {}  # map frozenset of frag_ids -> energy
    charges_map = {}
    total_gmbe = 0.0
    for k in range(1, args.order + 1):
        coeff = (-1)**(k+1)
        print(f"Generating all {k}-fragment intersections (coeff = {coeff})...")
        for combo in itertools.combinations(frag_ids, k):
            combo_key = frozenset(combo)
            # Determine intersection of atoms
            frag_atoms_list = [base_groups[fid]['atoms'] for fid in combo]
            # for k ≥ 2 use the union of the k fragments
            atoms_union = {a.get_serial_number(): a for sub in frag_atoms_list for a in sub}
            union_atoms = list(atoms_union.values())
            capped = add_caps(union_atoms, all_atoms)
            # Compute net charge for this intersection = sum of base fragment charges
            frag_charge = sum(base_groups[fid]['charge'] for fid in combo)
            charges_map[combo_key] = frag_charge
            # inter_atoms = find_intersection_atoms(frag_atoms_list)
            # if not inter_atoms:
            #     continue
            # Cap the intersection atoms
            #capped = add_caps(inter_atoms, all_atoms)
            # Write XYZ (optional)
            xyz_name = wd / f"inter_{'_'.join(combo)}.xyz"
            write_xyz(capped, str(xyz_name))
            # Create ORCA input with correct charge
            inp_name = wd / f"inter_{'_'.join(combo)}.inp"
            #os.chdir(wd)
            generate_orca_input(
                workdir=wd,
                frag_id=f"inter_{'_'.join(combo)}",
                atom_coords=capped,
                method=args.method,
                basis=args.basis,
                charge=frag_charge,
                mult=args.mult,
                embedding_charges=None
            )
            # Run ORCA
            out_name = run_orca(args.orca_path, str(inp_name))
            e = parse_orca_energy(str(out_name))
            energies[combo_key] = e
            total_gmbe += coeff * e
            print(f"  Combo {combo} (charge={frag_charge}) → E = {e:.6f} Ha → add {coeff*e:.6f}")
        print(f"Completed order {k} terms.")

    print(f"GMBE({args.order}) total energy = {total_gmbe:.6f} Ha")

    # Write summary to file
    summary = wd / 'gmbe_energy_summary.txt'
    with open(summary, 'w') as f:
        f.write(f"GMBE up to order {args.order}")
        f.write("Residue-based charges used. Intersection charge = sum of base charges.")
        f.write(f"Total GMBE Energy: {total_gmbe:.6f} Ha")

        for combo_key, e in energies.items():
            k = len(combo_key)
            coeff = (-1)**(k+1)
            cq = charges_map[combo_key]
            f.write(f"Term {sorted(combo_key)} (k={k}, charge={cq}): E = {e:.6f}, coeff = {coeff}, contribution = {coeff*e:.6f}")
            
    print(f"Summary written to {summary}")

if __name__ == '__main__':
    main()
