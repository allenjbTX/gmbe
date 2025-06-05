#!/usr/bin/env python3
"""
Generalized Many-Body Expansion (GMBE) to arbitrary order for proteins,
using ORCA for electronic structure calculations and the Liu & Herbert
(2016) partition scheme (P1/P2) for fragmenting residues.

This script:
 1. Reads an input PDB of a protein.
 2. Fragments the protein into residue-based groups (P1 or P2 splitting), preserving all atoms including hydrogens.
 3. Uses unions of these base fragments to build the GMBE up to order N.
 4. For each non-empty union (combo of base fragments), builds a capped geometry,
    writes an ORCA input file, runs ORCA, and parses the single-point energy.
 5. Computes the many-body corrections θ(I) via:
       θ(I) = E_union(I) - sum_{J ⊂ I, J ≠ I} θ(J)
    and sums θ(I) over all |I| ≤ N to get E_GMBE.

Charges:
 - Base fragment (whole residue, or P2-main) uses the standard residue net charge:
     ASP, GLU: -1; LYS, ARG: +1; HIS: 0.
 - For P2 partition, side-chain fragments are assigned zero charge.
 - Union-charge = sum of base fragment charges in the combination.

Usage:
    python3 gmbe.py --pdb protein.pdb --partition P1 --order 2 \
          --method GFN2-xTB --basis xtb --orca-path /path/to/orca \
          --workdir gmbe_out

"""
import os
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
    # assume HIS neutral by default
}

# ------------------------- Utility Functions -------------------------

def vdw_radius(element):
    """Return approximate van der Waals radius (Å) for common elements."""
    radii = {
        'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80,
        'P': 1.80, 'F': 1.47, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98
    }
    return radii.get(element, 1.75)


def define_residue_groups(structure, partition='P1'):
    """
    Define base fragment groups per Liu & Herbert (2016):
      - P1: one group per residue, including all atoms (heavy + H).
      - P2: if residue has >15 heavy atoms, split into main-chain and side-chain (each including any H bonded).
    Also assign net charge to each base fragment.

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
                all_atoms_res = [atom for atom in residue]  # include H
                heavy_atoms = [atom for atom in all_atoms_res if atom.element != 'H']
                if partition == 'P2' and len(heavy_atoms) > 15:
                    backbone_names = set(['N', 'CA', 'C', 'O', 'CB'])
                    main_atoms = []
                    side_atoms = []
                    for atom in all_atoms_res:
                        if atom.element == 'H':
                            parent_name = atom.get_parent().get_resname()
                            # attach H to heavy atom: include if bonded to a backbone heavy
                            # approximate by distance to any backbone heavy
                            assigned = False
                            for hb in heavy_atoms:
                                if hb.get_name() in backbone_names:
                                    dist = np.linalg.norm(atom.get_coord() - hb.get_coord())
                                    if dist < 1.2:  # H-bond length ~1.0, threshold 1.2
                                        main_atoms.append(atom)
                                        assigned = True
                                        break
                            if not assigned:
                                side_atoms.append(atom)
                        else:
                            if atom.get_name() in backbone_names:
                                main_atoms.append(atom)
                            else:
                                side_atoms.append(atom)
                    main_id = f"res{idx:04d}_main"
                    groups[main_id] = {'atoms': main_atoms, 'charge': z_res}
                    idx += 1
                    side_id = f"res{idx:04d}_side"
                    groups[side_id] = {'atoms': side_atoms, 'charge': 0}
                    idx += 1
                else:
                    gid = f"res{idx:04d}"
                    groups[gid] = {'atoms': all_atoms_res, 'charge': z_res}
                    idx += 1
    return groups


def find_union_atoms(frag_atoms_list):
    """
    Return the union of atom lists from frag_atoms_list (no duplicates), preserving coordinates.
    """
    union_dict = {}
    for frag in frag_atoms_list:
        for atom in frag:
            serial = atom.get_serial_number()
            if serial not in union_dict:
                union_dict[serial] = atom
    return list(union_dict.values())


def add_caps(atom_list, all_atoms):
    """
    Cap severed bonds by placing H at (R1+RH) along vector from atom toward severed neighbor.
    Returns list of (elem, x, y, z) including original atoms + caps.
    """
    inter_serials = {atom.get_serial_number() for atom in atom_list}
    capped = []
    # Build set of heavy atoms for detecting severed bonds
    heavy_all = [atom for atom in all_atoms if atom.element != 'H']
    for atom in atom_list:
        elem = atom.element
        coord = atom.get_coord()
        capped.append((elem, coord[0], coord[1], coord[2]))
        if elem == 'H':
            continue
        for neigh in heavy_all:
            if neigh.get_serial_number() in inter_serials:
                continue
            dist = np.linalg.norm(atom.get_coord() - neigh.get_coord())
            if dist < 1.8:
                R1 = vdw_radius(atom.element)
                RH = vdw_radius('H')
                r1 = atom.get_coord()
                r2 = neigh.get_coord()
                direction = (r2 - r1) / np.linalg.norm(r2 - r1)
                r_cap = r1 + direction * (R1 + RH)
                capped.append(('H', r_cap[0], r_cap[1], r_cap[2]))
    return capped


def write_xyz(atom_coords, filename):
    """Write list of (elem, x, y, z) to an XYZ file."""
    with open(filename, 'w') as f:
        f.write(f"{len(atom_coords)}\n")
        f.write(f"{filename}\n")
        for elem, x, y, z in atom_coords:
            f.write(f"{elem} {x:.6f} {y:.6f} {z:.6f}\n")


def generate_orca_input(workdir, frag_id, atom_coords, method, basis, charge, mult, embedding_charges=None):
    """
    Create ORCA input file for fragment.
    """
    filename = f"{frag_id}.inp"
    filepath = workdir / filename
    with open(filepath, 'w') as f:
        if basis:
            f.write(f"! {method} {basis} TightSCF NoRI\n")
        else:
            f.write(f"! {method} TightSCF NoRI\n")
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
    """Run ORCA; return output filename."""
    output_file = input_file.replace('.inp', '.out')
    with open(output_file, 'w') as out:
        subprocess.run([orcapath, input_file], stdout=out, stderr=subprocess.STDOUT)
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
    parser.add_argument('--order', type=int, required=True, help='Max GMBE order')
    parser.add_argument('--method', required=True, help='QM method for ORCA (e.g., MP2, B3LYP)')
    parser.add_argument('--basis', required=False, help='Basis set for ORCA (e.g., def2-SVP, cc-pVDZ)')
    parser.add_argument('--mult', type=int, default=1, help='Multiplicity for each fragment')
    parser.add_argument('--orca-path', default='orca', help='Path to ORCA executable')
    parser.add_argument('--workdir', default='gmbe_order', help='Working directory')
    args = parser.parse_args()

    # Prepare working directory
    wd = Path(args.workdir)
    wd.mkdir(parents=True, exist_ok=True)
    pdb_target = wd / 'protein.pdb'
    from shutil import copyfile
    copyfile(args.pdb, pdb_target)

    # Parse structure
    parser_pdb = PDBParser(QUIET=True)
    structure = parser_pdb.get_structure('protein', str(pdb_target))
    all_atoms = [atom for atom in structure.get_atoms()]

    # Fragment base groups
    base_groups = define_residue_groups(structure, partition=args.partition)
    frag_ids = sorted(base_groups.keys())
    print(f"Defined {len(frag_ids)} base fragments. Charges assigned per residue.")

    E_union = {}  # raw union energy for each subset
    theta = {}    # many-body correction for each subset
    total_gmbe = 0.0

    # Check order
    max_order = len(frag_ids)
    if args.order > max_order:
        print(f"Warning: order {args.order} > number of fragments ({max_order}); truncating to {max_order}.")
        args.order = max_order

    for k in range(1, args.order + 1):
        print(f"Computing all {k}-fold unions...")
        # Compute E_union for each subset of size k
        for combo in itertools.combinations(frag_ids, k):
            combo_key = frozenset(combo)
            frag_atoms_list = [base_groups[f]['atoms'] for f in combo]
            union_atoms = find_union_atoms(frag_atoms_list)
            if not union_atoms:
                continue
            frag_charge = sum(base_groups[f]['charge'] for f in combo)
            capped = add_caps(union_atoms, all_atoms)
            xyz_name = wd / f"union_{'_'.join(combo)}.xyz"
            write_xyz(capped, str(xyz_name))
            inp_name = wd / f"union_{'_'.join(combo)}.inp"
            #os.chdir(wd)
            generate_orca_input(
                workdir=wd,
                frag_id=f"union_{'_'.join(combo)}",
                atom_coords=capped,
                method=args.method,
                basis=args.basis,
                charge=frag_charge,
                mult=args.mult,
                embedding_charges=None
            )
            out_name = run_orca(args.orca_path, str(inp_name))
            e = parse_orca_energy(str(out_name))
            E_union[combo_key] = e
            print(f"  E_union({combo}) = {e:.6f} Ha")

        print(f"Computing θ contributions for order k={k}...")
        for combo in itertools.combinations(frag_ids, k):
            combo_key = frozenset(combo)
            if combo_key not in E_union:
                continue
            sum_subsets = 0.0
            for r in range(1, k):
                for sub in itertools.combinations(combo, r):
                    sum_subsets += theta[frozenset(sub)]
            theta_val = E_union[combo_key] - sum_subsets
            theta[combo_key] = theta_val
            total_gmbe += theta_val
            print(f"  θ({combo}) = {theta_val:.6f} Ha; cumulative E_GMBE = {total_gmbe:.6f}")
        print(f"Completed ΔE for k={k}.\n")

    print(f"Final GMBE({args.order}) total energy = {total_gmbe:.6f} Ha")

    summary = wd / 'gmbe_energy_summary.txt'
    with open(summary, 'w') as f:
        f.write(f"GMBE up to order {args.order}\n")
        f.write("Using θ(I) = E_union(I) - sum_{J subset I} θ(J)\n")
        f.write(f"Total GMBE Energy: {total_gmbe:.6f} Ha\n\n")
        for combo_key, th in theta.items():
            k = len(combo_key)
            cq = sum(base_groups[f]['charge'] for f in combo_key)
            f.write(f"θ({sorted(combo_key)}) (|I|={k}, charge={cq}) = {th:.6f}\n")
    print(f"Summary written to {summary}")

if __name__ == '__main__':
    main()
