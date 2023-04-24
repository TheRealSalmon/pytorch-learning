import os
import subprocess
from copy import deepcopy
from tempfile import TemporaryDirectory
from typing import List, Tuple, Iterable

import matplotlib.pyplot as plt

from rdkit import Chem, RDLogger
from rdkit.Chem.AllChem import EmbedMolecule
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.rdMolTransforms import SetDihedralDeg

# import torch
# from torch_geometric.data import Data


RDLogger.DisableLog('rdApp.*')

FIX_ATOMS = """
$fix
   atoms: {}
$end
"""


def atom_to_node_feat_vec(atom: Chem.Atom) -> List[float]:
    return [
        atom.GetAtomicNum(),
        atom.GetFormalCharge(),
        atom.GetTotalDegree(),
        atom.GetTotalValence(),
        atom.GetIsAromatic() * 1.0,
        atom.IsInRingSize(3) * 1.0,
        atom.IsInRingSize(4) * 1.0,
        atom.IsInRingSize(5) * 1.0,
        atom.IsInRingSize(6) * 1.0,
        atom.IsInRingSize(7) * 1.0,
        atom.IsInRingSize(8) * 1.0,
    ]


def bonds_to_edge_indices(bonds: Iterable[Chem.Bond]) -> List[List[int]]:
    edge_indices = []
    for bond in bonds:
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices += [[i, j], [j, i]]

    return (
        torch.tensor(edge_indices)
        .t()
        .to(torch.long)
        .view(2, -1)
    )


def smi_to_pyg_data(smi: str):
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    node_feat_vecs = [atom_to_node_feat_vec(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(node_feat_vecs, dtype=torch.float).view(-1, len(node_feat_vecs[0]))
    edge_index = bonds_to_edge_indices(mol.GetBonds())

    return Data(x, edge_index)


def get_dihedral_indices(mol: Chem.Mol) -> List[Tuple[int]]:
    dihedral_bond_smarts = '*~*!@*~*'
    dihedral_bond = Chem.MolFromSmarts(dihedral_bond_smarts)

    all_dihedrals = mol.GetSubstructMatches(dihedral_bond)
    unique_dihedrals = []
    for dihedral in all_dihedrals:
        if not any([dihedral[1:3] == u_d[1:3] or dihedral[1:3] == tuple(reversed(u_d[1:3])) for u_d in unique_dihedrals]):
            unique_dihedrals.append(dihedral)

    return unique_dihedrals


def smi_to_xtb_data_package(smi, show_torsions=False):
    """
    Takes a SMILES string and returns a list of XYZ blocks evaluated at the
    GFN2-xTB level.

    The data package will contain torsion drives at a resolution of 15 degrees
    and one complete geometry optimization trajectory.
    """
    # create mol with Hs and conformer
    smi = Chem.CanonSmiles(smi)
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    EmbedMolecule(mol)

    # optimize conformer with xtb and collect trajectory
    with TemporaryDirectory() as tmp_dir:
        Chem.MolToXYZFile(mol, os.path.join(tmp_dir, 'input.xyz'))
        subprocess.run([
            'xtb',
            'input.xyz',
            '--opt',
            '--chrg',
            str(Chem.GetFormalCharge(mol)),
            '--gfn',
            '2',
            '--alpb',
            'water'
        ], cwd=tmp_dir, stdout=-1, stderr=-1)
        opt_trj_lines = open(os.path.join(tmp_dir, 'xtbopt.log')).readlines()
    block_size = mol.GetNumAtoms() + 2
    opt_trj_blocks = [
        ''.join(opt_trj_lines[i * block_size : (i + 1) * block_size])
        for i in range(len(opt_trj_lines) // block_size)
    ]

    # perform dihedral scans with xtb
    torsion_scan_blocks = []
    if show_torsions:
        IPythonConsole.drawOptions.addAtomIndices = True
        display(mol)
    for dihedral in get_dihedral_indices(mol):
        if show_torsions:
            angles = []
            energies = []
        for angle in range(0, 360, 15):
            mol_copy = deepcopy(mol)
            SetDihedralDeg(
                mol_copy.GetConformer(),
                *dihedral,
                angle,
            )
            with TemporaryDirectory() as tmp_dir:
                Chem.MolToXYZFile(mol_copy, os.path.join(tmp_dir, 'input.xyz'))
                constraints = FIX_ATOMS.format(','.join([str(idx) for idx in dihedral]))
                open(os.path.join(tmp_dir, 'xtb.inp'), 'w').write(constraints)
                proc = subprocess.run([
                    'xtb',
                    'input.xyz',
                    '--input',
                    'xtb.inp',
                    '--opt',
                    '--chrg',
                    str(Chem.GetFormalCharge(mol)),
                    '--gfn',
                    '2',
                    '--alpb',
                    'water'
                ], cwd=tmp_dir, stdout=-1, stderr=-1)
                if proc.returncode == 0:
                    torsion_scan_blocks.append(open(os.path.join(tmp_dir, 'xtbopt.xyz')).read())
            if show_torsions:
                angles.append(angle)
                energies.append(
                    float(torsion_scan_blocks[-1].split('\n')[1].split()[1])
                )
        if show_torsions:
            print('dihedral indicies:', dihedral)
            plt.scatter(angles, energies)
            plt.xlabel('angle (degrees)')
            plt.ylabel('energy (Hartrees)')
            plt.show()

    # reformat xyz blocks
    xyz_blocks = opt_trj_blocks + torsion_scan_blocks
    for i in range(len(xyz_blocks)):
        xyz_block = xyz_blocks[i]
        xyz_block_lines = xyz_block.split('\n')
        energy = float(xyz_block_lines[1].split()[1])
        xyz_block_lines[1] = str(energy)
        xyz_blocks[i] = '\n'.join(xyz_block_lines)

    return xyz_blocks
