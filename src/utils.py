import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit import DataStructs

DESCRIPTOR_NAMES = [
    'MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 'TPSA',
    'NumRotatableBonds', 'RingCount', 'HeavyAtomCount'
]

FP_NAMES = [f'Morgan_{i}' for i in range(2048)]

ALL_FEATURE_NAMES = DESCRIPTOR_NAMES + FP_NAMES

def feature_extraction(smiles):
    """
    Safely converts a SMILES string into an exactly 2056-feature numpy array.
    First 8 values are physicochemical descriptors.
    Next 2048 values are Morgan Fingerprint bits (radius=2).
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.full((2056,), np.nan)
        
        # Extracted precisely in order of DESCRIPTOR_NAMES
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        num_h_donors = Descriptors.NumHDonors(mol)
        num_h_acceptors = Descriptors.NumHAcceptors(mol)
        tpsa = Descriptors.TPSA(mol)
        num_rotbonds = Descriptors.NumRotatableBonds(mol)
        ring_count = Descriptors.RingCount(mol)
        heavy_atoms = mol.GetNumHeavyAtoms()
        
        descriptors = np.array([
            mw, logp, num_h_donors, num_h_acceptors, tpsa, 
            num_rotbonds, ring_count, heavy_atoms
        ], dtype=np.float32)
        
        # Extract 2048-bit Morgan Fingerprint
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        fp_arr = np.zeros((2048,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, fp_arr)
        
        # Combine into complete feature vector
        combined_features = np.concatenate((descriptors, fp_arr))
        return combined_features
        
    except Exception:
        return np.full((2056,), np.nan)
