from ogb.utils import smiles2graph
from ogb.lsc import PygPCQM4MDataset

ROOT = '../data'

pyg_dataset = PygPCQM4MDataset(root = ROOT, smiles2graph = smiles2graph)
