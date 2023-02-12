import pandas as pd
import numpy as np
import os
import json,pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from utils import *







def create(dataset_index):
    pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                     'X']

    pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
    pro_res_aromatic_table = ['F', 'W', 'Y']
    pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
    pro_res_acidic_charged_table = ['D', 'E']
    pro_res_basic_charged_table = ['H', 'K', 'R']

    res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                        'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                        'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}

    res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                     'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                     'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}

    res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                     'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                     'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}

    res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                     'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                     'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}

    res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                    'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                    'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}

    res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                                 'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                                 'T': 13, 'V': 79, 'W': 84, 'Y': 49}

    res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                                 'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                                 'T': 13, 'V': 76, 'W': 97, 'Y': 63}

    ###########################################################################################################
    #

    def seq_cat(prot):
        x = np.zeros(max_seq_len)
        for i, ch in enumerate(prot[:max_seq_len]):
            x[i] = seq_dict[ch]
        return x

    def smiles_cat(drug):
        x = np.zeros(max_smi_len)
        for i, ch in enumerate(drug[:max_smi_len]):
            x[i] = CHARCANSMISET[ch]
        return x

    # nomarlize
    def dic_normalize(dic):
        # print(dic)
        max_value = dic[max(dic, key=dic.get)]
        min_value = dic[min(dic, key=dic.get)]
        # print(max_value)
        interval = float(max_value) - float(min_value)
        for key in dic.keys():
            dic[key] = (dic[key] - min_value) / interval
        dic['X'] = (max_value + min_value) / 2.0
        return dic

    def residue_features(residue):
        res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0,
                         1 if residue in pro_res_polar_neutral_table else 0,
                         1 if residue in pro_res_acidic_charged_table else 0,
                         1 if residue in pro_res_basic_charged_table else 0]
        res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue],
                         res_pkx_table[residue],
                         res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
        # print(np.array(res_property1 + res_property2).shape)
        return np.array(res_property1 + res_property2)

    # mol atom feature for mol graph
    def atom_features(atom):
        # 44 +11 +11 +11 +1
        return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                              ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
                                               'As',
                                               'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                               'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
                                               'Cr',
                                               'Pt', 'Hg', 'Pb', 'X']) +
                        one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                        one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                        [atom.GetIsAromatic()])

    # one ont encoding
    def one_of_k_encoding(x, allowable_set):
        if x not in allowable_set:
            # print(x)
            raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
        return list(map(lambda s: x == s, allowable_set))

    def one_of_k_encoding_unk(x, allowable_set):
        '''Maps inputs not in the allowable set to the last element.'''
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: x == s, allowable_set))

    # mol smile to mol graph edge index
    def smile_to_graph(smile):
        mol = Chem.MolFromSmiles(smile)

        c_size = mol.GetNumAtoms()

        features = []
        for atom in mol.GetAtoms():
            feature = atom_features(atom)
            features.append(feature / sum(feature))

        edges = []
        for bond in mol.GetBonds():
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        g = nx.Graph(edges).to_directed()
        edge_index = []
        mol_adj = np.zeros((c_size, c_size))
        for e1, e2 in g.edges:
            mol_adj[e1, e2] = 1
            # edge_index.append([e1, e2])
        mol_adj += np.matrix(np.eye(mol_adj.shape[0]))
        index_row, index_col = np.where(mol_adj >= 0.5)
        for i, j in zip(index_row, index_col):
            edge_index.append([i, j])
        # print('smile_to_graph')
        # print(np.array(features).shape)
        return c_size, features, edge_index

    # target feature for target graph
    def PSSM_calculation(aln_file, pro_seq):
        pfm_mat = np.zeros((len(pro_res_table), len(pro_seq)))
        with open(aln_file, 'r') as f:
            line_count = len(f.readlines())
            for line in f.readlines():
                if len(line) != len(pro_seq):
                    print('error', len(line), len(pro_seq))
                    continue
                count = 0
                for res in line:
                    if res not in pro_res_table:
                        count += 1
                        continue
                    pfm_mat[pro_res_table.index(res), count] += 1
                    count += 1
        # ppm_mat = pfm_mat / float(line_count)
        pseudocount = 0.8
        ppm_mat = (pfm_mat + pseudocount / 4) / (float(line_count) + pseudocount)
        pssm_mat = ppm_mat
        # k = float(len(pro_res_table))
        # pwm_mat = np.log2(ppm_mat / (1.0 / k))
        # pssm_mat = pwm_mat
        # print(pssm_mat)
        return pssm_mat

    def seq_feature(pro_seq):
        pro_hot = np.zeros((len(pro_seq), len(pro_res_table)))
        pro_property = np.zeros((len(pro_seq), 12))
        for i in range(len(pro_seq)):
            # if 'X' in pro_seq:
            #     print(pro_seq)
            pro_hot[i,] = one_of_k_encoding(pro_seq[i], pro_res_table)
            pro_property[i,] = residue_features(pro_seq[i])
        return np.concatenate((pro_hot, pro_property), axis=1)

    def target_feature(aln_file, pro_seq):
        pssm = PSSM_calculation(aln_file, pro_seq)
        other_feature = seq_feature(pro_seq)
        # print('target_feature')
        # print(pssm.shape)
        # print(other_feature.shape)

        # print(other_feature.shape)
        # return other_feature
        return np.concatenate((np.transpose(pssm, (1, 0)), other_feature), axis=1)

    # target aln file save in data/dataset/aln
    def target_to_feature(target_key, target_sequence, aln_dir):
        # aln_dir = 'data/' + dataset + '/aln'
        aln_file = os.path.join(aln_dir, target_key + '.aln')
        # if 'X' in target_sequence:
        #     print(target_key)
        feature = target_feature(aln_file, target_sequence)
        return feature

    # pconsc4 predicted contact map save in data/dataset/pconsc4
    def target_to_graph(target_key, target_sequence, contact_dir, aln_dir):
        target_edge_index = []
        target_size = len(target_sequence)
        # contact_dir = 'data/' + dataset + '/pconsc4'
        contact_file = os.path.join(contact_dir, target_key + '.npy')
        contact_map = np.load(contact_file)
        contact_map += np.matrix(np.eye(contact_map.shape[0]))
        index_row, index_col = np.where(contact_map >= 0.5)
        for i, j in zip(index_row, index_col):
            target_edge_index.append([i, j])
        target_feature = target_to_feature(target_key, target_sequence, aln_dir)
        target_edge_index = np.array(target_edge_index)
        return target_size, target_feature, target_edge_index

    # to judge whether the required files exist
    def valid_target(key, dataset):
        contact_dir = 'data/' + dataset + '/pconsc4'
        aln_dir = 'data/' + dataset + '/aln'
        contact_file = os.path.join(contact_dir, key + '.npy')
        aln_file = os.path.join(aln_dir, key + '.aln')
        # print(contact_file, aln_file)
        if os.path.exists(contact_file) and os.path.exists(aln_file):
            return True
        else:
            return False

    #
    ###########################################################################################################

    def atom_features(atom):
        return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                              ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
                                               'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co',
                                               'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn',
                                               'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                        one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                        one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                        [atom.GetIsAromatic()])

    def one_of_k_encoding(x, allowable_set):
        if x not in allowable_set:
            raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
        return list(map(lambda s: x == s, allowable_set))

    def one_of_k_encoding_unk(x, allowable_set):
        """Maps inputs not in the allowable set to the last element."""
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: x == s, allowable_set))

    def smile_to_graph(smile):
        mol = Chem.MolFromSmiles(smile)

        c_size = mol.GetNumAtoms()

        features = []
        for atom in mol.GetAtoms():
            feature = atom_features(atom)
            features.append(feature / sum(feature))

        edges = []
        for bond in mol.GetBonds():
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        g = nx.Graph(edges).to_directed()
        edge_index = []
        for e1, e2 in g.edges:
            edge_index.append([e1, e2])

        return c_size, features, edge_index

    res_weight_table = dic_normalize(res_weight_table)
    res_pka_table = dic_normalize(res_pka_table)
    res_pkb_table = dic_normalize(res_pkb_table)
    res_pkx_table = dic_normalize(res_pkx_table)
    res_pl_table = dic_normalize(res_pl_table)
    res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
    res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)

    # from DeepDTA data
    all_prots = []
    datasets = [['kiba','davis'][dataset_index]]
    for dataset in datasets:
        print('convert data from DeepDTA for ', dataset)
        fpath = 'data/' + dataset + '/'
        train_fold = json.load(open(fpath + "folds/train_fold_setting1.txt"))
        train_fold = [ee for e in train_fold for ee in e ]
        valid_fold = json.load(open(fpath + "folds/test_fold_setting1.txt"))
        # 药物文件：pubchem号+smiles表达式
        ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
        # 靶标文件：靶标名+序列表达式
        proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
        affinity = pickle.load(open(fpath + "Y","rb"), encoding='latin1')
        # zll 药物smile，按照ligands_can.txt中顺序存储
        drugs = []
        # zll 蛋白质字母序列，按照proteins.txt中顺序存储
        prots = []
        target_key = []
        for d in ligands.keys():
            lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]),isomericSmiles=True)
            drugs.append(lg)
        for t in proteins.keys():
            prots.append(proteins[t])
            target_key.append(t)
        if dataset == 'davis':
            affinity = [-np.log10(y/1e9) for y in affinity]
        affinity = np.asarray(affinity)
        opts = ['train','test']
        for opt in opts:
            rows, cols = np.where(np.isnan(affinity)==False)
            if opt=='train':
                rows,cols = rows[train_fold], cols[train_fold]
            elif opt=='test':
                rows,cols = rows[valid_fold], cols[valid_fold]
            with open('data/' + dataset + '_' + opt + '.csv', 'w') as f:
                f.write('compound_iso_smiles,target_sequence,target_name,affinity\n')
                for pair_ind in range(len(rows)):
                    if not valid_target(target_key[cols[pair_ind]], dataset):  # ensure the contact and aln files exists
                        continue
                    ls = []
                    ls += [ drugs[rows[pair_ind]]  ]
                    ls += [ prots[cols[pair_ind]]  ]
                    ls += [ target_key[cols[pair_ind]]]
                    ls += [ affinity[rows[pair_ind],cols[pair_ind]]  ]
                    f.write(','.join(map(str,ls)) + '\n')
        print('\ndataset:', dataset)
        print('train_fold:', len(train_fold))
        print('test_fold:', len(valid_fold))
        print('len(set(drugs)),len(set(prots)):', len(set(drugs)),len(set(prots)))
        all_prots += list(set(prots))


    seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
    seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}
    seq_dict_len = len(seq_dict)
    max_seq_len = 850

    # label drug
    max_smi_len = 100
    CHARCANSMISET = { "#": 1, "%": 2, ")": 3, "(": 4, "+": 5, "-": 6,
                      ".": 7, "1": 8, "0": 9, "3": 10, "2": 11, "5": 12,
                      "4": 13, "7": 14, "6": 15, "9": 16, "8": 17, "=": 18,
                      "A": 19, "C": 20, "B": 21, "E": 22, "D": 23, "G": 24,
                      "F": 25, "I": 26, "H": 27, "K": 28, "M": 29, "L": 30,
                      "O": 31, "N": 32, "P": 33, "S": 34, "R": 35, "U": 36,
                      "T": 37, "W": 38, "V": 39, "Y": 40, "[": 41, "Z": 42,
                      "]": 43, "_": 44, "a": 45, "c": 46, "b": 47, "e": 48,
                      "d": 49, "g": 50, "f": 51, "i": 52, "h": 53, "m": 54,
                      "l": 55, "o": 56, "n": 57, "s": 58, "r": 59, "u": 60,
                      "t": 61, "y": 62}
    CHARCANSMILEN = 62

    compound_iso_smiles = []
    target_name_key = []

    for dt_name in [['kiba','davis'][dataset_index]]:

        opts = ['train','test']
        for opt in opts:
            df = pd.read_csv('data/' + dt_name + '_' + opt + '.csv')
            compound_iso_smiles += list( df['compound_iso_smiles'] )
            target_name_key += list(df['target_name'])
    compound_iso_smiles = set(compound_iso_smiles)

    smile_graph = {}
    for smile in compound_iso_smiles:
        g = smile_to_graph(smile)
        smile_graph[smile] = g



    datasets = [['kiba','davis'][dataset_index]]
    # convert to PyTorch data format
    for dataset in datasets:
        # processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
        # processed_data_file_test = 'data/processed/' + dataset + '_test.pt'

        dataset_path = 'data/' + dataset + '/'
        proteins = json.load(open(dataset_path + 'proteins.txt'), object_pairs_hook=OrderedDict)
        msa_path = 'data/' + dataset + '/aln'
        contac_path = 'data/' + dataset + '/pconsc4'

        # if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):

        target_name_key = set(target_name_key)
        target_graph = {}
        for key in target_name_key:
            if not valid_target(key, dataset):  # ensure the contact and aln files exists
                continue
            g = target_to_graph(key, proteins[key], contac_path, msa_path)
            target_graph[key] = g


        df = pd.read_csv('data/' + dataset + '_train.csv')
        train_drugs, train_prots,  train_Y = list(df['compound_iso_smiles']),list(df['target_sequence']),list(df['affinity'])
        train_target_key = list(df['target_name'])
        train_target_key = np.asarray(train_target_key)
        XT = [seq_cat(t) for t in train_prots]
        drug_smiles_label = [smiles_cat(t) for t in train_drugs]
        train_drugs, train_prots,  train_Y = np.asarray(train_drugs), np.asarray(XT), np.asarray(train_Y)
        train_drugs_smi = np.asarray(drug_smiles_label)

        df = pd.read_csv('data/' + dataset + '_test.csv')
        test_drugs, test_prots,  test_Y = list(df['compound_iso_smiles']),list(df['target_sequence']),list(df['affinity'])
        test_target_key = list(df['target_name'])
        test_target_key = np.asarray(test_target_key)
        XT = [seq_cat(t) for t in test_prots]
        drug_smiles_label = [smiles_cat(t) for t in test_drugs]
        test_drugs, test_prots,  test_Y = np.asarray(test_drugs), np.asarray(XT), np.asarray(test_Y)
        test_drugs_smi = np.asarray(drug_smiles_label)

        # make data PyTorch Geometric ready
        print('preparing ', dataset + '_train.pt in pytorch format!')
        train_data = TestbedDataset(root='data', dataset=dataset+'_train', xd=train_drugs, smi=train_drugs_smi, xt=train_prots, y=train_Y,smile_graph=smile_graph,
                                    target_graph=target_graph, target_key=train_target_key)
        print('preparing ', dataset + '_test.pt in pytorch format!')
        test_data = TestbedDataset(root='data', dataset=dataset+'_test', xd=test_drugs, smi=test_drugs_smi, xt=test_prots, y=test_Y,smile_graph=smile_graph,
                                   target_graph=target_graph, target_key=test_target_key)

        # print(processed_data_file_train, ' and ', processed_data_file_test, ' have been created')
        return train_data,test_data
        # else:
        #     print(processed_data_file_train, ' and ', processed_data_file_test, ' are already created')
