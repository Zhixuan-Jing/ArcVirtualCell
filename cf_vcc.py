# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("embedding", type = str, help = "embedding path, None for no embedding")
# parser.add_argument("data", type = str, help = "data path")
# args = parser.parse_args()

import scanpy as sc
from cellflow.model import CellFlow
import requests
import pandas as pd
import torch
import pickle
from esm import pretrained
from UniProtMapper import ProtMapper
import random
class ESMConverter:
  def __init__(self, model:str):
    self.model, self.alphabet = pretrained.load_model_and_alphabet(model)
    self.batch_converter = self.alphabet.get_batch_converter()

  def convert(self, sequences):
    batch_labels, batch_strs, batch_tokens = self.batch_converter(sequences)
    with torch.no_grad():
      token_embeddings = self.model(batch_tokens, repr_layers=[33])
      embeddings = token_embeddings['representations'][33]
      average_embeddings = embeddings.mean(dim=1)
    return average_embeddings
  
def get_protein_sequence_by_gene(gene_name):
    mapper = ProtMapper()
    result, failed = mapper.get(
        ids=gene_name, from_db="Gene_Name", to_db="UniProtKB"
    )
    result = result[(result['Organism'] == "Homo sapiens (Human)")&(result['Reviewed'] == "reviewed")]
    protein = result.iloc[0]["Entry"]
    # print(protein)
    # Define the UniProt API endpoint
    sequence_url = f"https://www.uniprot.org/uniprot/{protein}.fasta"
    sequence_response = requests.get(sequence_url)
        
    if sequence_response.status_code == 200:
        # Extract and return the protein sequence
        sequence = ''.join(sequence_response.text.splitlines()[1:])
        return sequence
    else:
        return "NONE"
    
print("Load dataset...")
filePath = "./data/vcc_sample.h5ad"

adata = sc.read_h5ad(filePath)

adata.obs['control'] = [(lambda x: True if x == "non-targeting" else False)(x) for x in adata.obs['target_gene']]

# Parameters for preparing data
sample_rep = "X"
control_key = "control"
perturbation_covariates = {"gene": ("target_gene",)}
split_covariates = ["batch"]
perturbation_covariate_reps = {"gene": "gene_embedding"}
sample_covariates = None
sample_covariate_reps = None

flag = True
print("Load dataset complete")
print("Load embedding data...")
if flag:
    embedding = pickle.load(open("./paths/subsample_gene_embedding.pkl", "rb"))
else:
    # If not, prepare gene embeddings
    # Sort out target genes
    genes = adata.obs[adata.obs['control'] == False]['target_gene'].to_list()
    genes = list(set(genes))

    embedding = pd.DataFrame(columns=["gene", "protein", "embedding"])
    embedding["gene"] = genes
    embedding.index = genes
    embedding["protein"] = embedding['gene'].apply(get_protein_sequence_by_gene)
    converter = ESMConverter("esm2_t33_650M_UR50D")
    sequences = list(zip(embedding['gene'], embedding['protein']))
    em = []
    for s in sequences:
        em.append(converter.convert([s]))
    embedding['embedding'] = em
    # Save embedding to pickle
    pd.to_pickle(embedding,"subsample_gene_embedding.pkl")

embedding['embedding'] = embedding['embedding'].apply(torch.flatten)
adata.uns['gene_embedding'] = {}
for g in embedding['gene']:
    adata.uns['gene_embedding'][g] = embedding.loc[g]['embedding']

print("Load embedding complete")
print("Split train/test data...")
# Split train/test data
# Split train_test data
x = adata[adata.obs['control'] == True]
y = adata[adata.obs['control'] == False]
fraction = 0.2
# For runability test, sample little data
# x_t = sc.pp.sample(x, n = 1000, copy = True)
# y_t = sc.pp.sample(y, n = 5000, copy = True)
x_train = x[:int(fraction*x.n_obs), :]
y_train = y[:int(fraction*y.n_obs), :]
x_eval = x[int(fraction*x.n_obs):, :]
y_eval = y[int(fraction*y.n_obs):, :]

train = x_train.concatenate(y_train)
eval = x_eval.concatenate(y_eval)
x_eval.obs['target_gene'] = random.sample(list(y_eval.obs['target_gene']), x_eval.n_obs)

train.uns = adata.uns
eval.uns = adata.uns
print("Split complete")
print("Initialization...")
cf = CellFlow(train)
cf.prepare_data(
    sample_rep = sample_rep,
    control_key = control_key,
    perturbation_covariates = perturbation_covariates,
    perturbation_covariate_reps = perturbation_covariate_reps
)
cf.prepare_model()
cf.prepare_validation_data(eval, name = "test")
print("Initialization complete")
cf.train(num_iterations=10, batch_size = 1024)
cf.save("./models/","cf-default", overwrite=False)