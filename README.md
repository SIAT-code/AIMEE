# AIMEE
## Reqirements:
- Pytorch=1.2.0
- torchvision=0.4.0
- cudatoolkit=10.0 (View version: cat /usr/local/cuda/version.txt)
- Python3
- Rdkit (https://www.rdkit.org/docs/GettingStartedInPython.html#getting-started-with-the-rdkit-in-python)
- transformers=2.7.0 (How to install: huggingface/transformers)
## Model architecture:
![model](https://github.com/Siat-Code/AIMEE/blob/main/image/model.jpg)
#### (1) Transformer pretrained model (for protein sequence)
- We use a large number of protein sequence data to pre train a modified transformer model, which is saved in the path `./pretrained_model/pytorch_model.pth`.
- The configuration of the model is saved in the path `./pretrained_model/bert_config.json`.
#### (2) GAT (for protein structure)
- The relevant files of this model are saved in the directory `./AttentiveFP`.
#### (3) GAT (for molecular graph)
- The relevant files of this model are saved in the directory `./ProEmb`, meanwhile, other models of proteins are also saved in this directory.
## Usage:
The entry of the whole program is in the `main.py`. It includes the selection of input and output files and the definition of parameters. It can run directly by `python main.py`.
## Cite:
If you use the code, please cite this paper:
> Hu, Fan, et al. "Structure Enhanced Protein-Drug Interaction Prediction using Transformer and Graph Embedding." 2020 IEEE International Conference on Bioinformatics and Biomedicine (BIBM). IEEE, 2020.
> 


