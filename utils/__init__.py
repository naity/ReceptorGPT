import torch

from pathlib import Path

DATA = Path(__file__).parent.parent / "data"

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# df_cols = [
#     "Similarity Score",
#     "Antigen Epitope",
#     "Antigen Protein",
#     "Antigen Source",
#     "Species",
#     "CDR3.alpha.aa",
#     "TRAV",
#     "TRAJ",
#     "CDR3.beta.aa",
#     "TRBV",
#     "TRBJ",
#     "Reference",
# ]
