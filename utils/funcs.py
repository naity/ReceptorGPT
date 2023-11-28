import re
import torch
import requests
import py3Dmol
import subprocess
import pandas as pd
import numpy as np
import streamlit as st
import importlib.resources as importlib_resources

from utils import *
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel


def extract_substring_between_parentheses(s):
    """Extracts the substring between parentheses from a string using regular expressions.

    Args:
      string: The string to extract the substring from.

    Returns:
      The substring between parentheses, or the original string if there are no
      parentheses in the string.
    """

    pattern = r"\(([^)]+)\)"
    match = re.search(pattern, s)
    if match:
        return match.group(1)
    else:
        return s


@st.cache_resource(show_spinner="Loading models...")
def get_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name)
    model.to(DEVICE)
    return tokenizer, model


def get_embeddings(tokenizer, model, seqs):
    """Get embeddings for a list of sequences."""
    inputs = tokenizer(seqs, padding="longest", return_tensors="pt").to(DEVICE)
    batch_lens = (inputs["input_ids"] != tokenizer.get_vocab()["<pad>"]).sum(dim=1)

    with torch.no_grad():
        results = model(**inputs)
    token_representations = results.last_hidden_state

    # Generate per-sequence representations via averaging
    # First token is CLS, last token is EOS
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(
            token_representations[i, 1 : tokens_len - 1].mean(0).numpy(force=True)
        )

    return np.array(sequence_representations)


def upsert_to_collection(collection, ids, embeddings, metadatas, batch_size=10000):
    """
    Upsert items to collection.
    New items will be added, existing items will be updated.
    """
    assert (
        len(ids) == len(embeddings) == len(metadatas)
    ), "The lengths of ids, embeddings, and metadatas must match"

    for i in tqdm(range(0, len(ids), batch_size)):
        collection.upsert(
            ids=ids[i : i + batch_size],
            embeddings=embeddings[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )


def search_embedding(collection, embedding, top_k):
    """Search a single embedding against the TCR vector database."""
    results = collection.query(
        query_embeddings=embedding,
        n_results=top_k,
        include=["metadatas", "distances"],
    )

    # return the first items
    ids = results["ids"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    return {"ids": ids, "metadatas": metadatas, "distances": distances}


def format_result(tcr_matches):
    """format response"""
    df = pd.DataFrame(tcr_matches["metadatas"])
    df["Similarity Score"] = [min(1, 1 - d) for d in tcr_matches["distances"]]

    # rearrange columns
    df = df[
        [
            "Similarity Score",
            "Species",
            "Antigen Epitope",
            "Antigen Protein",
            "Antigen Source",
            "CDR3.beta.aa",
            "TRBV",
            "TRBJ",
            "Reference",
            "Database",
        ]
    ]
    return df.sort_values("Similarity Score", ascending=False)


def run_stitchr(tcr, verbose=False):
    species = tcr["Species"].lower()

    v = tcr["TRBV"]
    j = tcr["TRBJ"]
    cdr3 = tcr["CDR3.beta.aa"]

    if species not in ("human", "mouse"):
        if verbose:
            print("Species not supported.")
        return None

    # empyt strings
    if v == "" or j == "" or cdr3 == "":
        if verbose:
            print("Missing values.")
        return None

    cmd = f"stitchr -s {species} -v {v} -j {j} -cdr3 {cdr3} -m AA"
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True, check=True)
    except Exception as e:
        if verbose:
            print(e)
        # return None if stitchr fails
        return None
    return result.stdout.decode("utf-8").strip()


def fold_sequence(sequence):
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
    }

    return requests.post(
        "https://api.esmatlas.com/foldSequence/v1/pdb/",
        headers=headers,
        data=sequence,
        # verify=False,
    )


def generate_3d_view(response):
    view = py3Dmol.view()
    view.addModelsAsFrames(response.text)
    view.setStyle({"model": -1}, {"cartoon": {"color": "spectrum"}})
    view.zoomTo()
    return view
