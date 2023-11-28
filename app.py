# sqlite3 version issue
__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")


import chromadb
import streamlit as st

from utils.funcs import *
from stmol import showmol
from streamlit import session_state as _state


# title image
title_image_path = "images/logo.jpg"

st.set_page_config(
    page_title="ReceptorGPT",
    page_icon=title_image_path,
)

# ESM model
tokenizer, model = get_model("facebook/esm2_t6_8M_UR50D")

# TCR database
client = chromadb.PersistentClient(path="models")
collection = client.get_collection(name="tcrs")

# side bar
with st.sidebar:
    st.title(":desktop_computer: Database")
    st.info(
        """
        Number of TCRs: **34,099**  
        Last updated: **11/25/2023**
        """
    )

    st.title("👋 Contact")
    st.info(
        """
        **[Yuan Tian](https://www.ytiancompbio.com)**:  
        [LinkedIn](https://www.linkedin.com/in/ytiancompbio/) | [Twitter](https://twitter.com/ytiancompbio) | [GitHub](https://github.com/naity)
        """
    )

    st.title(":open_book: Resources")
    st.info(
        """
        - [IEDB](https://www.iedb.org/)
        - [CEDAR](https://cedar.iedb.org/)
        - [VDJdb](https://vdjdb.cdr3.net/)
        - [McPAS-TCR](http://friedmanlab.weizmann.ac.il/McPAS-TCR/)
        - [ESM](https://github.com/facebookresearch/esm)
        """
    )

# title
left_col, right_col = st.columns([1, 2])
with left_col:
    st.image(str(title_image_path))
with right_col:
    st.markdown(
        '# <span style="color:#ff4500">ReceptorGPT</span>', unsafe_allow_html=True
    )
    st.markdown("### Discover TCR matches, antigen specificity, and structure with AI")

# app description
st.markdown(
    "<font size='4'> **ReceptorGPT** is an AI-powered app that uses Transformer-based protein language models to help you identify T cell receptor (TCR) matches, discover antigen specificity, and predict TCR structure. ReceptorGPT leverages embeddings to represent TCR sequences in a compact and informative way, allowing it to efficiently query a database of TCRs with known antigen specificity.</font>",
    unsafe_allow_html=True,
)
st.divider()


col1, col2 = st.columns([1, 1])
with col1:
    species = st.radio(
        "Species", ("Human", "Mouse"), index=0, horizontal=True, key="species"
    )
with col2:
    k = st.slider("Number of results to return", 1, 10, 5, key="k")

col1, col2, col3 = st.columns([1.1, 1, 1])
with col1:
    cdr3_beta_aa = st.text_input(
        "Beta Chain CDR3 amino acid sequence", "CASSESAGGYYNEQF", key="cdr3_beta_aa"
    )
with col2:
    trbv = st.text_input("Beta Chain Variable segment allele", "TRBV2", key="trbv")
with col3:
    trbj = st.text_input("Beta Chain Joining segment allele", "TRBJ2-1", key="trbj")

st.caption(
    ":bulb: Beta chain CDR3 amino acid sequences must be provided.",
)

run_fold = st.checkbox("Predict Structure", value=True, key="run_fold")
if _state.run_fold:
    st.caption(
        ":bulb: It is recommended to supply V and J gene segments for structure prediction.",
    )


def search_tcr():
    """Execute search and display results."""
    if not _state.cdr3_beta_aa:
        with container:
            st.warning(
                "Please provide Beta chain CDR3 amino acid sequences.",
                icon="⚠️",
            )
        return

    # format as a Series
    tcr = pd.Series(
        {
            "Species": _state.species,
            "CDR3.beta.aa": _state.cdr3_beta_aa.strip(),
            "TRBV": _state.trbv.strip(),
            "TRBJ": _state.trbj.strip(),
        }
    )

    try:
        # get embedding
        embedding = get_embeddings(tokenizer, model, [tcr["CDR3.beta.aa"]])
        # convert to list
        embedding = embedding.tolist()
        tcr_matches = search_embedding(collection, embedding, _state.k)
        df = format_result(tcr_matches)

        if _state.run_fold and tcr["TRBV"] and tcr["TRBJ"]:
            # try to get full sequence
            beta_aa = run_stitchr(tcr)
            if beta_aa is not None:
                tcr["beta.aa"] = beta_aa

        # display df
        with container:
            if len(df) > 0:
                st.markdown("**Top TCR matches:**")
                st.dataframe(
                    df,
                    column_config={
                        "Reference": st.column_config.LinkColumn("Reference"),
                        "Similarity Score": st.column_config.NumberColumn(
                            format="%.3f"
                        ),
                    },
                    hide_index=True,
                )

                # ESM fold
                if _state.run_fold:
                    structure_seq = (
                        tcr["beta.aa"] if "beta.aa" in tcr else tcr["CDR3.beta.aa"]
                    )
                    fold_response = fold_sequence(structure_seq)
                    view = generate_3d_view(fold_response)
                    st.markdown("##### Predicted structure:")
                    showmol(view, height=500, width=800)

            else:
                st.markdown(":warning: No matching TCRs were found in the database.")

    except:
        with container:
            st.warning(
                "Something went wrong, please refresh and try again",
                icon="⚠️",
            )


container = st.container()

with container:
    if st.button(
        "Search",
        type="primary",
    ):
        search_tcr()
