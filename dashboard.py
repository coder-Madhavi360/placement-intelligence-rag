import os
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Evaluation Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Placement Intelligence RAG Evaluation Dashboard")

# ---------------------------------------------------
# Check evaluation files
# ---------------------------------------------------

if not os.path.exists("data/eval_results.csv"):
    st.error(
        "Evaluation results not found.\n\n"
        "Run:\n"
        "python scripts/evaluate.py"
    )
    st.stop()

# ---------------------------------------------------
# Load evaluation results
# ---------------------------------------------------

df = pd.read_csv("data/eval_results.csv")

# ---------------------------------------------------
# Top Metrics
# ---------------------------------------------------

total_queries = len(df)

avg_accuracy = (
    df["keyword_score"].mean() * 100
)

avg_latency = (
    df["latency_s"].mean()
)

avg_retrieval = (
    df["retrieval_quality"].mean()
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Queries Tested",
        total_queries
    )

with col2:
    st.metric(
        "Accuracy",
        f"{avg_accuracy:.1f}%"
    )

with col3:
    st.metric(
        "Avg Latency",
        f"{avg_latency:.2f}s"
    )

with col4:
    st.metric(
        "Retrieval Quality",
        f"{avg_retrieval:.2f}"
    )

st.divider()

# ---------------------------------------------------
# Difficulty Chart
# ---------------------------------------------------

st.subheader("Accuracy by Difficulty")

difficulty_scores = (
    df.groupby("difficulty")["keyword_score"]
      .mean()
      .reset_index()
)

st.bar_chart(
    difficulty_scores.set_index("difficulty")
)

# ---------------------------------------------------
# Latency Chart
# ---------------------------------------------------

st.subheader("Latency by Query")

latency_df = df[["id", "latency_s"]]

st.line_chart(
    latency_df.set_index("id")
)

# ---------------------------------------------------
# Multi-Hop Evaluation
# ---------------------------------------------------

if os.path.exists("data/multihop_results.csv"):

    mh = pd.read_csv(
        "data/multihop_results.csv"
    )

    multihop_score = (
        mh["keyword_score"].mean() * 100
    )

    st.subheader(
        "Multi-Hop Reasoning Performance"
    )

    st.metric(
        "Multi-Hop Accuracy",
        f"{multihop_score:.1f}%"
    )

# ---------------------------------------------------
# Failure Analysis
# ---------------------------------------------------

st.subheader("Failed Queries")

failed = df[
    df["keyword_score"] < 0.5
]

if len(failed) > 0:

    st.dataframe(
        failed[
            [
                "id",
                "query",
                "keyword_score",
                "difficulty"
            ]
        ]
    )

else:
    st.success(
        "No failed queries detected."
    )

# ---------------------------------------------------
# Full Evaluation Results
# ---------------------------------------------------

st.subheader("Complete Evaluation Results")

st.dataframe(
    df,
    use_container_width=True
)