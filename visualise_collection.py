# Standard library imports
import os
import sys

# Third-party imports
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import plotly.express as px
from sklearn.decomposition import PCA


def load_vector_db(chroma_db_dir: str) -> Chroma:
    """Load and return a Chroma vector database from the specified directory.

    Args:
        chroma_db_dir: Directory path containing the Chroma database

    Returns:
        Chroma: Initialized vector database instance
    """
    return Chroma(
        persist_directory=chroma_db_dir,
        embedding_function=HuggingFaceEmbeddings(
            model_name="jinaai/jina-embeddings-v2-base-code",
            model_kwargs={"trust_remote_code": True},
        ),
    )


def visualize_chroma_db(chroma_db_dir: str) -> None:
    """Load and visualize a Chroma database.

    Args:
        chroma_db_dir: Directory path containing the Chroma database
    """
    vectordb = load_vector_db(chroma_db_dir)
    visualize_collection(vectordb)


def visualize_collection(chroma_db: Chroma) -> None:
    """Create a 3D visualization of embeddings from a Chroma database.

    Args:
        chroma_db: Chroma database instance to visualize
    """
    embeddings = chroma_db._collection.get(
        include=['embeddings', 'metadatas'], 
        limit=900
    )

    vectors = embeddings['embeddings']
    metadatas = embeddings['metadatas']
    sources = [metadata['source'] for metadata in metadatas]
    
    langs = [
        metadata.get('language', 'unknown') 
        for metadata in metadatas
    ]

    pca = PCA(n_components=3)
    vis_dims = pca.fit_transform(vectors)
    
    fig = px.scatter_3d(
        x=vis_dims[:, 0],
        y=vis_dims[:, 1],
        z=vis_dims[:, 2],
        text=sources,
        color=langs,
        labels={'x': '', 'y': '', 'z': ''}, 
        title="Code Embeddings Visualization"
    )

    fig.show()


def main():
    """Main entry point for the visualization script."""
    if len(sys.argv) != 2:
        print("Usage: python visualize_chroma_db.py <directory>")
        sys.exit(1)

    chroma_db_dir = sys.argv[1]

    if not os.path.isdir(chroma_db_dir):
        print(f"Error: {chroma_db_dir} is not a valid directory.")
        sys.exit(1)

    visualize_chroma_db(chroma_db_dir)


if __name__ == "__main__":
    main()
