from pathlib import Path

import dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

parent = Path(__file__).parents[1]
REVIEWS_CSV_PATH = parent / "data/reviews.csv"
REVIEWS_CHROMA_PATH = str(parent / "chroma_data")

dotenv.load_dotenv()

# In practice, if you're embedding a large document, you should use a text
# splitter. For this example, you can embed each review individually because
# they're relatively small.
loader = CSVLoader(file_path=REVIEWS_CSV_PATH, source_column="review")
reviews = loader.load()

reviews_vector_db = Chroma.from_documents(
    reviews, OpenAIEmbeddings(), persist_directory=REVIEWS_CHROMA_PATH
)
