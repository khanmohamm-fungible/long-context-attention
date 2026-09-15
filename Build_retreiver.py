from dataset import documents
from retrieval import FAISSVectorRetriever

retriever = FAISSVectorRetriever()

for position, document in enumerate(documents):
    retriever.upsert(
        document_id=f"doc_{position}",
        text=document,
        metadata={},
    )

results = retriever.search(
    query="What can make driving dangerous?",
    top_k=2,
)

for result in results:
    print("-", result.text)