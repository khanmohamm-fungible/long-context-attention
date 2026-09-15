from dataclasses import dataclass


documents = [
    "Sudden acceleration can make driving unsafe.",
    "Sharp turns may increase the probability of losing control.",
    "Braking too late can increase collision risk.",
    "Nitrogen deficiency often causes yellowing in older leaves.",
    "Phosphorus deficiency can affect root development.",
    "Paris is the capital of France."
]


@dataclass(frozen=True)
class RetrievalExample:
    """A causal-LM sample whose answer is available only in external evidence."""

    document_id: str
    query: str
    prompt: str
    answer: str
    text: str


def make_synthetic_retrieval_data(size: int = 256) -> tuple[list[str], list[RetrievalExample]]:
    """Create a reproducible RAG smoke corpus larger than the old six strings.

    The model input is a question followed by its answer; retrieval receives
    only the question. The evidence is external to the LM input, avoiding
    future-token query leakage while retaining a small offline test corpus.
    """
    if size < 8:
        raise ValueError("size must be at least 8")
    knowledge, examples = [], []
    for index in range(size):
        code = f"{(index * 7919 + 104729) % 100000:05d}"
        subject = f"archive item {index}"
        knowledge.append(f"The access code for {subject} is {code}.")
        query = f"What is the access code for {subject}?"
        prompt = f"Question: {query} Answer:"
        answer = f" {code}"
        examples.append(RetrievalExample(f"fact_{index}", query, prompt, answer, prompt + answer))
    return knowledge, examples
