from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer


tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = "The driver suddenly applied the brakes."

encoded = tokenizer(
    text,
    return_tensors="pt"
)

input_ids = encoded["input_ids"]

print(input_ids)
print(input_ids.shape)

tokenizer.pad_token = tokenizer.eos_token


sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(sentences)
print(embeddings)
