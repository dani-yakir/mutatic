# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNA_bert_3", trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained("zhihan1996/DNA_bert_3", trust_remote_code=True)

def embed(seq):
    tokens = tokenizer(seq, return_tensors="pt")
    with torch.no_grad():
        outputs = model.bert(**tokens)
    
    # Use CLS token embedding as sequence embedding
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding

# Sequences
seq1 = "AAA"
seq2 = "AAT"

emb1 = embed(seq1)
emb2 = embed(seq2)

# Cosine similarity
cos_sim = F.cosine_similarity(emb1, emb2).item()
print(f"Cosine similarity between '{seq1}' and '{seq2}':", cos_sim)