import faiss
import numpy as np

dim = 384
index = faiss.IndexFlatL2(dim)
vectors = np.random.rand(10, dim).astype("float32")
index.add(vectors)
D, I = index.search(vectors[:1], k=3)
print(I)
