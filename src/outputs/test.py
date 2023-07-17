import pickle

with open("embeddings.pickle", "rb") as new:
    d1 = pickle.load(new)
    
print(d1['names'])
