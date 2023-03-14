import fasttext

cutoff = 0.75
model = fasttext.load_model("/workspace/datasets/fasttext/title_model_100.bin")
with open('/workspace/datasets/fasttext/synonyms.csv', 'w') as s:
    with open('/workspace/datasets/fasttext/top_words.txt', 'r') as w:
        words = w.read().splitlines()
        for word in words:
            nn = model.get_nearest_neighbors(word)
            filtered_synonyms = [nn[1] for nn in nn if float(nn[0]) >= cutoff]
            if len(filtered_synonyms) > 0:
                joined = ','.join(filtered_synonyms)
                s.write(f'{word},{joined}\n')
