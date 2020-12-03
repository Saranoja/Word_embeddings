import re
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# with open('dataset_processed.txt') as file:
#     with open('dataset_processed2.txt', 'w') as file2:
#         i = 1000
#         for row in file:
#             file2.write(row)
#             i -= 1
#             if i == 0:
#                 break


word_tokenizer = re.compile("[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+")

with open('dataset_processed2.txt') as file:
    ar = np.array([])
    for row in file:
        words = word_tokenizer.findall(row)
        words = list(map(lambda word: word.lower(), words))
        ar = np.concatenate([ar, words])

print(len(ar))

with open('english_stopwords.txt') as file:
    for stopword in file:
        stopword = stopword.rstrip('\n')
        ar = np.delete(ar, np.argwhere(ar == stopword))

print(len(ar))

# define example
# data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']
# ar = data

print(ar)
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(ar)
print(integer_encoded)

# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
print(integer_encoded)

onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)
print(onehot_encoded.T)
print(onehot_encoded.shape)
