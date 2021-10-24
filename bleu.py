from nltk.translate.bleu_score import sentence_bleu
from sacremoses import MosesTokenizer, MosesDetokenizer
import string

mt_en = MosesTokenizer(lang='en')

with open ("xao_out2.csv", "r") as data:
    candidate = []
    referee = []
    for line in data.read().split("\n"):
        for i, sentence in enumerate(line.split("\t")):
            if i == 1:
                sentence = sentence.strip("<pad> ").strip("</s")
                candidate_tok = mt_en.tokenize(sentence, return_str=True, escape=False)
                candidate.append([word for word in candidate_tok.split() if word not in string.punctuation])
            elif i  == 2:
                referee_tok = mt_en.tokenize(sentence, return_str=True, escape=False)
                referee.append([word for word in referee_tok.split() if word not in string.punctuation])
    del candidate[0]
    del referee[0]


for r, c in zip(referee, candidate):
    score = sentence_bleu([r], c)
    # print(f'REFERENCE : {r} \nCANDIDATE : {c}')
    print(score)

# print(f'REFERENCE : {referee[14]} \nCANDIDATE : {candidate[14]}')
# print(sentence_bleu([referee[14]], candidate[14]))
