from nltk.translate.bleu_score import sentence_bleu

with open ("xao_out2.csv", "r") as data:
    candidate = []
    referee = []
    for line in data.read().split("\n"):
        for i, sentence in enumerate(line.split("\t")):
            if i == 1:
                candidate.append([word for word in sentence.strip("<pad> ").strip("</s").split()])
            elif i  == 2:
                referee.append([word for word in sentence.split()])
    del candidate[0]
    del referee[0]


for r, c in zip(referee, candidate):
    score = sentence_bleu([r], c)
    # print(f'REFERENCE : {r} \nCANDIDATE : {c}')
    print(score)
