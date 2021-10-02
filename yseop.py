from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import csv

tokenizer = AutoTokenizer.from_pretrained("yseop/FNP_T5_D2T_complete")

model = AutoModelForSeq2SeqLM.from_pretrained("yseop/FNP_T5_D2T_complete")

p = 0.82
k = 90


def generate_sentence(triple):
    input_ids = tokenizer.encode(": {}".format(triple), return_tensors="pt")
    # outputs = model.generate(
    #     input_ids, do_sample=True, top_p=p, top_k=k, early_stopping=True
    # )
    outputs = model.generate(
        input_ids,
        max_length=200,
        num_beams=2,
        repetition_penalty=2.5,
        top_k=50,
        top_p=0.98,
        length_penalty=1.0,
        early_stopping=True,
    )
    return outputs


if __name__ == "__main__":
    # triple = [
    #     "Group profit | valIs | € 115.7 million && € 115.7 million | dTime | in 2019",
    # ]  exemple tuto
    with open("xao.csv", "r") as input, open("xao_out2.csv", "w") as output:
        reader = csv.reader(input, delimiter="\t")
        writer = csv.writer(output, delimiter="\t", quotechar='"')
        writer.writerow(
            (
                "Triplets",
                "Phrase générée",
                "Phrase attendue"  # ,
                # "Omission",
                # "Hallucination",
                # "BLEU",
            )
        )
        for i, row in enumerate(reader):
            _, expected_sentence, triple = row
            generated_sentence = tokenizer.decode(generate_sentence(triple)[0])
            output_row = (triple, generated_sentence, expected_sentence)
            print(i, output_row)
            writer.writerow(output_row)
