# Projet Indus M2

[Overleaf](https://fr.overleaf.com/project/615c9fe3e1a8120147b935a1)

[Google Sheets annotation](https://docs.google.com/spreadsheets/d/1RWf7DEeyveHVb6NeW_HFATMZE_G53V4QXr72DgQJ18I/edit?usp=sharing)

modèle : https://huggingface.co/yseop/FNP_T5_D2T_complete

## TODO

- [x] générer les phrases
- [x] se répartir le travail d'annotation / évaluation
- [x] faire les annotations (omissions, hallucinations)
- [ ] calculer les scores BLEU avec [nltk.align.bleu_score](https://www.nltk.org/_modules/nltk/align/bleu_score.html)
- [ ] comparer score BLEU et score humain (coefficient de Pearson, dispo dans Google Sheets)
- [ ] bilan : 
  - nombre d’erreurs
  - nombre de phrases «conformes»
  - votre estimation sur BLEU: est-ce que ce score est utile pour
    évaluer la génération ?

## Fichiers

`xao.csv` fichier d'origine, `yseop.py` script réalisé à partir du [tuto](https://huggingface.co/yseop/FNP_T5_D2T_complete)

`xao_out.csv` généré avec cette config :

```python
outputs = model.generate(input_ids,
                         do_sample=True,
                        top_p=0.82,
                        top_k=90,
                        early_stopping=True)
```

`xao_out2.csv` généré avec celle-ci :

```python
outputs = model.generate(input_ids, 
                         max_length=200, 
                         num_beams=2, repetition_penalty=2.5, 
                         top_k=50, top_p=0.98,
                         length_penalty=1.0,
                         early_stopping=True)
```

:woman_shrugging:

## Installer l'environnement virtuel 

Pour créer l'environnement virtuel et installer les requirements :

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

pour sortir de l'environnement virtuel :

```
deactivate
```

## Ressources

- Article 
- [BLEU score - Wikipedia](https://en.wikipedia.org/wiki/BLEU)
