import spacy
import pandas as pd
from spacy.training import Example

data = pd.read_csv("data_train.csv")
print("Start model training")
print("0/10569")
spacy.cli.download("ru_core_news_lg")
nlp = spacy.load("ru_core_news_lg")
nlp.remove_pipe("ner")
nlp.add_pipe("ner")
nlp.initialize()
c=1
examples = []
for row in data.iterrows():
    word = row[1]["word"]
    labels = row[1]["label"]
    predicted = nlp(word)
    dict={"entities": ["U-"+labels]}
    example = Example.from_dict(predicted, dict)
    examples.append(example)
    print(str(c) + "/10569")
    c+=1
print("Model training finished")
nlp.begin_training()
nlp.update(examples)
nlp.to_disk("my_model")