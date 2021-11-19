import json
import random
import spacy

from tqdm import tqdm
from settings import MODEL_DIR, JSON_FILE_PATH, EPOCHS


class NERTrainer:
    def __init__(self):
        self.iter = EPOCHS
        self.nlp = spacy.blank('en')
        print("Created blank 'en' model")
        if 'ner' not in self.nlp.pipe_names:
            self.ner = self.nlp.create_pipe('ner')
            self.nlp.add_pipe(self.ner, last=True)
        else:
            self.ner = self.nlp.get_pipe('ner')

    def train(self):
        init_train_data = json.load(open(JSON_FILE_PATH))
        train_data = []
        for t_data in init_train_data:
            text = t_data["text"]
            for t_lbl in t_data["label"]:
                train_data.append((text, {'entities': [(int(t_lbl["start"]) - 1, int(t_lbl["end"]) - 1,
                                                        t_lbl["labels"][0])]}))

        for _, annotations in train_data:
            for ent in annotations.get('entities'):
                self.ner.add_label(ent[2])

        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != 'ner']
        with self.nlp.disable_pipes(*other_pipes):  # only train NER
            optimizer = self.nlp.begin_training()
            for itn in range(self.iter):
                random.shuffle(train_data)
                losses = {}
                for text, annotations in tqdm(train_data):
                    self.nlp.update([text], [annotations], drop=0.5, sgd=optimizer, losses=losses)
                print(f"[INFO] Iteration: {itn}, losses: {losses}")

        self.nlp.to_disk(MODEL_DIR)
        print(f"[INFO] Saved model to {MODEL_DIR}")

        return


if __name__ == '__main__':
    NERTrainer().train()
