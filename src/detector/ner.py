import spacy

from spacy.lang.en import English  # updated
from settings import MODEL_DIR


class NERDetector:
    def __init__(self):
        self.model = spacy.load(MODEL_DIR)
        self.nlp = English()
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))  # updated

    def run(self, txt_file_path):
        result = []
        with open(txt_file_path, 'r') as f:
            txt_content = f.read()
        doc = self.nlp(txt_content)
        sentences = [sent.string.strip() for sent in doc.sents]
        for sent in sentences:
            res = self.model(sent)
            if res.ents:
                for ent in res.ents:
                    beams = self.model.entity.beam_parse([res], beam_width=16, beam_density=0.0001)
                    for beam in beams:
                        for score, ents in self.model.entity.moves.get_beam_parses(beam):
                            result.extend(
                                {ent.label_: ent.text, "confidence": score}
                                for start, end, label in ents
                                if start == ent.start
                            )
                            break

        return result


if __name__ == '__main__':
    NERDetector().run(txt_file_path="")
