import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch


class Scorer(nn.Module):
    def __init__(self, tokenizer, model, max_length, device):
        super(Scorer, self).__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.max_length = max_length
        self.device = device

    def prepare_input(self, query, passage):
        inputs = self.tokenizer.encode_plus(text=query,
                                            text_pair=passage,
                                            max_length=self.max_length,
                                            pad_to_max_length=True,
                                            return_token_type_ids=True,
                                            return_attention_mask=True)
        return [inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids']]

    def prepare_inputs(self, queries, passages):
        inputs = []
        for query, passage in zip(queries, passages):
            inputs.append(self.prepare_input(query, passage))
        return inputs

    def forward(self, inputs, labels=None):
        inputs = torch.transpose(inputs, 0, 1)
        assert len(inputs[0]) == len(inputs[1]) and len(
            inputs[0]) == len(inputs[2])
        x = self.model(inputs[0], attention_mask=inputs[1],
                       token_type_ids=inputs[2], labels=labels)
        return x

    def predict(self, inputs, batch_size):
        with torch.no_grad():
            scores = []
            for i in range(int(np.ceil(len(inputs)/batch_size))):
                input = inputs[i*batch_size:(i+1)*batch_size]
                input = torch.as_tensor(input).to(self.device)
                output = self(input)
                score = F.softmax(output.logits, dim=1)
                scores.append(score.cpu())
        return torch.vstack(scores)

    def score_query_passage(self, query, passage):
        return self.score_from_prediction(self.predict([self.prepare_input(query, passage)]))

    def score_query_passages(self, query, passages, batch_size):
        queries = [query] * len(passages)
        inputs = self.prepare_inputs(queries, passages)
        return self.score_from_prediction(self.predict(inputs, batch_size=batch_size))

    @staticmethod
    def score_from_prediction(prediction):
        prediction = prediction[:, -1]
        np_prediction = np.asarray(prediction)
        return list(np_prediction)
