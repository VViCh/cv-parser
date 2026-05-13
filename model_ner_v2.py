import torch
import torch.nn as nn
from transformers import AutoModel
from torchcrf import CRF


class ResumeNERModel(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", num_labels=9):
        super(ResumeNERModel, self).__init__()

        self.transformer = AutoModel.from_pretrained(model_name)

        if hasattr(self.transformer, "encoder") and hasattr(self.transformer.encoder, "layer"):
            layers = self.transformer.encoder.layer
        elif hasattr(self.transformer, "transformer") and hasattr(self.transformer.transformer, "layer"):
            layers = self.transformer.transformer.layer
        else:
            layers = []

        num_layers_to_freeze = len(layers) // 2
        for layer in layers[:num_layers_to_freeze]:
            for param in layer.parameters():
                param.requires_grad = False

        embedding_dim = self.transformer.config.hidden_size

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(embedding_dim, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        # Force float32 for CRF stability (prevents dtype mismatch with DeBERTa)
        sequence_output = sequence_output.float()

        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output).float()

        if labels is not None:
            valid_mask = attention_mask.bool() & (labels != -100)
            valid_mask[:, 0] = True

            safe_labels = labels.masked_fill(labels.lt(0), 0)

            with torch.amp.autocast('cuda', enabled=False):
                loss = -self.crf(emissions, safe_labels, mask=valid_mask, reduction='mean')
            return loss
        else:
            with torch.amp.autocast('cuda', enabled=False):
                return self.crf.decode(emissions, mask=attention_mask.bool())
