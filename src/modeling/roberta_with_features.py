from transformers import AutoModel, AutoConfig
from torch import nn
import torch

AUX_DIM = 8 

class RobertaWithFeatures(nn.Module):
    def __init__(self, base_name: str, num_labels: int, aux_dim: int = AUX_DIM):
        super().__init__()
        self.config = AutoConfig.from_pretrained(base_name, num_labels=num_labels)
        self.roberta = AutoModel.from_pretrained(base_name, config=self.config)
        hidden = self.config.hidden_size
        self.aux_dim = aux_dim

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(hidden + aux_dim, num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        aux_features=None,
        labels=None,
    ):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)

        if aux_features is None:
            aux_features = torch.zeros(
                (pooled.size(0), self.aux_dim),
                device=pooled.device,
                dtype=pooled.dtype,
            )
        else:
            aux_features = aux_features.to(pooled.dtype)

        fused = torch.cat([pooled, aux_features], dim=-1)
        logits = self.classifier(fused)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits}
