#!/usr/bin/env python
# coding: utf-8


from torch import nn
from transformers import AutoModel, AutoConfig

class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class Encoder(nn.Module):
    def __init__(self, model_checkpoint):
        super(Encoder, self).__init__()
        self.model_checkpoint = model_checkpoint
        config = AutoConfig.from_pretrained(self.model_checkpoint)
        
        if self.model_checkpoint == 'monologg/koelectra-base-v3-discriminator':
            self.pooler = BertPooler(config)
        config = AutoConfig.from_pretrained(self.model_checkpoint)
        self.model = AutoModel.from_pretrained(self.model_checkpoint, config=config)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None):
        outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids)
        if self.model_checkpoint == 'monologg/koelectra-base-v3-discriminator':
            sequence_output = outputs[0]
            pooled_output = self.pooler(sequence_output)
        else:
            pooled_output = outputs[1]
        return pooled_output