import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModel

class ConvModel(nn.Module):
    def __init__(self, model_name, model_config, tokenizer_name):
        super().__init__() 
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.backbone_model = AutoModel.from_pretrained(model_name, config=model_config)
        self.conv1d_layer1 = nn.Conv1d(model_config.hidden_size, 1024, kernel_size=1)
        self.conv1d_layer3 = nn.Conv1d(model_config.hidden_size, 1024, kernel_size=3, padding=1)
        self.conv1d_layer5 = nn.Conv1d(model_config.hidden_size, 1024, kernel_size=5, padding=2)
        self.dropout = nn.Dropout(0.3)
        self.dense_layer = nn.Linear(1024 * 3, 2, bias=True)
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, start_positions=None, end_positions=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        if "xlm" in self.tokenizer_name:
            outputs = self.backbone_model(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        else:
            outputs = self.backbone_model(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        
        sequence_output = outputs[0] # Convolution 연산을 위해 Transpose (B * hidden_size * max_seq_legth)
        conv_input = sequence_output.transpose(1, 2) # Conv 연산을 위한 Transpose (B * hidden_size * max_seq_length)
        conv_output1 = F.relu(self.conv1d_layer1(conv_input)) # Conv연산의 결과 (B * num_conv_filter * max_seq_legth)
        conv_output3 = F.relu(self.conv1d_layer3(conv_input)) # Conv연산의 결과 (B * num_conv_filter * max_seq_legth)
        conv_output5 = F.relu(self.conv1d_layer5(conv_input)) # Conv연산의 결과 (B * num_conv_filter * max_seq_legth)
        concat_output = torch.cat((conv_output1, conv_output3, conv_output5), dim=1) # Concatenation (B * num_conv_filter x 3 * max_seq_legth)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return {"start_logits" : start_logits, "end_logits" : end_logits, "hidden_states" :  outputs.hidden_states, "attentions" : outputs.attentions} 