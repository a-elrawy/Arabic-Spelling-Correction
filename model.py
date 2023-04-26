import os

import torch
import transformers
from torch import nn
import torch.optim as optim
# from peft import get_peft_model, LoraConfig, TaskType

from torch.nn.utils.rnn import pad_sequence


def create_optimizer(optimizer_name, learning_rate, parameters):
    if optimizer_name == "adam":
        optimizer = optim.AdamW(parameters, lr=learning_rate)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(parameters, lr=learning_rate)
    elif optimizer_name == "rmsprop":
        optimizer = optim.RMSprop(parameters, lr=learning_rate)
    else:
        raise ValueError("Invalid optimizer name")
    return optimizer


def load_model(model_name, num_layers=None, hidden_size=None, model_path=None,
               apply_lora=False, encoder_only=False):
    """
    Load a pre-trained model and tokenizer.

    Args:
        model_name (str): The name of the pre-trained model to load.
        num_layers (int): The number of layers to use for the model.
        hidden_size (int): The hidden size to use for the model.
        model_path (str): The path to the saved model, if loading from a saved model.
        apply_lora (bool): Whether to apply LoRA to the model.
    Returns:
        The loaded model and tokenizer.
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    default_model = transformers.AutoModel if encoder_only else transformers.AutoModelForSeq2SeqLM

    if num_layers is not None or hidden_size is not None:
        config = default_model.from_pretrained(model_name)
        if num_layers is not None:
            config.num_hidden_layers = num_layers
        if hidden_size is not None:
            config.hidden_size = hidden_size
        model = default_model.from_config(config=config)
    else:
        model = default_model.from_pretrained(model_name)

    # if apply_lora:
    #     peft_config = LoraConfig(
    #         task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
    #     )
    #     model = get_peft_model(model, peft_config)
    #     model.print_trainable_parameters()

    if model_path is not None and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))

    return model, tokenizer


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, input):
        input = input.long()
        embedded = self.embedding(input)
        output, (hidden, cell) = self.rnn(embedded)
        # concatenate the forward and backward hidden states
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        # pass the concatenated hidden state through a linear layer
        hidden = self.fc(hidden)
        return output, hidden, cell


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # calculate attention scores
        attn_weights = self.v(torch.tanh(
            self.attn(torch.cat((hidden.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1), encoder_outputs), dim=2))))
        # apply softmax to get attention weights
        attn_weights = F.softmax(attn_weights, dim=1)
        # apply attention weights to encoder outputs
        context = torch.bmm(attn_weights.transpose(1, 2), encoder_outputs)
        # concatenate context vector with hidden state
        output = torch.cat((context.squeeze(1), hidden), dim=1)
        return output, attn_weights


class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size * 2, hidden_size, num_layers)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, hidden, cell, encoder_outputs):
        # calculate attention-weighted context vector
        print(hidden.shape, encoder_outputs.shape)
        hidden, encoder_outputs = pad_sequence([hidden, encoder_outputs], batch_first=True, padding_value=0)
        context, attn_weights = self.attention(hidden, encoder_outputs)
        # embed input word
        embedded = self.embedding(input.unsqueeze(0))
        # concatenate embedded input word with context vector
        rnn_input = torch.cat((embedded, context.unsqueeze(0)), dim=2)
        # pass through RNN layer
        output, (hidden, cell) = self.rnn(rnn_input, (hidden.unsqueeze(0), cell.unsqueeze(0)))
        # concatenate output with context vector
        output = torch.cat((output.squeeze(0), context), dim=1)
        # pass through linear layer to get predicted output word
        prediction = self.fc(output)
        return prediction, hidden, cell, attn_weights.squeeze(2)
