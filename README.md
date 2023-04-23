# Spelling Corrector to CODA standard
This is a Python script for converting Arabic text to CODA standard using pre-trained transformer models.

## Introduction
This is a spelling corrector to the CODA standard. It uses the [CODA MADARA corpus](https://camel.abudhabi.nyu.edu/madar-coda-corpus/) 
to train a spelling corrector. The corrector is based on transformer architecture. by fine-tuning [AraBART](https://arxiv.org/abs/2203.10945) and
[AraT5](https://arxiv.org/abs/2109.12068) models.

## Requirements

This script requires the following Python packages:

- transformers
- torch
- numpy
- pandas
- sentencepiece
- wandb

To install these packages, run the following command:

```
pip install -r requirements.txt
```


## Usage

To run the script, open a terminal and navigate to the directory containing the script. Then, run the following command:

```
python coda_conversion.py [OPTIONS]
```

The following options are available:

- `--wandb`: If this flag is set, the script logs the training process to [WandB](https://www.wandb.com/).
- `--test`: If this flag is set, the script runs the test function instead of the train function.
- `--model_name`: The name of the pre-trained transformer model to use. Default is "moussaKam/AraBART".
- `--hidden_size`: The number of hidden units in the transformer model. Default is 300.
- `--num_layers`: The number of layers in the transformer model. Default is 2.
- `--learning_rate`: The learning rate to use for training the model. Default is 0.00003.
- `--num_epochs`: The number of epochs to train the model for. Default is 10.
- `--optimizer`: The optimizer to use for training the model. Default is "adam".
- `--batch_size`: The batch size to use for training the model. Default is 8.
- `--sentence`: The Arabic sentence to convert to the target dialect. If this option is not set, the script enters interactive mode and prompts the user to enter a sentence.
- `--path`: The path to the directory containing the CODA corpus. Default is "coda-corpus".

For example, to run the script with the default hyperparameters and log the training process to WandB, run the following command:

```
python coda_conversion.py --wandb
```

To train the model for 20 epochs, run the following command:

```
python coda_conversion.py --num_epochs 20
```

To run the script in test mode, run the following command:

```
python coda_conversion.py --test
```

To specify a different model, run the following command:

```
python coda_conversion.py --model_name UBC-NLP/AraT5-base
```

To convert a specific sentence, run the following command: 

```
python coda_conversion.py --sentence "الجملة العربية التي تريد تحويلها او تصحيحها"
```

