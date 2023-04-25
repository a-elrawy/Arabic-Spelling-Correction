import torch
from data_loader import get_loaders
from model import load_model, create_optimizer
from train import Trainer
import wandb
import argparse


def main(config):
    # Load the model and tokenizer
    model_name=config['model_name']
    model_path = model_name.split('/')[-1]
    model, tokenizer = load_model(model_name, num_layers=config['num_layers'],
                                  hidden_size=config['hidden_size'], model_path=f"{model_path}.pt")

    # Set up the optimizer and loss function
    optimizer = create_optimizer(config['optimizer'], config['learning_rate'], model.parameters())
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    if config['wandb']:
        wandb.init(project='CODA-conversion')
        wandb.config.update(config)
    # Load the data loaders
    train_loader, val_loader, test_loader = get_loaders(path=PATH, batch_size=config['batch_size'], shuffle=True)
    trainer = Trainer(model, optimizer, criterion, train_loader, tokenizer)

    # Train model
    if config['test']:
        if config['sentence']:
            print(trainer.corrected(config['sentence']))
        else:
            trainer.evaluate(test_loader)

    else:
        trainer.train(val_loader, num_epochs=config['num_epochs'], wandb_log=config['wandb'])
        torch.save(model.state_dict(), f"{model_path}.pt")


def test_config():
    wandb.login()

    def train_and_log():
        # Initialize WandB
        wandb.init(project='CODA-conversion')
        wconfig = wandb.config

        model, tokenizer = load_model(wconfig.model_name, num_layers=wconfig.num_layers, hidden_size=wconfig.hidden_size)
        optimizer = create_optimizer(wconfig.optimizer, wconfig.learning_rate, model.parameters())
        criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

        train_loader, val_loader, test_loader = get_loaders(path=PATH, batch_size=wconfig.batch_size, shuffle=True)

        trainer = Trainer(model, optimizer, criterion, train_loader, tokenizer)
        trainer.train(val_loader, num_epochs=wconfig.num_epochs, wandb_log=True)
        # torch.save(model.state_dict(), f"{wconfig.model_name}.pt")

    # Define a multilevel sweep configuration
    sweep_config = {
        "name": "multimodel-sweep",
        "metric": {"name": "val_loss", "goal": "minimize"},
        "method": "grid",
        "parameters": {
            "model_name": {"values": ["moussaKam/AraBART", "UBC-NLP/AraT5-base"]},
            "hidden_size": {"values": [300, 540, 768]},
            "num_layers": {"values": [2, 4, 6]},
            "learning_rate": {"values": [0.00003, 0.0001]},
            "num_epochs": {"values": [10]},
            "optimizer": {"values": ["adam"]},
            "batch_size": {"values": [8]},
        },
    }

    # Initialize a WandB sweep
    sweep_id = wandb.sweep(sweep_config)

    # Run the sweep with different configurations for each model
    wandb.agent(sweep_id, function=train_and_log)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--wandb", action="store_true")
    args.add_argument("--test", action="store_true")
    args.add_argument("--model_name", type=str, default="moussaKam/AraBART")
    args.add_argument("--hidden_size", type=int, default=300)
    args.add_argument("--num_layers", type=int, default=2)
    args.add_argument("--learning_rate", type=float, default=0.00003)
    args.add_argument("--num_epochs", type=int, default=10)
    args.add_argument("--optimizer", type=str, default="adam")
    args.add_argument("--batch_size", type=int, default=8)
    args.add_argument("--sentence", type=str, default=None)
    args.add_argument("--path", type=str, default="coda-corpus")
    args.add_argument("--test_architecture", action="store_true", help="test the model architecture (no training, just testing the")

    args = args.parse_args()
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(hash("a") % 2 ** 32 - 1)
    torch.cuda.manual_seed_all(hash("a") % 2 ** 32 - 1)
    PATH = args.path

    if args.test_architecture:
        test_config()
        exit()

    arguments = {
        "model_name": args.model_name,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "optimizer": args.optimizer,
        "batch_size": args.batch_size,
        "test": args.test,
        "sentence": args.sentence,
        "wandb": args.wandb
    }

    main(config=arguments)
