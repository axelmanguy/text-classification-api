import os
import yaml
import torch
import logging
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
from transformers import CamembertTokenizer, CamembertForSequenceClassification, AdamW, \
    AutoModelForSequenceClassification, AutoTokenizer
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

from src.training.prepare_data import data_preparation
from src.utils.data_loader import load_data
from src.utils.config_loader import load_config
from src.utils.logger import train_logger as logger
from src.utils.logger import train_log_file

# Load configuration
train_config = load_config("hyperparameters.yaml")


# Define dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


# Define Lightning module
class CamembertClassifier(pl.LightningModule):
    def __init__(self, num_labels, learning_rate):
        super(CamembertClassifier, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained("almanach/camembertav2-base",
                                                                        num_labels=num_labels)
        self.learning_rate = learning_rate

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        self.train()
        outputs = self(batch["input_ids"], batch["attention_mask"], batch["labels"])
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.eval()
        outputs = self(batch["input_ids"], batch["attention_mask"], batch["labels"])
        loss = outputs.loss
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.learning_rate)

# Save model after training
def save_model(model, tokenizer, save_dir="/home/mng/PycharmProjects/text-classification-api/output"):
    os.makedirs(save_dir, exist_ok=True)

    # Save Hugging Face model (can be reloaded with from_pretrained)
    model.model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # Save TorchScript model
    scripted_model = torch.jit.script(model.model)
    scripted_model.save(os.path.join(save_dir, "camembert_scripted.pt"))

    logger.info(f"Model saved to {save_dir}")

# Load dataset# Training pipeline
def train():
    logger.info("[CAMEMBERT TRAIN] Loading dataset...")
    train_texts, val_texts, train_labels, val_labels = data_preparation('stages-votes.json')

    logger.info("[CAMEMBERT TRAIN] Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("almanach/camembertav2-base")

    train_dataset = TextDataset(list(train_texts), np.array(train_labels), tokenizer, train_config['camembert']["max_length"])
    val_dataset = TextDataset(list(val_texts), np.array(val_labels), tokenizer, train_config['camembert']["max_length"])

    train_loader = DataLoader(train_dataset, batch_size=train_config['camembert']["batch_size"], shuffle=True, num_workers=10)
    val_loader = DataLoader(val_dataset, batch_size=train_config['camembert']["batch_size"], num_workers=10)

    logger.info("[CAMEMBERT TRAIN] Initializing model...")
    model = CamembertClassifier(num_labels=2, learning_rate=float(train_config['camembert']["learning_rate"]))

    # Model checkpointing every 1/4 epoch
    checkpoint_callback = ModelCheckpoint(
        dirpath="/home/mng/PycharmProjects/TXTCLS/output/ckpt",
        filename="camembert_epoch_{epoch:02d}_step_{step}",
        save_top_k=3,  # Save all checkpoints
        every_n_train_steps=len(train_loader) // 4,  # Save every 1/4 epoch
        save_weights_only=True,
        monitor="val_loss",
        mode="min"
    )

    tb_logger = TensorBoardLogger("/home/mng/PycharmProjects/TXTCLS/output", name="camembert_experiment")
    trainer = pl.Trainer(
        max_epochs=train_config['camembert']["epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=tb_logger,
        callbacks=[checkpoint_callback],
        val_check_interval=0.1
    )

    logger.info("[CAMEMBERT TRAIN]Starting training...")
    trainer.fit(model, train_loader, val_loader)

    # Save model for inference
    logger.info("Saving final model...")
    save_model(model, tokenizer)

    logger.info("[CAMEMBERT TRAIN]Training completed. Model saved to /src/models")


if __name__ == "__main__":
    train()
