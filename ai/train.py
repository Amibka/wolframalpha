"""
Скрипт обучения Seq2Seq модели
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent))

from config import *
from model.seq2seq import create_model
from tokenizer.math_tokenizer import MathTokenizer


class MathDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Кодируем вход и выход
        src = self.tokenizer.encode(sample["input"], max_length=self.max_len)
        tgt = self.tokenizer.encode(sample["output"], max_length=self.max_len)

        # Добавляем [SOS] и [EOS] к target
        tgt_input = [SPECIAL_TOKENS["SOS"]] + tgt[:-1]
        tgt_output = tgt + [SPECIAL_TOKENS["EOS"]]

        return {
            "src": torch.LongTensor(src),
            "tgt_input": torch.LongTensor(tgt_input),
            "tgt_output": torch.LongTensor(tgt_output)
        }


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        src = batch["src"].to(device)
        tgt_input = batch["tgt_input"].to(device)
        tgt_output = batch["tgt_output"].to(device)

        optimizer.zero_grad()

        # Forward pass
        output = model(src, tgt_input)

        # Reshape для loss
        output = output.reshape(-1, output.shape[-1])
        tgt_output = tgt_output.reshape(-1)

        # Loss
        loss = criterion(output, tgt_output)

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            src = batch["src"].to(device)
            tgt_input = batch["tgt_input"].to(device)
            tgt_output = batch["tgt_output"].to(device)

            output = model(src, tgt_input)
            output = output.reshape(-1, output.shape[-1])
            tgt_output = tgt_output.reshape(-1)

            loss = criterion(output, tgt_output)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    print("🚀 Начинаем обучение модели Math NLP → SymPy\n")

    # Устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Устройство: {device}\n")

    # Загружаем данные
    print(f"📂 Загрузка датасета: {TRAINING_DATA_PATH}")
    with open(TRAINING_DATA_PATH, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # Разделяем на train/val
    split_idx = int(len(dataset) * TRAINING_CONFIG["train_split"])
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]

    print(f"📊 Train: {len(train_data)}, Val: {len(val_data)}\n")

    # Загружаем токенизатор
    print(f"🔤 Загрузка токенизатора: {VOCAB_PATH}")
    tokenizer = MathTokenizer(VOCAB_PATH)
    print(f"📏 Размер словаря: {tokenizer.vocab_size}\n")

    # Датасеты и dataloaders
    train_dataset = MathDataset(train_data, tokenizer, MODEL_CONFIG["max_seq_length"])
    val_dataset = MathDataset(val_data, tokenizer, MODEL_CONFIG["max_seq_length"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=False,
        num_workers=4
    )

    # Создаём модель
    print("🧠 Создание модели...")
    model = create_model(MODEL_CONFIG, tokenizer.vocab_size, tokenizer.vocab_size)
    model = model.to(device)

    # Оптимизатор и loss
    optimizer = optim.Adam(model.parameters(), lr=TRAINING_CONFIG["learning_rate"])
    criterion = nn.CrossEntropyLoss(ignore_index=SPECIAL_TOKENS["PAD"])

    print(f"✅ Модель создана: {sum(p.numel() for p in model.parameters())} параметров\n")

    # Обучение
    best_val_loss = float('inf')

    for epoch in range(TRAINING_CONFIG["num_epochs"]):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch + 1}/{TRAINING_CONFIG['num_epochs']}")
        print(f"{'=' * 50}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(f"\n📊 Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Сохранение checkpoint
        if (epoch + 1) % TRAINING_CONFIG["save_every"] == 0 or val_loss < best_val_loss:
            checkpoint_path = CHECKPOINTS_DIR / f"model_epoch_{epoch + 1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"💾 Checkpoint сохранён: {checkpoint_path}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = CHECKPOINTS_DIR / "best_model.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, best_path)
                print(f"⭐ Лучшая модель сохранена: {best_path}")

    print("\n🎉 Обучение завершено!")


if __name__ == "__main__":
    main()