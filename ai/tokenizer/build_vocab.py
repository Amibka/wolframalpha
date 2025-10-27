"""
Построение словаря токенов из датасета
"""
import json
import sys
from pathlib import Path

# Добавляем корень проекта в путь
sys.path.append(str(Path(__file__).parent.parent))

from tokenizer.math_tokenizer import MathTokenizer
from config import TRAINING_DATA_PATH, VOCAB_PATH


def build_vocabulary():
    """Строит словарь из training_data.json"""

    # Загружаем датасет
    print(f"Загрузка датасета: {TRAINING_DATA_PATH}")
    with open(TRAINING_DATA_PATH, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # Собираем все тексты (входы и выходы)
    all_texts = []
    for sample in dataset:
        all_texts.append(sample["input"])
        all_texts.append(sample["output"])

    print(f"Всего текстов: {len(all_texts)}")

    # Строим словарь
    tokenizer = MathTokenizer()
    tokenizer.build_vocab(all_texts, min_freq=2)

    # Сохраняем
    tokenizer.save_vocab(VOCAB_PATH)
    print(f"Словарь сохранён: {VOCAB_PATH}")
    print(f"Размер словаря: {tokenizer.vocab_size} токенов")


if __name__ == "__main__":
    build_vocabulary()