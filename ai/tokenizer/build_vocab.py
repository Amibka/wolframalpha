"""
Построение словаря токенов из датасета
"""
import json
import sys
from pathlib import Path

# Добавляем корень проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Импорты из локальных файлов
from tokenizer.math_tokenizer import MathTokenizer
from config import TRAINING_DATA_PATH, VOCAB_PATH

def build_vocabulary():
    """Строит словарь из training_data.json"""
    
    # Проверяем существование датасета
    if not TRAINING_DATA_PATH.exists():
        print(f"❌ Датасет не найден: {TRAINING_DATA_PATH}")
        print("Сначала запустите: python utils/dataset_generator.py")
        return
    
    # Загружаем датасет
    print(f"📂 Загрузка датасета: {TRAINING_DATA_PATH}")
    with open(TRAINING_DATA_PATH, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Собираем все тексты (входы и выходы)
    all_texts = []
    for sample in dataset:
        all_texts.append(sample["input"])
        all_texts.append(sample["output"])
    
    print(f"📊 Всего текстов: {len(all_texts)}")
    
    # Строим словарь
    tokenizer = MathTokenizer()
    tokenizer.build_vocab(all_texts, min_freq=2)
    
    # Создаём директорию если не существует
    VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем
    tokenizer.save_vocab(VOCAB_PATH)
    print(f"💾 Словарь сохранён: {VOCAB_PATH}")
    print(f"📏 Размер словаря: {tokenizer.vocab_size} токенов")

if __name__ == "__main__":
    build_vocabulary()