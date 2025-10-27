"""
Кастомный токенизатор для математических выражений
"""
import re
import json
from typing import List, Dict
from collections import Counter


class MathTokenizer:
    def __init__(self, vocab_path=None):
        self.special_tokens = {
            "[PAD]": 0,
            "[SOS]": 1,
            "[EOS]": 2,
            "[UNK]": 3,
        }

        if vocab_path and vocab_path.exists():
            self.load_vocab(vocab_path)
        else:
            self.token2id = self.special_tokens.copy()
            self.id2token = {v: k for k, v in self.token2id.items()}

    def tokenize(self, text: str) -> List[str]:
        """
        Токенизирует текст с учётом математических символов

        Примеры:
        "реши уравнение x^2 + 1 = 0" → ["реши", "уравнение", "x", "^", "2", "+", "1", "=", "0"]
        "solve(x**2 + 1, x)" → ["solve", "(", "x", "**", "2", "+", "1", ",", "x", ")"]
        """
        # Нормализация
        text = text.replace("^", "**")  # ^ → ** для степени
        text = text.replace("х", "x")  # русская х → английская x
        text = text.replace("×", "*")  # × → *
        text = text.replace("÷", "/")  # ÷ → /

        # Регулярка для токенизации с сохранением математических символов
        pattern = r'(\*\*|[a-zA-Zа-яА-Я_][a-zA-Z0-9а-яА-Я_]*|\d+\.?\d*|[+\-*/()=,\[\]{}]|\.\.\.)'
        tokens = re.findall(pattern, text)

        # Убираем пустые токены
        tokens = [t.strip() for t in tokens if t.strip()]

        return tokens

    def encode(self, text: str, max_length=None) -> List[int]:
        """Кодирует текст в список ID"""
        tokens = self.tokenize(text)
        ids = [self.token2id.get(t, self.token2id["[UNK]"]) for t in tokens]

        if max_length:
            ids = ids[:max_length]
            ids = ids + [self.token2id["[PAD]"]] * (max_length - len(ids))

        return ids

    def decode(self, ids: List[int]) -> str:
        """Декодирует список ID в текст"""
        tokens = []
        for id in ids:
            if id == self.token2id["[PAD]"]:
                break
            if id == self.token2id["[EOS]"]:
                break
            token = self.id2token.get(id, "[UNK]")
            if token not in ["[SOS]", "[PAD]", "[EOS]"]:
                tokens.append(token)

        # Склеиваем токены
        text = " ".join(tokens)

        # Убираем пробелы вокруг знаков операций
        text = re.sub(r'\s*([+\-*/()=,\[\]{}])\s*', r'\1', text)

        return text

    def build_vocab(self, texts: List[str], min_freq=2):
        """Строит словарь из текстов"""
        counter = Counter()

        for text in texts:
            tokens = self.tokenize(text)
            counter.update(tokens)

        # Добавляем только частые токены
        self.token2id = self.special_tokens.copy()

        for token, freq in counter.most_common():
            if freq >= min_freq and token not in self.token2id:
                self.token2id[token] = len(self.token2id)

        self.id2token = {v: k for k, v in self.token2id.items()}

        print(f"✅ Словарь построен: {len(self.token2id)} токенов")

    def save_vocab(self, path):
        """Сохраняет словарь"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                "token2id": self.token2id,
                "id2token": {str(k): v for k, v in self.id2token.items()}
            }, f, ensure_ascii=False, indent=2)

    def load_vocab(self, path):
        """Загружает словарь"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.token2id = data["token2id"]
            self.id2token = {int(k): v for k, v in data["id2token"].items()}

    @property
    def vocab_size(self):
        return len(self.token2id)

    def __len__(self):
        return self.vocab_size