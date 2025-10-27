"""
Inference: преобразование текста в SymPy код
"""
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from config import *
from model.seq2seq import create_model
from tokenizer.math_tokenizer import MathTokenizer
from preprocessing.text_normalizer import TextNormalizer


class MathTranslator:
    def __init__(self, model_path, vocab_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Загружаем токенизатор
        self.tokenizer = MathTokenizer(vocab_path)
        self.normalizer = TextNormalizer()

        # Создаём и загружаем модель
        self.model = create_model(MODEL_CONFIG, self.tokenizer.vocab_size, self.tokenizer.vocab_size)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"✅ Модель загружена: {model_path}")
        print(f"📏 Размер словаря: {self.tokenizer.vocab_size}")

    def translate(self, text, max_length=128, beam_size=5):
        """
        Переводит текст в SymPy код с использованием beam search

        Args:
            text: входной текст (RU/EN)
            max_length: максимальная длина выхода
            beam_size: размер beam для beam search

        Returns:
            sympy_code: сгенерированный SymPy код
        """
        # Нормализация
        text = self.normalizer.normalize(text)

        # Кодируем вход
        src = torch.LongTensor(self.tokenizer.encode(text, max_length=max_length)).unsqueeze(0)
        src = src.to(self.device)

        with torch.no_grad():
            # Кодируем source
            encoder_output = self.model.encode(src)
            src_mask = self.model.make_src_mask(src)

            # Инициализация для beam search
            beams = [(torch.LongTensor([[SPECIAL_TOKENS["SOS"]]]).to(self.device), 0.0)]

            for _ in range(max_length):
                new_beams = []

                for beam, score in beams:
                    if beam[0, -1].item() == SPECIAL_TOKENS["EOS"]:
                        new_beams.append((beam, score))
                        continue

                    # Декодируем один шаг
                    output = self.model.decode(beam, encoder_output, src_mask)
                    logits = output[:, -1, :]
                    log_probs = torch.log_softmax(logits, dim=-1)

                    # Берём top-k
                    top_log_probs, top_indices = log_probs.topk(beam_size)

                    for i in range(beam_size):
                        token = top_indices[0, i].unsqueeze(0).unsqueeze(0)
                        new_beam = torch.cat([beam, token], dim=1)
                        new_score = score + top_log_probs[0, i].item()
                        new_beams.append((new_beam, new_score))

                # Оставляем top beam_size
                beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

                # Если все beams закончились, выходим
                if all(beam[0, -1].item() == SPECIAL_TOKENS["EOS"] for beam, _ in beams):
                    break

            # Берём лучший beam
            best_beam, _ = beams[0]
            output_ids = best_beam[0].tolist()

            # Декодируем
            sympy_code = self.tokenizer.decode(output_ids)

        return sympy_code

    def batch_translate(self, texts, max_length=128):
        """Переводит batch текстов"""
        return [self.translate(text, max_length) for text in texts]


def main():
    """Демонстрация работы"""
    model_path = CHECKPOINTS_DIR / "best_model.pt"

    if not model_path.exists():
        print(f"❌ Модель не найдена: {model_path}")
        print("Сначала обучите модель: python train.py")
        return

    translator = MathTranslator(model_path, VOCAB_PATH)

    # Примеры
    test_cases = [
        "реши уравнение 4x + 1 = 10",
        "производная sin(x^2)",
        "интеграл от 0 до 5 x + 1",
        "упрости (x^2 + 2x + 1)/(x + 1)",
        "разложи на множители x^2 - 4",
        "предел sin(x)/x при x -> 0",
        "solve equation 2x - 5 = 0",
        "derivative of cos(x^2)",
        "simplify (x + 1)^2",
    ]

    print("\n" + "=" * 60)
    print("🧪 ТЕСТИРОВАНИЕ МОДЕЛИ")
    print("=" * 60 + "\n")

    for i, text in enumerate(test_cases, 1):
        sympy_code = translator.translate(text)
        print(f"{i}. Вход: {text}")
        print(f"   Выход: {sympy_code}\n")


if __name__ == "__main__":
    main()