import json
import os


class TranslationManager:
    """Менеджер переводов приложения"""

    def __init__(self):
        self.current_language = "ru"
        self.translations = {}
        self.load_translations()

    def load_translations(self):
        """Загружает все доступные переводы"""
        locales_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "locales")

        # Маппинг языков на имена файлов
        lang_files = {
            "ru": "ru.json",
            "eng": "eng.json"
        }

        for lang_code, filename in lang_files.items():
            file_path = os.path.join(locales_dir, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    self.translations[lang_code] = json.load(f)
                print(f"Loaded translation: {lang_code} from {filename}")
            except FileNotFoundError:
                print(f"Warning: Translation file not found: {file_path}")
                self.translations[lang_code] = {}
            except json.JSONDecodeError as e:
                print(f"Error parsing {file_path}: {e}")
                self.translations[lang_code] = {}

    def set_language(self, language_code):
        """Устанавливает текущий язык"""
        if language_code in self.translations:
            self.current_language = language_code
            print(f"Language set to: {language_code}")
        else:
            print(f"Warning: Language '{language_code}' not available")

    def get(self, key_path, default=""):
        """
        Получает перевод по пути ключа

        Пример: get("app.title") -> "🔬 WolframAlpha Calculator"
        """
        keys = key_path.split(".")
        value = self.translations.get(self.current_language, {})

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default

        return value if value is not None else default

    def get_all(self, section):
        """
        Получает все переводы из секции

        Пример: get_all("buttons") -> {"solve": "Решить", "cancel": "Отмена", ...}
        """
        keys = section.split(".")
        value = self.translations.get(self.current_language, {})

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key, {})
            else:
                return {}

        return value if isinstance(value, dict) else {}


# Глобальный экземпляр менеджера переводов
translator = TranslationManager()


def t(key_path, default=""):
    """Сокращённая функция для получения перевода"""
    return translator.get(key_path, default)