"""
Менеджер тем оформления
"""

THEMES = {
    "dark": """
        * {
            transition: all 0.3s ease;
        }
        QWidget {
            background-color: #1e1e2e;
            color: #cdd6f4;
        }
        QFrame#header {
            background-color: #181825;
            border-bottom: 1px solid #313244;
        }
        QFrame#card {
            background-color: #181825;
            border-radius: 12px;
            border: 1px solid #313244;
        }
        QLineEdit {
            background-color: #313244;
            color: #cdd6f4;
            border: 2px solid #45475a;
            border-radius: 8px;
            padding: 10px 15px;
            font-size: 15px;
        }
        QLineEdit:focus {
            border-color: #89b4fa;
            background-color: #45475a;
        }
        QPushButton {
            background-color: #89b4fa;
            color: #1e1e2e;
            border: none;
            border-radius: 8px;
            font-weight: bold;
            font-size: 14px;
            padding: 5px 15px;
        }
        QPushButton:hover {
            background-color: #b4befe;
        }
        QPushButton:pressed {
            background-color: #74c7ec;
        }
        IconButton {
            background-color: #313244;
            color: #cdd6f4;
            border: 1px solid #45475a;
            border-radius: 22px;
            font-weight: bold;
            font-size: 14px;
        }
        IconButton:hover {
            background-color: #45475a;
            border-color: #89b4fa;
        }
        QTextEdit {
            background-color: #313244;
            color: #cdd6f4;
            border: 1px solid #45475a;
            border-radius: 8px;
            padding: 15px;
            font-size: 13px;
        }
        QScrollBar:vertical {
            background-color: #181825;
            width: 10px;
            border-radius: 5px;
        }
        QScrollBar::handle:vertical {
            background-color: #45475a;
            border-radius: 5px;
            min-height: 30px;
        }
        QScrollBar::handle:vertical:hover {
            background-color: #585b70;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
    """,

    "light": """
        * {
            transition: all 0.3s ease;
        }
        QWidget {
            background-color: #eff1f5;
            color: #4c4f69;
        }
        QFrame#header {
            background-color: #e6e9ef;
            border-bottom: 1px solid #dce0e8;
        }
        QFrame#card {
            background-color: #e6e9ef;
            border-radius: 12px;
            border: 1px solid #dce0e8;
        }
        QLineEdit {
            background-color: #dce0e8;
            color: #4c4f69;
            border: 2px solid #ccd0da;
            border-radius: 8px;
            padding: 10px 15px;
            font-size: 15px;
        }
        QLineEdit:focus {
            border-color: #1e66f5;
            background-color: #ccd0da;
        }
        QPushButton {
            background-color: #1e66f5;
            color: #ffffff;
            border: none;
            border-radius: 8px;
            font-weight: bold;
            font-size: 14px;
            padding: 5px 15px;
        }
        QPushButton:hover {
            background-color: #04a5e5;
        }
        QPushButton:pressed {
            background-color: #209fb5;
        }
        IconButton {
            background-color: #dce0e8;
            color: #4c4f69;
            border: 1px solid #ccd0da;
            border-radius: 22px;
            font-weight: bold;
            font-size: 14px;
        }
        IconButton:hover {
            background-color: #ccd0da;
            border-color: #1e66f5;
        }
        QTextEdit {
            background-color: #dce0e8;
            color: #4c4f69;
            border: 1px solid #ccd0da;
            border-radius: 8px;
            padding: 15px;
            font-size: 13px;
        }
        QScrollBar:vertical {
            background-color: #e6e9ef;
            width: 10px;
            border-radius: 5px;
        }
        QScrollBar::handle:vertical {
            background-color: #ccd0da;
            border-radius: 5px;
            min-height: 30px;
        }
        QScrollBar::handle:vertical:hover {
            background-color: #bcc0cc;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
    """,

    "ocean": """
        * {
            transition: all 0.3s ease;
        }
        QWidget {
            background-color: #1b2838;
            color: #c5d1de;
        }
        QFrame#header {
            background-color: #16202d;
            border-bottom: 1px solid #243447;
        }
        QFrame#card {
            background-color: #16202d;
            border-radius: 12px;
            border: 1px solid #243447;
        }
        QLineEdit {
            background-color: #243447;
            color: #c5d1de;
            border: 2px solid #2d4356;
            border-radius: 8px;
            padding: 10px 15px;
            font-size: 15px;
        }
        QLineEdit:focus {
            border-color: #66c0f4;
            background-color: #2d4356;
        }
        QPushButton {
            background-color: #5ba3d0;
            color: #ffffff;
            border: none;
            border-radius: 8px;
            font-weight: bold;
            font-size: 14px;
            padding: 5px 15px;
        }
        QPushButton:hover {
            background-color: #66c0f4;
        }
        QPushButton:pressed {
            background-color: #4a90b8;
        }
        IconButton {
            background-color: #243447;
            color: #c5d1de;
            border: 1px solid #2d4356;
            border-radius: 22px;
            font-weight: bold;
            font-size: 14px;
        }
        IconButton:hover {
            background-color: #2d4356;
            border-color: #66c0f4;
        }
        QTextEdit {
            background-color: #243447;
            color: #c5d1de;
            border: 1px solid #2d4356;
            border-radius: 8px;
            padding: 15px;
            font-size: 13px;
        }
        QScrollBar:vertical {
            background-color: #16202d;
            width: 10px;
            border-radius: 5px;
        }
        QScrollBar::handle:vertical {
            background-color: #2d4356;
            border-radius: 5px;
            min-height: 30px;
        }
        QScrollBar::handle:vertical:hover {
            background-color: #3a5466;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
    """
}

# Метаданные тем для диалога настроек
THEME_INFO = {
    "dark": {
        "name": "🌙 Темная тема",
        "name_en": "🌙 Dark Theme",
        "description": "Мягкая темная тема для комфортной работы",
        "description_en": "Soft dark theme for comfortable work",
        "colors": {
            "header": "#181825",
            "body": "#1e1e2e",
            "elements": "#313244",
            "hover": "#89b4fa"
        }
    },
    "light": {
        "name": "☀️ Светлая тема",
        "name_en": "☀️ Light Theme",
        "description": "Чистая светлая тема для дневной работы",
        "description_en": "Clean light theme for daytime work",
        "colors": {
            "header": "#e6e9ef",
            "body": "#eff1f5",
            "elements": "#dce0e8",
            "hover": "#1e66f5"
        }
    },
    "ocean": {
        "name": "🌊 Океан",
        "name_en": "🌊 Ocean",
        "description": "Спокойная морская тема с голубыми акцентами",
        "description_en": "Calm ocean theme with blue accents",
        "colors": {
            "header": "#16202d",
            "body": "#1b2838",
            "elements": "#243447",
            "hover": "#66c0f4"
        }
    }
}


def get_theme(theme_name):
    """Возвращает CSS для указанной темы"""
    return THEMES.get(theme_name, THEMES["dark"])


def get_theme_info(theme_name, language="ru"):
    """Возвращает информацию о теме на указанном языке"""
    info = THEME_INFO.get(theme_name, THEME_INFO["dark"])
    result = {
        "name": info.get(f"name_{language}" if language == "en" else "name"),
        "description": info.get(f"description_{language}" if language == "en" else "description"),
        "colors": info["colors"]
    }
    return result


def get_all_themes():
    """Возвращает список всех доступных тем"""
    return list(THEMES.keys())