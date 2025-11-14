"""
ui/themes.py - СОВРЕМЕННАЯ СИСТЕМА ТЕМ
Полностью новый дизайн с градиентами, тенями и анимациями
"""

THEMES = {
    "dark": """
        * {
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        QWidget {
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:1,
                stop:0 #0f0f23,
                stop:1 #1a1a35
            );
            color: #e0e7ff;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        
        QFrame#header {
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:0,
                stop:0 rgba(30, 30, 60, 0.95),
                stop:1 rgba(20, 20, 50, 0.95)
            );
            border: none;
            border-bottom: 2px solid qlineargradient(
                x1:0, y1:0, x2:1, y2:0,
                stop:0 #6366f1,
                stop:0.5 #8b5cf6,
                stop:1 #d946ef
            );
        }
        
        QFrame#card {
            background: qlineargradient(
                x1:0, y1:0, x2:0, y2:1,
                stop:0 rgba(30, 30, 60, 0.6),
                stop:1 rgba(20, 20, 50, 0.8)
            );
            border-radius: 16px;
            border: 1px solid rgba(99, 102, 241, 0.3);
        }
        
        QFrame#card:hover {
            border: 1px solid rgba(139, 92, 246, 0.5);
            background: qlineargradient(
                x1:0, y1:0, x2:0, y2:1,
                stop:0 rgba(35, 35, 70, 0.7),
                stop:1 rgba(25, 25, 60, 0.9)
            );
        }
        
        QLineEdit {
            background: rgba(30, 30, 60, 0.5);
            color: #e0e7ff;
            border: 2px solid rgba(99, 102, 241, 0.3);
            border-radius: 12px;
            padding: 12px 18px;
            font-size: 15px;
            font-weight: 500;
        }
        
        QLineEdit:focus {
            border: 2px solid qlineargradient(
                x1:0, y1:0, x2:1, y2:0,
                stop:0 #6366f1,
                stop:1 #8b5cf6
            );
            background: rgba(40, 40, 80, 0.7);
        }
        
        QPushButton {
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:1,
                stop:0 #6366f1,
                stop:1 #8b5cf6
            );
            color: #ffffff;
            border: none;
            border-radius: 12px;
            font-weight: 600;
            font-size: 14px;
            padding: 8px 20px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        QPushButton:hover {
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:1,
                stop:0 #7c3aed,
                stop:1 #a855f7
            );
            padding: 8px 22px;
        }
        
        QPushButton:pressed {
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:1,
                stop:0 #5b21b6,
                stop:1 #7e22ce
            );
            padding: 8px 18px;
        }
        
        IconButton {
            background: rgba(99, 102, 241, 0.15);
            color: #e0e7ff;
            border: 2px solid rgba(99, 102, 241, 0.3);
            border-radius: 25px;
            font-size: 18px;
        }
        
        IconButton:hover {
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:1,
                stop:0 rgba(99, 102, 241, 0.3),
                stop:1 rgba(139, 92, 246, 0.3)
            );
            border: 2px solid rgba(139, 92, 246, 0.6);
        }
        
        IconButton:pressed {
            background: rgba(99, 102, 241, 0.4);
        }
        
        QTextEdit {
            background: rgba(20, 20, 50, 0.5);
            color: #e0e7ff;
            border: 1px solid rgba(99, 102, 241, 0.2);
            border-radius: 12px;
            padding: 16px;
            font-size: 14px;
            line-height: 1.6;
        }
        
        QScrollBar:vertical {
            background: rgba(15, 15, 35, 0.3);
            width: 12px;
            border-radius: 6px;
            margin: 2px;
        }
        
        QScrollBar::handle:vertical {
            background: qlineargradient(
                x1:0, y1:0, x2:0, y2:1,
                stop:0 #6366f1,
                stop:1 #8b5cf6
            );
            border-radius: 6px;
            min-height: 40px;
        }
        
        QScrollBar::handle:vertical:hover {
            background: qlineargradient(
                x1:0, y1:0, x2:0, y2:1,
                stop:0 #7c3aed,
                stop:1 #a855f7
            );
        }
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
        
        QLabel {
            background: transparent;
        }
    """,

    "light": """
        * {
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        QWidget {
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:1,
                stop:0 #f8fafc,
                stop:1 #f1f5f9
            );
            color: #0f172a;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        
        QFrame#header {
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:0,
                stop:0 rgba(255, 255, 255, 0.95),
                stop:1 rgba(248, 250, 252, 0.95)
            );
            border: none;
            border-bottom: 2px solid qlineargradient(
                x1:0, y1:0, x2:1, y2:0,
                stop:0 #3b82f6,
                stop:0.5 #8b5cf6,
                stop:1 #ec4899
            );
        }
        
        QFrame#card {
            background: qlineargradient(
                x1:0, y1:0, x2:0, y2:1,
                stop:0 rgba(255, 255, 255, 0.9),
                stop:1 rgba(248, 250, 252, 0.9)
            );
            border-radius: 16px;
            border: 1px solid rgba(203, 213, 225, 0.6);
        }
        
        QFrame#card:hover {
            border: 1px solid rgba(147, 197, 253, 0.8);
            background: qlineargradient(
                x1:0, y1:0, x2:0, y2:1,
                stop:0 rgba(255, 255, 255, 1),
                stop:1 rgba(241, 245, 249, 1)
            );
        }
        
        QLineEdit {
            background: rgba(255, 255, 255, 0.8);
            color: #0f172a;
            border: 2px solid rgba(203, 213, 225, 0.6);
            border-radius: 12px;
            padding: 12px 18px;
            font-size: 15px;
            font-weight: 500;
        }
        
        QLineEdit:focus {
            border: 2px solid qlineargradient(
                x1:0, y1:0, x2:1, y2:0,
                stop:0 #3b82f6,
                stop:1 #8b5cf6
            );
            background: rgba(255, 255, 255, 1);
        }
        
        QPushButton {
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:1,
                stop:0 #3b82f6,
                stop:1 #8b5cf6
            );
            color: #ffffff;
            border: none;
            border-radius: 12px;
            font-weight: 600;
            font-size: 14px;
            padding: 8px 20px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        QPushButton:hover {
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:1,
                stop:0 #2563eb,
                stop:1 #7c3aed
            );
            padding: 8px 22px;
        }
        
        QPushButton:pressed {
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:1,
                stop:0 #1d4ed8,
                stop:1 #6d28d9
            );
            padding: 8px 18px;
        }
        
        IconButton {
            background: rgba(59, 130, 246, 0.1);
            color: #1e40af;
            border: 2px solid rgba(59, 130, 246, 0.3);
            border-radius: 25px;
            font-size: 18px;
        }
        
        IconButton:hover {
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:1,
                stop:0 rgba(59, 130, 246, 0.2),
                stop:1 rgba(139, 92, 246, 0.2)
            );
            border: 2px solid rgba(59, 130, 246, 0.6);
        }
        
        IconButton:pressed {
            background: rgba(59, 130, 246, 0.3);
        }
        
        QTextEdit {
            background: rgba(255, 255, 255, 0.9);
            color: #0f172a;
            border: 1px solid rgba(203, 213, 225, 0.5);
            border-radius: 12px;
            padding: 16px;
            font-size: 14px;
            line-height: 1.6;
        }
        
        QScrollBar:vertical {
            background: rgba(241, 245, 249, 0.5);
            width: 12px;
            border-radius: 6px;
            margin: 2px;
        }
        
        QScrollBar::handle:vertical {
            background: qlineargradient(
                x1:0, y1:0, x2:0, y2:1,
                stop:0 #3b82f6,
                stop:1 #8b5cf6
            );
            border-radius: 6px;
            min-height: 40px;
        }
        
        QScrollBar::handle:vertical:hover {
            background: qlineargradient(
                x1:0, y1:0, x2:0, y2:1,
                stop:0 #2563eb,
                stop:1 #7c3aed
            );
        }
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
        
        QLabel {
            background: transparent;
        }
    """,

    "ocean": """
        * {
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        QWidget {
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:1,
                stop:0 #0c1e2e,
                stop:1 #1a3a52
            );
            color: #cfe9ff;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        
        QFrame#header {
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:0,
                stop:0 rgba(15, 30, 50, 0.95),
                stop:1 rgba(20, 40, 65, 0.95)
            );
            border: none;
            border-bottom: 2px solid qlineargradient(
                x1:0, y1:0, x2:1, y2:0,
                stop:0 #06b6d4,
                stop:0.5 #3b82f6,
                stop:1 #8b5cf6
            );
        }
        
        QFrame#card {
            background: qlineargradient(
                x1:0, y1:0, x2:0, y2:1,
                stop:0 rgba(20, 40, 65, 0.6),
                stop:1 rgba(15, 30, 50, 0.8)
            );
            border-radius: 16px;
            border: 1px solid rgba(6, 182, 212, 0.3);
        }
        
        QFrame#card:hover {
            border: 1px solid rgba(6, 182, 212, 0.5);
            background: qlineargradient(
                x1:0, y1:0, x2:0, y2:1,
                stop:0 rgba(25, 50, 80, 0.7),
                stop:1 rgba(20, 40, 70, 0.9)
            );
        }
        
        QLineEdit {
            background: rgba(20, 40, 65, 0.5);
            color: #cfe9ff;
            border: 2px solid rgba(6, 182, 212, 0.3);
            border-radius: 12px;
            padding: 12px 18px;
            font-size: 15px;
            font-weight: 500;
        }
        
        QLineEdit:focus {
            border: 2px solid qlineargradient(
                x1:0, y1:0, x2:1, y2:0,
                stop:0 #06b6d4,
                stop:1 #3b82f6
            );
            background: rgba(25, 50, 85, 0.7);
        }
        
        QPushButton {
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:1,
                stop:0 #06b6d4,
                stop:1 #3b82f6
            );
            color: #ffffff;
            border: none;
            border-radius: 12px;
            font-weight: 600;
            font-size: 14px;
            padding: 8px 20px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        QPushButton:hover {
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:1,
                stop:0 #0891b2,
                stop:1 #2563eb
            );
            padding: 8px 22px;
        }
        
        QPushButton:pressed {
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:1,
                stop:0 #0e7490,
                stop:1 #1d4ed8
            );
            padding: 8px 18px;
        }
        
        IconButton {
            background: rgba(6, 182, 212, 0.15);
            color: #cfe9ff;
            border: 2px solid rgba(6, 182, 212, 0.3);
            border-radius: 25px;
            font-size: 18px;
        }
        
        IconButton:hover {
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:1,
                stop:0 rgba(6, 182, 212, 0.3),
                stop:1 rgba(59, 130, 246, 0.3)
            );
            border: 2px solid rgba(6, 182, 212, 0.6);
        }
        
        IconButton:pressed {
            background: rgba(6, 182, 212, 0.4);
        }
        
        QTextEdit {
            background: rgba(15, 30, 50, 0.5);
            color: #cfe9ff;
            border: 1px solid rgba(6, 182, 212, 0.2);
            border-radius: 12px;
            padding: 16px;
            font-size: 14px;
            line-height: 1.6;
        }
        
        QScrollBar:vertical {
            background: rgba(12, 30, 46, 0.3);
            width: 12px;
            border-radius: 6px;
            margin: 2px;
        }
        
        QScrollBar::handle:vertical {
            background: qlineargradient(
                x1:0, y1:0, x2:0, y2:1,
                stop:0 #06b6d4,
                stop:1 #3b82f6
            );
            border-radius: 6px;
            min-height: 40px;
        }
        
        QScrollBar::handle:vertical:hover {
            background: qlineargradient(
                x1:0, y1:0, x2:0, y2:1,
                stop:0 #0891b2,
                stop:1 #2563eb
            );
        }
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
        
        QLabel {
            background: transparent;
        }
    """,

    "sunset": """
        * {
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        QWidget {
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:1,
                stop:0 #1e1030,
                stop:1 #2d1b3d
            );
            color: #ffd4e5;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        
        QFrame#header {
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:0,
                stop:0 rgba(30, 16, 48, 0.95),
                stop:1 rgba(45, 27, 61, 0.95)
            );
            border: none;
            border-bottom: 2px solid qlineargradient(
                x1:0, y1:0, x2:1, y2:0,
                stop:0 #f97316,
                stop:0.5 #ec4899,
                stop:1 #a855f7
            );
        }
        
        QFrame#card {
            background: qlineargradient(
                x1:0, y1:0, x2:0, y2:1,
                stop:0 rgba(45, 27, 61, 0.6),
                stop:1 rgba(30, 16, 48, 0.8)
            );
            border-radius: 16px;
            border: 1px solid rgba(249, 115, 22, 0.3);
        }
        
        QFrame#card:hover {
            border: 1px solid rgba(236, 72, 153, 0.5);
            background: qlineargradient(
                x1:0, y1:0, x2:0, y2:1,
                stop:0 rgba(55, 35, 75, 0.7),
                stop:1 rgba(40, 25, 60, 0.9)
            );
        }
        
        QLineEdit {
            background: rgba(45, 27, 61, 0.5);
            color: #ffd4e5;
            border: 2px solid rgba(249, 115, 22, 0.3);
            border-radius: 12px;
            padding: 12px 18px;
            font-size: 15px;
            font-weight: 500;
        }
        
        QLineEdit:focus {
            border: 2px solid qlineargradient(
                x1:0, y1:0, x2:1, y2:0,
                stop:0 #f97316,
                stop:1 #ec4899
            );
            background: rgba(55, 35, 80, 0.7);
        }
        
        QPushButton {
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:1,
                stop:0 #f97316,
                stop:1 #ec4899
            );
            color: #ffffff;
            border: none;
            border-radius: 12px;
            font-weight: 600;
            font-size: 14px;
            padding: 8px 20px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        QPushButton:hover {
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:1,
                stop:0 #ea580c,
                stop:1 #db2777
            );
            padding: 8px 22px;
        }
        
        QPushButton:pressed {
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:1,
                stop:0 #c2410c,
                stop:1 #be185d
            );
            padding: 8px 18px;
        }
        
        IconButton {
            background: rgba(249, 115, 22, 0.15);
            color: #ffd4e5;
            border: 2px solid rgba(249, 115, 22, 0.3);
            border-radius: 25px;
            font-size: 18px;
        }
        
        IconButton:hover {
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:1,
                stop:0 rgba(249, 115, 22, 0.3),
                stop:1 rgba(236, 72, 153, 0.3)
            );
            border: 2px solid rgba(236, 72, 153, 0.6);
        }
        
        IconButton:pressed {
            background: rgba(249, 115, 22, 0.4);
        }
        
        QTextEdit {
            background: rgba(30, 16, 48, 0.5);
            color: #ffd4e5;
            border: 1px solid rgba(249, 115, 22, 0.2);
            border-radius: 12px;
            padding: 16px;
            font-size: 14px;
            line-height: 1.6;
        }
        
        QScrollBar:vertical {
            background: rgba(30, 16, 48, 0.3);
            width: 12px;
            border-radius: 6px;
            margin: 2px;
        }
        
        QScrollBar::handle:vertical {
            background: qlineargradient(
                x1:0, y1:0, x2:0, y2:1,
                stop:0 #f97316,
                stop:1 #ec4899
            );
            border-radius: 6px;
            min-height: 40px;
        }
        
        QScrollBar::handle:vertical:hover {
            background: qlineargradient(
                x1:0, y1:0, x2:0, y2:1,
                stop:0 #ea580c,
                stop:1 #db2777
            );
        }
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
        
        QLabel {
            background: transparent;
        }
    """
}

# Обновлённые метаданные тем
THEME_INFO = {
    "dark": {
        "name": "🌙 Cosmic Night",
        "name_en": "🌙 Cosmic Night",
        "description": "Глубокий космический дизайн с фиолетовыми акцентами",
        "description_en": "Deep cosmic design with purple accents",
        "colors": {
            "header": "#1a1a35",
            "body": "#0f0f23",
            "elements": "#2a2a50",
            "hover": "#6366f1"
        }
    },
    "light": {
        "name": "☀️ Crystal Clear",
        "name_en": "☀️ Crystal Clear",
        "description": "Кристально чистый светлый дизайн с градиентами",
        "description_en": "Crystal clear light design with gradients",
        "colors": {
            "header": "#ffffff",
            "body": "#f8fafc",
            "elements": "#e2e8f0",
            "hover": "#3b82f6"
        }
    },
    "ocean": {
        "name": "🌊 Deep Ocean",
        "name_en": "🌊 Deep Ocean",
        "description": "Глубокие океанские тона с бирюзовыми акцентами",
        "description_en": "Deep ocean tones with turquoise accents",
        "colors": {
            "header": "#0f1e2e",
            "body": "#0c1e2e",
            "elements": "#1a3a52",
            "hover": "#06b6d4"
        }
    },
    "sunset": {
        "name": "🌅 Sunset Dream",
        "name_en": "🌅 Sunset Dream",
        "description": "Тёплые закатные тона с оранжево-розовыми переходами",
        "description_en": "Warm sunset tones with orange-pink gradients",
        "colors": {
            "header": "#1e1030",
            "body": "#1e1030",
            "elements": "#2d1b3d",
            "hover": "#f97316"
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
