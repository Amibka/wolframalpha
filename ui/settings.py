"""
ui/settings.py - СОВРЕМЕННЫЙ ДИАЛОГ НАСТРОЕК
"""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QRadioButton, QButtonGroup
)

from ui.themes import THEME_INFO
from utils.translations import t


class ThemePreview(QFrame):
    """Превью темы с современным дизайном"""

    def __init__(self, theme_name, colors):
        super().__init__()
        self.theme_name = theme_name
        self.colors = colors
        self.setFixedSize(220, 140)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Верхняя полоска (header)
        header = QFrame()
        header.setFixedHeight(35)
        header.setStyleSheet(f"""
            background: {self.colors['header']};
            border-radius: 12px 12px 0 0;
        """)
        layout.addWidget(header)

        # Основная область
        body = QFrame()
        body.setStyleSheet(f"""
            background: {self.colors['body']};
            border-radius: 0 0 12px 12px;
        """)
        body_layout = QVBoxLayout()
        body_layout.setContentsMargins(12, 12, 12, 12)
        body_layout.setSpacing(8)

        # Имитация элементов с градиентом
        for i in range(3):
            element = QFrame()
            element.setFixedHeight(18)
            opacity = 1.0 - (i * 0.15)
            element.setStyleSheet(f"""
                background: {self.colors['elements']};
                border-radius: 6px;
                opacity: {opacity};
            """)
            body_layout.addWidget(element)

        body.setLayout(body_layout)
        layout.addWidget(body)

        self.setLayout(layout)
        self.setStyleSheet(f"""
            QFrame {{
                border: 3px solid transparent;
                border-radius: 14px;
            }}
            QFrame:hover {{
                border-color: {self.colors['hover']};
                transform: scale(1.02);
            }}
        """)


class SettingsDialog(QDialog):
    """Современный диалог настроек темы"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_theme = parent.current_theme if parent else "dark"
        self.current_language = parent.current_language if parent else "ru"
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("⚙️ " + t("settings.title"))
        self.setMinimumWidth(700)
        self.setMinimumHeight(650)

        layout = QVBoxLayout()
        layout.setSpacing(25)
        layout.setContentsMargins(35, 35, 35, 35)

        # === Заголовок ===
        header_layout = QVBoxLayout()
        header_layout.setSpacing(10)

        title = QLabel(t("settings.theme_selection"))
        title.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(title)

        subtitle = QLabel(t("settings.subtitle"))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("""
            color: rgba(255, 255, 255, 0.6);
            font-size: 14px;
            letter-spacing: 0.5px;
        """)
        header_layout.addWidget(subtitle)

        layout.addLayout(header_layout)
        layout.addSpacing(10)

        # === Темы ===
        self.theme_buttons = {}
        self.button_group = QButtonGroup()

        for theme_id in ["dark", "light", "ocean", "sunset"]:
            theme_info = THEME_INFO[theme_id]

            name_key = "name" if self.current_language == "ru" else "name_en"
            desc_key = "description" if self.current_language == "ru" else "description_en"

            theme_data = {
                "name": theme_info[name_key],
                "description": theme_info[desc_key],
                "colors": theme_info["colors"]
            }

            theme_frame = self.create_theme_card(theme_id, theme_data)
            layout.addWidget(theme_frame)

        layout.addStretch()

        # === Кнопки действий ===
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()

        cancel_btn = QPushButton(f"✕  {t('buttons.cancel')}")
        cancel_btn.setFixedSize(140, 45)
        cancel_btn.setStyleSheet("""
            QPushButton {
                background: rgba(239, 68, 68, 0.2);
                color: #fff;
                border: 2px solid rgba(239, 68, 68, 0.4);
                border-radius: 10px;
                font-weight: 600;
                font-size: 14px;
            }
            QPushButton:hover {
                background: rgba(239, 68, 68, 0.3);
                border-color: rgba(239, 68, 68, 0.6);
            }
            QPushButton:pressed {
                background: rgba(239, 68, 68, 0.4);
            }
        """)
        cancel_btn.clicked.connect(self.reject)
        buttons_layout.addWidget(cancel_btn)

        apply_btn = QPushButton(f"✓  {t('buttons.apply')}")
        apply_btn.setFixedSize(140, 45)
        apply_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #10b981,
                    stop:1 #059669
                );
                color: #fff;
                border: none;
                border-radius: 10px;
                font-weight: 600;
                font-size: 14px;
            }
            QPushButton:hover {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #059669,
                    stop:1 #047857
                );
            }
            QPushButton:pressed {
                background: #047857;
            }
        """)
        apply_btn.clicked.connect(self.accept)
        apply_btn.setDefault(True)
        buttons_layout.addWidget(apply_btn)

        layout.addLayout(buttons_layout)

        self.setLayout(layout)
        self.apply_dialog_style()

    def create_theme_card(self, theme_id, theme_data):
        """Создает карточку темы"""
        card = QFrame()
        card.setObjectName("themeCard")
        card.setFixedHeight(170)

        card_layout = QHBoxLayout()
        card_layout.setContentsMargins(20, 20, 20, 20)
        card_layout.setSpacing(25)

        # Превью
        preview = ThemePreview(theme_id, theme_data["colors"])
        card_layout.addWidget(preview)

        # Информация
        info_layout = QVBoxLayout()
        info_layout.setSpacing(12)

        # Radio button + название
        radio_layout = QHBoxLayout()
        radio = QRadioButton()
        radio.setChecked(theme_id == self.selected_theme)
        self.button_group.addButton(radio)
        self.theme_buttons[theme_id] = radio
        radio_layout.addWidget(radio)

        name_label = QLabel(theme_data["name"])
        name_label.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        radio_layout.addWidget(name_label)
        radio_layout.addStretch()

        info_layout.addLayout(radio_layout)

        # Описание
        desc_label = QLabel(theme_data["description"])
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("""
            color: rgba(255, 255, 255, 0.6);
            font-size: 13px;
            line-height: 1.5;
        """)
        info_layout.addWidget(desc_label)

        info_layout.addStretch()

        card_layout.addLayout(info_layout, 1)

        card.setLayout(card_layout)

        # Делаем карточку кликабельной
        card.mousePressEvent = lambda e: self.select_theme(theme_id)
        preview.mousePressEvent = lambda e: self.select_theme(theme_id)

        return card

    def select_theme(self, theme_id):
        """Выбирает тему"""
        self.selected_theme = theme_id
        self.theme_buttons[theme_id].setChecked(True)

    def apply_dialog_style(self):
        """Применяет стили к диалогу"""
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0f0f23,
                    stop:1 #1a1a35
                );
                color: #e0e7ff;
            }

            QFrame#themeCard {
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(30, 30, 60, 0.6),
                    stop:1 rgba(20, 20, 50, 0.8)
                );
                border: 2px solid rgba(99, 102, 241, 0.3);
                border-radius: 14px;
            }

            QFrame#themeCard:hover {
                border-color: rgba(139, 92, 246, 0.6);
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(35, 35, 70, 0.7),
                    stop:1 rgba(25, 25, 60, 0.9)
                );
            }

            QRadioButton {
                font-size: 14px;
                spacing: 10px;
                color: #e0e7ff;
            }

            QRadioButton::indicator {
                width: 22px;
                height: 22px;
                border-radius: 11px;
                border: 3px solid rgba(99, 102, 241, 0.5);
                background: rgba(30, 30, 60, 0.5);
            }

            QRadioButton::indicator:checked {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #6366f1,
                    stop:1 #8b5cf6
                );
                border-color: #8b5cf6;
            }

            QRadioButton::indicator:hover {
                border-color: #8b5cf6;
                background: rgba(99, 102, 241, 0.2);
            }

            QLabel {
                background: transparent;
                color: #e0e7ff;
            }
        """)