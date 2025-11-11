from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QRadioButton, QButtonGroup
)
from PyQt6.QtGui import QFont
from ui.themes import THEME_INFO
from utils.translations import translator, t


class ThemePreview(QFrame):
    """Превью темы"""

    def __init__(self, theme_name, colors):
        super().__init__()
        self.theme_name = theme_name
        self.colors = colors
        self.setFixedSize(200, 120)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Верхняя полоска (header)
        header = QFrame()
        header.setFixedHeight(30)
        header.setStyleSheet(f"background-color: {self.colors['header']}; border-radius: 8px 8px 0 0;")
        layout.addWidget(header)

        # Основная область
        body = QFrame()
        body.setStyleSheet(f"""
            background-color: {self.colors['body']};
            border-radius: 0 0 8px 8px;
        """)
        body_layout = QVBoxLayout()
        body_layout.setContentsMargins(10, 10, 10, 10)
        body_layout.setSpacing(5)

        # Имитация элементов
        for i in range(3):
            element = QFrame()
            element.setFixedHeight(15)
            element.setStyleSheet(f"""
                background-color: {self.colors['elements']};
                border-radius: 4px;
            """)
            body_layout.addWidget(element)

        body.setLayout(body_layout)
        layout.addWidget(body)

        self.setLayout(layout)
        self.setStyleSheet(f"""
            QFrame {{
                border: 3px solid transparent;
                border-radius: 10px;
            }}
            QFrame:hover {{
                border-color: {self.colors['hover']};
            }}
        """)


class SettingsDialog(QDialog):
    """Диалог настроек темы"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_theme = parent.current_theme if parent else "dark"
        self.current_language = parent.current_language if parent else "ru"
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle(t("settings.title"))
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)

        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)

        # Заголовок
        title = QLabel(t("settings.theme_selection"))
        title.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel(t("settings.subtitle"))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: gray; font-size: 13px;")
        layout.addWidget(subtitle)

        layout.addSpacing(20)

        # Темы
        self.theme_buttons = {}
        self.button_group = QButtonGroup()

        for theme_id in ["dark", "light", "ocean"]:
            theme_info = THEME_INFO[theme_id]

            # Получаем локализованное название и описание
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

        # Кнопки действий
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()

        cancel_btn = QPushButton(f"❌ {t('buttons.cancel')}")
        cancel_btn.setFixedSize(120, 40)
        cancel_btn.clicked.connect(self.reject)
        buttons_layout.addWidget(cancel_btn)

        apply_btn = QPushButton(f"✅ {t('buttons.apply')}")
        apply_btn.setFixedSize(120, 40)
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
        card.setFixedHeight(150)

        card_layout = QHBoxLayout()
        card_layout.setContentsMargins(20, 15, 20, 15)
        card_layout.setSpacing(20)

        # Превью
        preview = ThemePreview(theme_id, theme_data["colors"])
        card_layout.addWidget(preview)

        # Информация
        info_layout = QVBoxLayout()
        info_layout.setSpacing(8)

        # Radio button + название
        radio_layout = QHBoxLayout()
        radio = QRadioButton()
        radio.setChecked(theme_id == self.selected_theme)
        self.button_group.addButton(radio)
        self.theme_buttons[theme_id] = radio
        radio_layout.addWidget(radio)

        name_label = QLabel(theme_data["name"])
        name_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        radio_layout.addWidget(name_label)
        radio_layout.addStretch()

        info_layout.addLayout(radio_layout)

        # Описание
        desc_label = QLabel(theme_data["description"])
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: gray; font-size: 12px;")
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
                background-color: #1a1a2e;
                color: #eee;
            }
            QFrame#themeCard {
                background-color: #16213e;
                border: 2px solid #0f3460;
                border-radius: 12px;
            }
            QFrame#themeCard:hover {
                border-color: #7b68ee;
                background-color: #1c2540;
            }
            QRadioButton {
                font-size: 14px;
                spacing: 8px;
            }
            QRadioButton::indicator {
                width: 20px;
                height: 20px;
                border-radius: 10px;
                border: 2px solid #533483;
                background-color: #0f3460;
            }
            QRadioButton::indicator:checked {
                background-color: #7b68ee;
                border-color: #7b68ee;
            }
            QRadioButton::indicator:hover {
                border-color: #7b68ee;
            }
            QPushButton {
                background-color: #7b68ee;
                color: white;
                border: none;
                border-radius: 8px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #9381ff;
            }
            QPushButton:pressed {
                background-color: #6554c0;
            }
            QLabel {
                background: transparent;
            }
        """)