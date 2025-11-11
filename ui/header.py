"""
ui/header.py - ОБНОВЛЁННАЯ ВЕРСИЯ С ЛОГОТИПОМ
"""

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel
from ui.widgets import IconButton, TextIconButton
import os


class HeaderWidget(QFrame):
    """Верхняя панель приложения"""

    theme_changed = pyqtSignal(str)
    language_changed = pyqtSignal()
    history_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setObjectName("header")
        self.setFixedHeight(65)
        self.setup_ui()

    def setup_ui(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(25, 0, 25, 0)
        layout.setSpacing(15)

        # Логотип с иконкой
        logo_layout = QHBoxLayout()
        logo_layout.setSpacing(10)

        # Иконка
        icon_label = QLabel()
        logo_path = "assets/logo.png"

        # Проверка существования файла
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            icon_label.setPixmap(
                pixmap.scaled(
                    40, 40,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
            )
        else:
            # Fallback если файл не найден
            icon_label.setText("🧮")
            icon_label.setStyleSheet("font-size: 30px;")

        logo_layout.addWidget(icon_label)

        # Текст
        self.logo_label = QLabel("WolframAlpha")
        self.logo_label.setStyleSheet("font-size: 17px; font-weight: bold;")
        logo_layout.addWidget(self.logo_label)

        layout.addLayout(logo_layout)
        layout.addStretch()

        # Кнопка истории
        self.history_button = IconButton("📚", "Открыть историю вычислений")
        self.history_button.clicked.connect(self.on_history_clicked)
        layout.addWidget(self.history_button)

        # Кнопка смены языка (текстовая)
        self.lang_button = TextIconButton("EN", "Сменить язык / Change language")
        self.lang_button.clicked.connect(self.on_language_clicked)
        layout.addWidget(self.lang_button)

        # Кнопка настроек темы
        self.theme_button = IconButton("🎨", "Сменить тему")
        self.theme_button.clicked.connect(self.on_theme_clicked)
        layout.addWidget(self.theme_button)

        self.setLayout(layout)

    def on_history_clicked(self):
        """Обработчик клика на кнопку истории"""
        self.history_requested.emit()

    def on_language_clicked(self):
        """Обработчик клика на кнопку смены языка"""
        self.language_changed.emit()

    def on_theme_clicked(self):
        """Обработчик клика на кнопку смены темы"""
        from ui.settings import SettingsDialog
        dialog = SettingsDialog(self.parent())
        if dialog.exec():
            self.theme_changed.emit(dialog.selected_theme)

    def update_language_button(self, current_lang):
        """Обновляет текст на кнопке языка"""
        self.lang_button.setText("ENG" if current_lang == "ru" else "RU")
