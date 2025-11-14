"""
ui/header.py - ФИНАЛЬНАЯ ВЕРСИЯ с логотипом и кнопкой примеров
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
    examples_requested = pyqtSignal()  # НОВЫЙ СИГНАЛ!

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
        self.logo_label = QLabel("Mathly")
        self.logo_label.setStyleSheet("font-size: 17px; font-weight: bold;")
        logo_layout.addWidget(self.logo_label)

        layout.addLayout(logo_layout)
        layout.addStretch()

        # === КНОПКА ПРИМЕРОВ (НОВАЯ!) ===
        self.examples_button = IconButton("📚", "Примеры и возможности")
        self.examples_button.clicked.connect(self.on_examples_clicked)
        self.examples_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(100, 150, 255, 0.25),
                    stop:1 rgba(150, 100, 255, 0.25)
                );
                border: 1px solid rgba(100, 150, 255, 0.4);
                border-radius: 8px;
                padding: 8px 12px;
                font-size: 20px;
                color: #ffffff;
            }
            QPushButton:hover {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(100, 150, 255, 0.4),
                    stop:1 rgba(150, 100, 255, 0.4)
                );
                border: 1px solid rgba(100, 150, 255, 0.6);
            }
            QPushButton:pressed {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(100, 150, 255, 0.3),
                    stop:1 rgba(150, 100, 255, 0.3)
                );
            }
        """)
        layout.addWidget(self.examples_button)

        # Кнопка истории
        self.history_button = IconButton("🕒", "Открыть историю вычислений")
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

    def on_examples_clicked(self):
        """Обработчик клика на кнопку примеров"""
        self.examples_requested.emit()

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
