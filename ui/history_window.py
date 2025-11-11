"""
ui/history_window.py
Окно истории вычислений
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QLineEdit, QComboBox,
    QFrame, QMessageBox, QHeaderView, QWidget, QMenu
)
from PyQt6.QtGui import QFont, QColor, QAction
from datetime import datetime
import json


class HistoryWindow(QDialog):
    """Окно просмотра истории вычислений"""

    entry_selected = pyqtSignal(dict)  # Сигнал при выборе записи

    def __init__(self, db_manager, parent=None):
        super().__init__(parent)
        self.db_manager = db_manager

        self.setWindowTitle("📚 История вычислений")
        self.resize(1200, 700)

        self.setup_ui()
        self.load_history()
        self.load_statistics()

    def setup_ui(self):
        """Создаёт интерфейс"""
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # === Header ===
        header = QHBoxLayout()

        title = QLabel("📚 История вычислений")
        title.setStyleSheet("font-size: 22px; font-weight: bold;")
        header.addWidget(title)

        header.addStretch()

        # Статистика
        self.stats_label = QLabel()
        self.stats_label.setStyleSheet("font-size: 13px; opacity: 0.7;")
        header.addWidget(self.stats_label)

        layout.addLayout(header)

        # === Фильтры ===
        filters_frame = QFrame()
        filters_frame.setObjectName("card")
        filters_layout = QHBoxLayout()
        filters_layout.setContentsMargins(15, 10, 15, 10)

        # Поиск
        search_label = QLabel("🔍 Поиск:")
        filters_layout.addWidget(search_label)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Поиск по тексту...")
        self.search_input.setFixedWidth(250)
        self.search_input.textChanged.connect(self.on_search)
        filters_layout.addWidget(self.search_input)

        filters_layout.addSpacing(20)

        # Фильтр по команде
        command_label = QLabel("⚙️ Команда:")
        filters_layout.addWidget(command_label)

        self.command_filter = QComboBox()
        self.command_filter.addItems([
            "Все",
            "solve", "plot", "derivative", "integral", "limit",
            "simplify", "expand", "factor"
        ])
        self.command_filter.setFixedWidth(150)
        self.command_filter.currentTextChanged.connect(self.on_filter_changed)
        filters_layout.addWidget(self.command_filter)

        filters_layout.addSpacing(20)

        # Избранное
        self.favorites_btn = QPushButton("⭐ Только избранное")
        self.favorites_btn.setCheckable(True)
        self.favorites_btn.clicked.connect(self.on_filter_changed)
        filters_layout.addWidget(self.favorites_btn)

        filters_layout.addStretch()

        # Кнопки действий
        clear_btn = QPushButton("🗑️ Очистить")
        clear_btn.clicked.connect(self.on_clear_history)
        filters_layout.addWidget(clear_btn)

        export_btn = QPushButton("💾 Экспорт")
        export_btn.clicked.connect(self.on_export)
        filters_layout.addWidget(export_btn)

        import_btn = QPushButton("📥 Импорт")
        import_btn.clicked.connect(self.on_import)
        filters_layout.addWidget(import_btn)

        filters_frame.setLayout(filters_layout)
        layout.addWidget(filters_frame)

        # === Таблица истории ===
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "⭐", "ID", "Время", "Команда", "Ввод", "Результат", "⏱️"
        ])

        # Настройка колонок
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)  # Избранное
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)  # ID
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)  # Время
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)  # Команда
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)  # Ввод
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)  # Результат
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)  # Время выполнения

        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.show_context_menu)
        self.table.cellDoubleClicked.connect(self.on_cell_double_clicked)

        layout.addWidget(self.table)

        # === Footer ===
        footer = QHBoxLayout()

        self.info_label = QLabel("Выберите запись для просмотра деталей")
        self.info_label.setStyleSheet("font-size: 12px; opacity: 0.6;")
        footer.addWidget(self.info_label)

        footer.addStretch()

        close_btn = QPushButton("Закрыть")
        close_btn.clicked.connect(self.close)
        footer.addWidget(close_btn)

        layout.addLayout(footer)

        self.setLayout(layout)

    def load_history(self):
        """Загружает историю из БД"""
        # Получаем фильтры
        command = self.command_filter.currentText()
        if command == "Все":
            command = None

        search = self.search_input.text() or None
        favorites = self.favorites_btn.isChecked()

        # Загружаем данные
        history = self.db_manager.get_history(
            limit=500,
            command_filter=command,
            search_query=search,
            favorites_only=favorites
        )

        # Очищаем таблицу
        self.table.setRowCount(0)

        # Заполняем таблицу
        for entry in history:
            self.add_table_row(entry)

        self.info_label.setText(f"Найдено записей: {len(history)}")

    def add_table_row(self, entry: dict):
        """Добавляет строку в таблицу"""
        row = self.table.rowCount()
        self.table.insertRow(row)

        # Избранное
        fav_item = QTableWidgetItem("⭐" if entry['favorite'] else "")
        fav_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.setItem(row, 0, fav_item)

        # ID
        id_item = QTableWidgetItem(str(entry['id']))
        id_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.setItem(row, 1, id_item)

        # Время
        timestamp = datetime.fromisoformat(entry['timestamp'])
        time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        time_item = QTableWidgetItem(time_str)
        self.table.setItem(row, 2, time_item)

        # Команда
        cmd_item = QTableWidgetItem(entry['command'] or "—")
        cmd_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

        # Цвет для команды
        if entry['result_type'] == 'error':
            cmd_item.setForeground(QColor("#ef4444"))
        elif entry['result_type'].startswith('plot'):
            cmd_item.setForeground(QColor("#3b82f6"))

        self.table.setItem(row, 3, cmd_item)

        # Ввод
        input_item = QTableWidgetItem(entry['input_text'][:100])
        self.table.setItem(row, 4, input_item)

        # Результат
        if entry['result_type'] == 'error':
            result_text = f"❌ {entry['error_message'][:80]}"
        elif entry['result_text']:
            result_text = entry['result_text'][:100]
        else:
            result_text = f"[{entry['result_type']}]"

        result_item = QTableWidgetItem(result_text)
        self.table.setItem(row, 5, result_item)

        # Время выполнения
        exec_time = entry.get('execution_time')
        time_text = f"{exec_time:.3f}s" if exec_time else "—"
        time_exec_item = QTableWidgetItem(time_text)
        time_exec_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.setItem(row, 6, time_exec_item)

        # Сохраняем entry в data для доступа
        id_item.setData(Qt.ItemDataRole.UserRole, entry)

    def load_statistics(self):
        """Загружает статистику"""
        stats = self.db_manager.get_statistics()

        text = f"Всего: {stats['total']} | "
        text += f"Избранное: {stats['favorites']} | "
        text += f"Ошибок: {stats['errors']} | "
        text += f"Ср. время: {stats['avg_execution_time']:.3f}s"

        self.stats_label.setText(text)

    def on_search(self):
        """Обработчик поиска"""
        self.load_history()

    def on_filter_changed(self):
        """Обработчик изменения фильтров"""
        self.load_history()

    def show_context_menu(self, pos):
        """Показывает контекстное меню"""
        item = self.table.itemAt(pos)
        if not item:
            return

        row = item.row()
        id_item = self.table.item(row, 1)
        entry = id_item.data(Qt.ItemDataRole.UserRole)

        menu = QMenu(self)

        # Избранное
        fav_text = "Убрать из избранного" if entry['favorite'] else "Добавить в избранное"
        fav_action = QAction(f"⭐ {fav_text}", self)
        fav_action.triggered.connect(lambda: self.toggle_favorite(entry['id']))
        menu.addAction(fav_action)

        menu.addSeparator()

        # Копировать ввод
        copy_input_action = QAction("📋 Копировать ввод", self)
        copy_input_action.triggered.connect(lambda: self.copy_to_clipboard(entry['input_text']))
        menu.addAction(copy_input_action)

        # Копировать результат
        if entry['result_text']:
            copy_result_action = QAction("📋 Копировать результат", self)
            copy_result_action.triggered.connect(lambda: self.copy_to_clipboard(entry['result_text']))
            menu.addAction(copy_result_action)

        menu.addSeparator()

        # Удалить
        delete_action = QAction("🗑️ Удалить", self)
        delete_action.triggered.connect(lambda: self.delete_entry(entry['id']))
        menu.addAction(delete_action)

        menu.exec(self.table.viewport().mapToGlobal(pos))

    def toggle_favorite(self, entry_id: int):
        """Переключает избранное"""
        self.db_manager.toggle_favorite(entry_id)
        self.load_history()
        self.load_statistics()

    def delete_entry(self, entry_id: int):
        """Удаляет запись"""
        reply = QMessageBox.question(
            self,
            "Подтверждение",
            "Удалить эту запись?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.db_manager.delete_entry(entry_id)
            self.load_history()
            self.load_statistics()

    def copy_to_clipboard(self, text: str):
        """Копирует текст в буфер обмена"""
        from PyQt6.QtWidgets import QApplication
        QApplication.clipboard().setText(text)
        self.info_label.setText("✅ Скопировано в буфер обмена")

    def on_cell_double_clicked(self, row: int, column: int):
        """Обработчик двойного клика"""
        id_item = self.table.item(row, 1)
        entry = id_item.data(Qt.ItemDataRole.UserRole)

        # Отправляем сигнал с записью
        self.entry_selected.emit(entry)

        # Можно закрыть окно или оставить открытым
        # self.close()

    def on_clear_history(self):
        """Очистка истории"""
        reply = QMessageBox.question(
            self,
            "Подтверждение",
            "Очистить всю историю?\n(Избранное будет сохранено)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.db_manager.clear_history(keep_favorites=True)
            self.load_history()
            self.load_statistics()

    def on_export(self):
        """Экспорт истории"""
        from PyQt6.QtWidgets import QFileDialog

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Экспорт истории",
            "history_export.json",
            "JSON Files (*.json)"
        )

        if filepath:
            try:
                self.db_manager.export_to_json(filepath)
                QMessageBox.information(self, "Успех", "История успешно экспортирована!")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка экспорта: {e}")

    def on_import(self):
        """Импорт истории"""
        from PyQt6.QtWidgets import QFileDialog

        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Импорт истории",
            "",
            "JSON Files (*.json)"
        )

        if filepath:
            try:
                self.db_manager.import_from_json(filepath)
                self.load_history()
                self.load_statistics()
                QMessageBox.information(self, "Успех", "История успешно импортирована!")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка импорта: {e}")