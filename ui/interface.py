"""
ui/main_window.py - ИТОГОВАЯ ВЕРСИЯ С ПРИМЕРАМИ И ИСТОРИЕЙ
"""

from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QLabel, QFrame,
    QGraphicsOpacityEffect, QSizePolicy, QScrollArea
)
from ui.widgets import MathOutputWidget, ModernButton, PlotWidget
from ui.header import HeaderWidget
from ui.history_window import HistoryWindow
from ui.examples_window import ExamplesWindow
from ui.themes import get_theme
from utils.translations import translator, t
from handlers.event_handler import on_enter_pressed
from database.db_manager import DatabaseManager
import time


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.current_theme = "dark"
        self.current_language = "ru"

        # ИНИЦИАЛИЗАЦИЯ БД
        self.db_manager = DatabaseManager()

        translator.set_language(self.current_language)

        self.setWindowTitle(t("app.window_title"))

        # ДИНАМИЧЕСКИЙ РАЗМЕР ОКНА (80% от экрана)
        from PyQt6.QtGui import QGuiApplication
        screen = QGuiApplication.primaryScreen().geometry()
        window_width = int(screen.width() * 0.8)
        window_height = int(screen.height() * 0.85)

        # Центрируем окно на экране
        x = (screen.width() - window_width) // 2
        y = (screen.height() - window_height) // 2
        self.setGeometry(x, y, window_width, window_height)

        # Минимальный размер окна
        self.setMinimumSize(1000, 700)

        # Главный layout
        main_layout = QVBoxLayout()
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # === Header ===
        self.header = HeaderWidget()
        self.header.theme_changed.connect(self.on_theme_changed)
        self.header.language_changed.connect(self.toggle_language)
        self.header.history_requested.connect(self.show_history)
        self.header.examples_requested.connect(self.show_examples)  # ДОБАВЛЕНО!
        main_layout.addWidget(self.header)

        # === Scroll Area для контента ===
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        # Content Widget
        content_widget = QWidget()
        content_layout = QVBoxLayout()
        content_layout.setSpacing(20)
        content_layout.setContentsMargins(60, 30, 60, 30)

        # === Заголовок ===
        title_container = QWidget()
        title_layout = QVBoxLayout()
        title_layout.setSpacing(5)
        title_layout.setContentsMargins(0, 0, 0, 0)

        self.title_label = QLabel(t("app.title"))
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("font-size: 26px; font-weight: bold;")

        self.subtitle_label = QLabel(t("app.subtitle"))
        self.subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle_label.setStyleSheet("font-size: 13px; opacity: 0.7;")

        title_layout.addWidget(self.title_label)
        title_layout.addWidget(self.subtitle_label)
        title_container.setLayout(title_layout)

        content_layout.addWidget(title_container)

        # === Поле ввода ===
        input_container = QFrame()
        input_layout = QHBoxLayout()
        input_layout.setSpacing(10)
        input_layout.setContentsMargins(0, 0, 0, 0)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText(t("input.placeholder"))
        self.input_field.setFixedHeight(50)

        self.solve_button = ModernButton(t("buttons.solve"), "⚡")
        self.solve_button.setFixedWidth(140)

        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.solve_button)
        input_container.setLayout(input_layout)

        content_layout.addWidget(input_container)

        # === Карточка ввода ===
        self.input_card = QFrame()
        self.input_card.setObjectName("card")
        input_card_layout = QVBoxLayout()
        input_card_layout.setSpacing(10)
        input_card_layout.setContentsMargins(20, 15, 20, 15)

        self.input_label = QLabel(t("input.label"))
        self.input_label.setStyleSheet("font-weight: bold; font-size: 15px;")

        self.input_box = MathOutputWidget()
        self.input_box.setReadOnly(True)
        self.input_box.setFixedHeight(60)

        input_card_layout.addWidget(self.input_label)
        input_card_layout.addWidget(self.input_box)
        self.input_card.setLayout(input_card_layout)

        self.input_card_opacity = QGraphicsOpacityEffect()
        self.input_card.setGraphicsEffect(self.input_card_opacity)

        content_layout.addWidget(self.input_card)

        # === Карточка результата ===
        self.result_card = QFrame()
        self.result_card.setObjectName("card")
        result_card_layout = QVBoxLayout()
        result_card_layout.setSpacing(10)
        result_card_layout.setContentsMargins(20, 15, 20, 15)

        self.result_label = QLabel(t("output.label"))
        self.result_label.setStyleSheet("font-weight: bold; font-size: 15px;")

        self.output_widget = MathOutputWidget()

        result_card_layout.addWidget(self.result_label)
        result_card_layout.addWidget(self.output_widget)
        self.result_card.setLayout(result_card_layout)

        self.result_card_opacity = QGraphicsOpacityEffect()
        self.result_card.setGraphicsEffect(self.result_card_opacity)

        content_layout.addWidget(self.result_card)

        # === Карточка графика ===
        self.plot_card = QFrame()
        self.plot_card.setObjectName("card")
        self.plot_card.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        plot_card_layout = QVBoxLayout()
        plot_card_layout.setSpacing(10)
        plot_card_layout.setContentsMargins(20, 15, 20, 15)

        self.plot_label = QLabel(t("plot.label"))
        self.plot_label.setStyleSheet("font-weight: bold; font-size: 15px;")

        self.plot_widget = PlotWidget()
        self.plot_widget.setMinimumHeight(750)

        plot_card_layout.addWidget(self.plot_label)
        plot_card_layout.addWidget(self.plot_widget)
        self.plot_card.setLayout(plot_card_layout)

        self.plot_card_opacity = QGraphicsOpacityEffect()
        self.plot_card.setGraphicsEffect(self.plot_card_opacity)

        content_layout.addWidget(self.plot_card)

        # Добавляем растягивающийся элемент
        content_layout.addStretch()

        content_widget.setLayout(content_layout)
        scroll.setWidget(content_widget)

        main_layout.addWidget(scroll)
        self.setLayout(main_layout)

        # Скрываем карточки
        self.input_card.hide()
        self.result_card.hide()
        self.plot_card.hide()

        # События
        self.input_field.returnPressed.connect(lambda: self.handle_input())
        self.solve_button.clicked.connect(lambda: self.handle_input())

        # Тема
        self.apply_theme()

    def on_theme_changed(self, theme):
        """Обработчик смены темы"""
        self.current_theme = theme
        self.apply_theme()

    def toggle_language(self):
        """Переключает язык"""
        self.current_language = "eng" if self.current_language == "ru" else "ru"
        translator.set_language(self.current_language)
        self.header.update_language_button(self.current_language)
        self.update_translations()

    def show_history(self):
        """Показывает окно истории"""
        history_window = HistoryWindow(self.db_manager, self)
        history_window.entry_selected.connect(self.load_history_entry)
        history_window.exec()

    def show_examples(self):
        """Показывает окно с примерами"""
        examples_window = ExamplesWindow(self)
        examples_window.example_selected.connect(self.load_example)
        examples_window.exec()

    def load_example(self, expression: str):
        """Загружает пример из окна примеров"""
        # Вставляем выражение в поле ввода
        self.input_field.setText(expression)
        self.input_field.setFocus()

        # Можно показать подсказку
        print(f"📚 Загружен пример: {expression}")

    def load_history_entry(self, entry: dict):
        """Загружает запись из истории"""
        # Устанавливаем ввод
        self.input_field.setText(entry['input_text'])

        # Показываем ввод
        self.input_box.setPlainText(entry['input_text'])
        if not self.input_card.isVisible():
            self.show_card_animated(self.input_card, self.input_card_opacity, 0)

        # Показываем результат
        if entry['result_type'] == 'error':
            self.plot_card.hide()
            self.output_widget.setPlainText(f"{entry['error_message']}")
            if not self.result_card.isVisible():
                self.show_card_animated(self.result_card, self.result_card_opacity, 100)

        elif entry['result_type'].startswith('plot'):
            self.result_card.hide()
            # Можно перестроить график из сохранённых данных
            if not self.plot_card.isVisible():
                self.show_card_animated(self.plot_card, self.plot_card_opacity, 100)

        else:
            self.plot_card.hide()
            if entry['result_text']:
                self.output_widget.setPlainText(entry['result_text'])
            if not self.result_card.isVisible():
                self.show_card_animated(self.result_card, self.result_card_opacity, 100)

    def update_translations(self):
        """Обновляет переводы"""
        self.title_label.setText(t("app.title"))
        self.subtitle_label.setText(t("app.subtitle"))
        self.input_field.setPlaceholderText(t("input.placeholder"))
        self.solve_button.setText(f"⚡  {t('buttons.solve')}")
        self.input_label.setText(t("input.label"))
        self.result_label.setText(t("output.label"))
        self.plot_label.setText(t("plot.label"))
        self.header.logo_label.setText(t("header.logo"))
        self.header.lang_button.setToolTip(t("header.language_tooltip"))
        self.header.theme_button.setToolTip(t("header.theme_tooltip"))

    def apply_theme(self):
        """Применяет тему"""
        self.setStyleSheet(get_theme(self.current_theme))

    def show_card_animated(self, card, opacity_effect, delay=0):
        """Плавно показывает карточку"""
        card.show()

        self.fade_animation = QPropertyAnimation(opacity_effect, b"opacity")
        self.fade_animation.setDuration(300)
        self.fade_animation.setStartValue(0.0)
        self.fade_animation.setEndValue(1.0)
        self.fade_animation.setEasingCurve(QEasingCurve.Type.OutCubic)

        if delay > 0:
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(delay, self.fade_animation.start)
        else:
            self.fade_animation.start()

    def handle_input(self):
        user_input = self.input_field.text().strip()
        if not user_input:
            return

        self.input_box.setPlainText(user_input)

        # Показываем карточку ввода
        if not self.input_card.isVisible():
            self.show_card_animated(self.input_card, self.input_card_opacity, 0)

        # Засекаем время
        start_time = time.time()

        # Получаем результат
        result = on_enter_pressed(self.input_field, self.output_widget)

        # Считаем время выполнения
        execution_time = time.time() - start_time

        # СОХРАНЯЕМ В БД
        self._save_to_database(user_input, result, execution_time)

        # Проверяем тип
        if isinstance(result, dict):
            result_type = result.get('type')

            if result_type == 'plot_2d':
                # 2D график
                self.result_card.hide()

                expr = result['expression']
                var = result['variables'][0] if result['variables'] else 'x'

                print(f"Строим 2D график: y = {expr}")
                self.plot_widget.plot_2d(expr, var=var)

                if not self.plot_card.isVisible():
                    self.show_card_animated(self.plot_card, self.plot_card_opacity, 100)

            elif result_type == 'plot_2d_implicit':
                # Неявный график
                self.result_card.hide()

                expr = result['expression']
                var1 = result['variables'][0] if len(result['variables']) > 0 else 'x'
                var2 = result['variables'][1] if len(result['variables']) > 1 else 'y'

                print(f"Строим неявный график: {expr} = 0")
                self.plot_widget.plot_2d_implicit(expr, var1=var1, var2=var2)

                if not self.plot_card.isVisible():
                    self.show_card_animated(self.plot_card, self.plot_card_opacity, 100)

            elif result_type == 'plot_3d':
                # 3D график
                self.result_card.hide()

                expr = result['expression']
                var1 = result['variables'][0] if len(result['variables']) > 0 else 'x'
                var2 = result['variables'][1] if len(result['variables']) > 1 else 'y'

                print(f"Строим 3D график: z = {expr}")
                self.plot_widget.plot_3d(expr, var1=var1, var2=var2)

                if not self.plot_card.isVisible():
                    self.show_card_animated(self.plot_card, self.plot_card_opacity, 100)

            elif result_type == 'error':
                # Ошибка
                self.plot_card.hide()

                error_message = result.get('message', 'Неизвестная ошибка')
                self.output_widget.setPlainText(f"{error_message}")

                if not self.result_card.isVisible():
                    self.show_card_animated(self.result_card, self.result_card_opacity, 100)
        else:
            # Обычный результат
            self.plot_card.hide()
            if not self.result_card.isVisible():
                self.show_card_animated(self.result_card, self.result_card_opacity, 100)

    def _save_to_database(self, user_input: str, result, execution_time: float):
        """Сохраняет результат в БД"""
        try:
            # Определяем команду из ввода
            user_lower = user_input.lower()
            command = 'solve'  # По умолчанию

            if any(kw in user_lower for kw in ['plot', 'график', 'graph']):
                command = 'plot'
            elif any(kw in user_lower for kw in ['derivative', 'производная']):
                command = 'derivative'
            elif any(kw in user_lower for kw in ['integral', 'интеграл', 'integrate']):
                command = 'integral'
            elif any(kw in user_lower for kw in ['limit', 'предел', 'lim']):
                command = 'limit'
            elif any(kw in user_lower for kw in ['simplify', 'упростить']):
                command = 'simplify'
            elif any(kw in user_lower for kw in ['expand', 'раскрыть']):
                command = 'expand'
            elif any(kw in user_lower for kw in ['factor', 'факторизовать']):
                command = 'factor'

            if isinstance(result, dict):
                # Результат - dict (график или ошибка)
                result_type = result.get('type', 'unknown')
                expression = result.get('expression', user_input)

                if result_type == 'error':
                    error_message = result.get('message', 'Неизвестная ошибка')
                    result_text = None
                    result_json = None
                else:
                    error_message = None
                    result_text = f"[{result_type}] {expression}"
                    result_json = result

            else:
                # Обычный результат (строка или объект SymPy)
                result_type = 'success'
                expression = user_input
                error_message = None

                # КРИТИЧНО: Конвертируем результат в строку
                if result is None:
                    result_text = "None"
                else:
                    try:
                        result_text = str(result)
                    except:
                        result_text = repr(result)

                result_json = None

            # Определяем теги
            tags = []
            if 'plot' in user_lower or 'график' in user_lower:
                tags.append('график')
            if any(kw in user_lower for kw in ['solve', 'решить', '=']):
                tags.append('уравнение')
            if any(kw in user_lower for kw in ['derivative', 'производная']):
                tags.append('производная')
            if any(kw in user_lower for kw in ['integral', 'интеграл']):
                tags.append('интеграл')
            if any(kw in user_lower for kw in ['limit', 'предел']):
                tags.append('предел')

            # Сохраняем
            self.db_manager.add_entry(
                input_text=user_input,
                command=command,
                expression=expression,
                result_type=result_type,
                result_text=result_text,
                result_json=result_json,
                error_message=error_message,
                execution_time=execution_time,
                tags=tags
            )

            print(f"Запись сохранена в БД (время: {execution_time:.3f}s)")

        except Exception as e:
            import traceback
            print(f"Ошибка сохранения в БД: {e}")
            print(traceback.format_exc())
