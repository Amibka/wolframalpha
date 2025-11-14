"""
ui/examples_window.py - Окно с примерами возможностей программы
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QScrollArea, QWidget, QFrame,
    QGridLayout, QLineEdit
)


class ExampleCard(QFrame):
    """Карточка с примером"""

    example_clicked = pyqtSignal(str)  # Сигнал при клике на пример

    def __init__(self, category: str, title: str, expression: str, description: str):
        super().__init__()
        self.expression = expression

        self.setObjectName("example_card")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(15, 12, 15, 12)

        # Категория (метка)
        category_label = QLabel(category)
        category_label.setObjectName("category_badge")
        category_label.setMaximumWidth(120)
        category_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Заголовок
        title_label = QLabel(title)
        title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        title_label.setWordWrap(True)

        # Выражение
        expr_label = QLabel(f"<code>{expression}</code>")
        expr_label.setStyleSheet("""
            background: rgba(100, 100, 100, 0.2);
            padding: 8px;
            border-radius: 4px;
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 13px;
        """)
        expr_label.setWordWrap(True)
        expr_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        # Описание
        desc_label = QLabel(description)
        desc_label.setStyleSheet("font-size: 12px; opacity: 0.7;")
        desc_label.setWordWrap(True)

        layout.addWidget(category_label)
        layout.addWidget(title_label)
        layout.addWidget(expr_label)
        layout.addWidget(desc_label)

        self.setLayout(layout)

        # Стиль при наведении
        self.setStyleSheet("""
            QFrame#example_card {
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 8px;
                padding: 5px;
            }
            QFrame#example_card:hover {
                background: rgba(100, 150, 255, 0.15);
                border: 1px solid rgba(100, 150, 255, 0.3);
            }
            QLabel#category_badge {
                background: rgba(100, 150, 255, 0.3);
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 11px;
                font-weight: bold;
            }
        """)

    def mousePressEvent(self, event):
        """При клике на карточку"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.example_clicked.emit(self.expression)
        super().mousePressEvent(event)


class ExamplesWindow(QDialog):
    """Окно с примерами"""

    example_selected = pyqtSignal(str)  # Сигнал при выборе примера

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("📚 Примеры и возможности")
        self.setModal(False)
        self.resize(1000, 700)

        # Центрируем окно
        if parent:
            parent_geo = parent.geometry()
            x = parent_geo.x() + (parent_geo.width() - 1000) // 2
            y = parent_geo.y() + (parent_geo.height() - 700) // 2
            self.move(x, y)

        self.init_ui()
        self.apply_theme()

    def init_ui(self):
        """Инициализация UI"""
        layout = QVBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        # === Заголовок ===
        header = QFrame()
        header.setObjectName("examples_header")
        header.setFixedHeight(80)

        header_layout = QVBoxLayout()
        header_layout.setContentsMargins(30, 15, 30, 15)

        title = QLabel("📚 Примеры и возможности")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")

        subtitle = QLabel("Нажмите на любой пример, чтобы использовать его")
        subtitle.setStyleSheet("font-size: 13px; opacity: 0.7;")

        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        header.setLayout(header_layout)

        layout.addWidget(header)

        # === Поиск ===
        search_container = QWidget()
        search_layout = QHBoxLayout()
        search_layout.setContentsMargins(30, 15, 30, 15)

        self.search_field = QLineEdit()
        self.search_field.setPlaceholderText("🔍 Поиск по примерам...")
        self.search_field.setFixedHeight(40)
        self.search_field.textChanged.connect(self.filter_examples)

        search_layout.addWidget(self.search_field)
        search_container.setLayout(search_layout)

        layout.addWidget(search_container)

        # === Scroll Area с примерами ===
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        self.content_layout = QVBoxLayout()
        self.content_layout.setSpacing(25)
        self.content_layout.setContentsMargins(30, 10, 30, 30)

        # === ПРИМЕРЫ ===
        self.examples_data = self.get_examples()
        self.create_example_sections()

        self.content_layout.addStretch()
        content.setLayout(self.content_layout)
        scroll.setWidget(content)

        layout.addWidget(scroll)

        # === Кнопка закрытия ===
        close_btn = QPushButton("✕ Закрыть")
        close_btn.setFixedHeight(45)
        close_btn.clicked.connect(self.close)
        close_btn.setStyleSheet("""
            QPushButton {
                background: rgba(255, 70, 70, 0.2);
                border: 1px solid rgba(255, 70, 70, 0.3);
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover {
                background: rgba(255, 70, 70, 0.3);
            }
        """)

        layout.addWidget(close_btn)

        self.setLayout(layout)

    def get_examples(self) -> dict:
        """Возвращает все примеры, сгруппированные по категориям"""
        return {
            "🧮 Решение уравнений": [
                ("Линейное уравнение", "2*x + 5 = 13", "Простое линейное уравнение"),
                ("Квадратное уравнение", "x**2 - 5*x + 6 = 0", "Решение через дискриминант"),
                ("Тригонометрическое", "sin(x) = 1/2", "Найти все значения x"),
                ("Логарифмическое", "log10(x) = 2", "Логарифм по основанию 10"),
                ("Экспоненциальное", "e**x = 10", "Решение с экспонентой"),
            ],

            "📊 Построение графиков": [
                ("2D функция (явная)", "plot y = x**2 - 4*x + 3", "Парабола"),
                ("Окружность (неявная)", "plot x**2 + y**2 = 25", "Радиус 5"),
                ("Эллипс", "plot x**2/16 + y**2/9 = 1", "Центр в начале координат"),
                ("Гипербола", "plot x**2 - y**2 = 1", "Классическая гипербола"),
                ("3D поверхность", " plot z = x**2 + y**2", "Параболоид"),
                ("3D волна", "plot z = sin(x) * cos(y)", "Волновая поверхность"),
                ("Тригонометрия", "plot y = sin(x) + cos(2*x)", "Сложение волн"),
            ],

            "∫ Интегралы": [
                ("Неопределённый", "integral x**2 dx", "Степенная функция"),
                ("С тригонометрией", "integral sin(x) dx", "Синус"),
                ("Определённый", "integral x**2 dx from 0 to 2", "С пределами"),
                ("Короткая форма", "x**2 dx", "Автоматически integral"),
                ("Сложная функция", "integral sqrt(1 + x**2) dx", "Корень"),
                ("С логарифмом", "integral 1/x dx", "Натуральный логарифм"),
            ],

            "lim Пределы": [
                ("Простой предел", "limit x -> 0 sin(x)/x", "Первый замечательный"),
                ("На бесконечности", "limit x -> oo (1 + 1/x)**x ", "Число e"),
                ("Односторонний +", "limit x -> 0+ 1/x", "Справа от нуля"),
                ("Односторонний -", "limit x -> 0- 1/x", "Слева от нуля"),
                ("Короткая форма", "x -> 0 sin(x)/x", "Без слова limit"),
            ],

            "d/dx Производные": [
                ("Простая", "derivative x**3", "Степенная функция"),
                ("С указанием переменной", "derivative x**2 + y**2 по x", "По x"),
                ("Тригонометрическая", "derivative sin(x)*cos(x)", "Произведение"),
                ("Логарифм", "derivative ln(x)", "Натуральный логарифм"),
                ("Экспонента", "derivative e**x", "Экспонента"),
            ],

            "🔧 Упрощение": [
                ("Simplify", "simplify (x**2 - 1)/(x - 1)", "Сокращение"),
                ("Expand", "expand (x + 1)**3", "Раскрытие скобок"),
                ("Factor", "factor x**2 - 4", "Разложение на множители"),
                ("Trigsimp", "trigsimp sin(x)**2 + cos(x)**2", "Тригонометрия"),
                ("Logcombine", "logcombine log(x) + log(y)", "Объединение логарифмов"),
                ("Cancel", "cancel (x**2 - 1)/(x - 1)", "Сокращение дробей"),
            ],

            "🔬 Специальные функции": [
                ("Корень n-й степени", "root3(8)", "Кубический корень"),
                ("Корень с cbrt", "cbrt(27)", "Кубический корень из 27"),
                ("Логарифм по основанию", "log10(100)", "Логарифм по основанию 10")
            ],
        }

    def create_example_sections(self):
        """Создаёт секции с примерами"""
        self.all_cards = []  # Для поиска

        for category, examples in self.examples_data.items():
            # Заголовок категории
            category_title = QLabel(category)
            category_title.setStyleSheet("""
                font-size: 18px;
                font-weight: bold;
                padding: 10px 0px;
                border-bottom: 2px solid rgba(100, 150, 255, 0.3);
            """)
            self.content_layout.addWidget(category_title)

            # Сетка с карточками (3 колонки)
            grid = QGridLayout()
            grid.setSpacing(15)

            for i, (title, expression, description) in enumerate(examples):
                card = ExampleCard(category, title, expression, description)
                card.example_clicked.connect(self.on_example_clicked)

                row = i // 3
                col = i % 3
                grid.addWidget(card, row, col)

                self.all_cards.append((card, category.lower(), title.lower(), expression.lower()))

            self.content_layout.addLayout(grid)

    def filter_examples(self, text: str):
        """Фильтрует примеры по поисковому запросу"""
        query = text.lower().strip()

        for card, category, title, expression in self.all_cards:
            if not query:
                card.show()
            else:
                # Поиск в категории, заголовке или выражении
                if query in category or query in title or query in expression:
                    card.show()
                else:
                    card.hide()

    def on_example_clicked(self, expression: str):
        """Обработчик клика на пример"""
        self.example_selected.emit(expression)
        self.close()

    def apply_theme(self):
        """Применяет тему"""
        # Используем тёмную тему по умолчанию
        self.setStyleSheet("""
            QDialog {
                background: #1a1a2e;
                color: #ffffff;
            }
            QFrame#examples_header {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #2d2d44,
                    stop:1 #1a1a2e
                );
                border-bottom: 2px solid rgba(100, 150, 255, 0.3);
            }
            QLineEdit {
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 8px;
                padding: 10px;
                font-size: 14px;
                color: #ffffff;
            }
            QLineEdit:focus {
                border: 1px solid rgba(100, 150, 255, 0.5);
                background: rgba(255, 255, 255, 0.15);
            }
            QScrollArea {
                border: none;
                background: transparent;
            }
        """)
