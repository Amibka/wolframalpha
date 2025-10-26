from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLineEdit, QLabel, QTextEdit, QSpacerItem, QSizePolicy
)
from handlers.event_handler import hide_widget
from handlers.event_handler import on_enter_pressed


# Основное окно приложения
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Поле чуть выше центра")
        self.showMaximized()

        # === Поле для ввода исходных данных ===
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Введите текст...")
        self.input_field.setFixedWidth(600)
        self.input_field.setFixedHeight(40)

        # === Основной layout ===
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        # Отступ сверху
        spacer_top = QSpacerItem(
            0, self.height() // 4,

            QSizePolicy.Policy.Minimum,
            QSizePolicy.Policy.Expanding
        )
        layout.addItem(spacer_top)

        layout.addWidget(self.input_field, alignment=Qt.AlignmentFlag.AlignHCenter)

        # === Надпись и поле "Input" ===
        self.input_label = QLabel("Input")
        self.input_label.setFixedHeight(30)
        self.input_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        # Поле для отображения введенного текста
        self.input_box = QTextEdit()
        self.input_box.setFixedWidth(600)
        self.input_box.setFixedHeight(40)
        self.input_box.setReadOnly(True)

        layout.addWidget(self.input_label, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.input_box, alignment=Qt.AlignmentFlag.AlignHCenter)

        # === Надпись и поле "Решение" ===
        self.result_label = QLabel("Решение")
        self.result_label.setFixedHeight(30)
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        self.output_box = QTextEdit()
        self.output_box.setFixedWidth(600)
        self.output_box.setFixedHeight(40)
        self.output_box.setReadOnly(True)

        layout.addWidget(self.result_label, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.output_box, alignment=Qt.AlignmentFlag.AlignHCenter)

        layout.addStretch(1)
        self.setLayout(layout)

        # Скрываем блоки до нажатия Enter
        hide_widget(self.input_box)
        hide_widget(self.input_label)
        hide_widget(self.output_box)
        hide_widget(self.result_label)

        # Связь Enter с функцией
        self.input_field.returnPressed.connect(
            lambda: on_enter_pressed(
                self.input_field,
                self.output_box,
                self.input_box,
                self.input_label,
                self.output_box,
                self.input_box,
                self.result_label
            )
        )

# def load_stylesheet(path: str) -> str:
#    """Загружает CSS-файл для оформления интерфейса"""
#    full_path = os.path.join(os.path.dirname(__file__), path)
#    with open(full_path, "r", encoding="utf-8") as f:
#        return f.read()
