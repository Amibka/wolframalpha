from PyQt6.QtWidgets import QApplication
from ui.interface import MainWindow
import sys


# Точка входа в приложение
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


# Запуск приложения
if __name__ == "__main__":
    main()
