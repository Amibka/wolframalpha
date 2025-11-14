import os
import tempfile

import numpy as np
import plotly.graph_objects as go
from PyQt6.QtCore import QUrl
from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QFont
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import QTextEdit, QPushButton, QGraphicsOpacityEffect, QWidget, QVBoxLayout, QSizePolicy
from sympy import symbols, lambdify, sympify, solve

# Импортируем настройки
try:
    from config.plot_settings import *
except ImportError:
    # Если файла нет - используем дефолтные значения
    PLOT_2D_WIDTH = None
    PLOT_2D_HEIGHT = 600
    PLOT_2D_POINTS = 2000
    PLOT_2D_IMPLICIT_SIZE = 0.6
    PLOT_2D_IMPLICIT_POINTS = 400
    PLOT_3D_WIDTH = None
    PLOT_3D_HEIGHT = 700
    PLOT_3D_POINTS = 80
    DEFAULT_2D_RANGE = (-10, 10)
    DEFAULT_3D_X_RANGE = (-5, 5)
    DEFAULT_3D_Y_RANGE = (-5, 5)
    LINE_COLOR_2D = '#3b82f6'
    LINE_WIDTH_2D = 3
    LINE_COLOR_IMPLICIT = '#3b82f6'
    LINE_WIDTH_IMPLICIT = 4
    FILL_COLOR_IMPLICIT = 'rgba(59,130,246,0.08)'
    COLORSCALE_3D = 'Viridis'
    GRID_COLOR = 'rgba(148,163,184,0.2)'
    AXIS_COLOR = 'rgba(128,128,128,0.3)'
    TITLE_FONT_SIZE = 18
    TITLE_FONT_COLOR = '#1e293b'
    AXIS_TITLE_FONT_SIZE = 14
    AXIS_TITLE_FONT_COLOR = '#475569'
    TICK_FONT_SIZE = 11
    PLOT_BACKGROUND = 'white'
    PAPER_BACKGROUND = 'white'
    CONTOUR_SHOW_FILL = False
    CONTOUR_SHOW_LABELS = False
    CONTOUR_LINE_SMOOTHING = 1.3
    MARGIN_LEFT = 60
    MARGIN_RIGHT = 30
    MARGIN_TOP = 60
    MARGIN_BOTTOM = 60
    SHOW_TOOLBAR = True
    SHOW_LEGEND = True
    HOVER_MODE_2D = 'x unified'
    HOVER_MODE_IMPLICIT = 'closest'
    WIDGET_MIN_HEIGHT = 650
    WIDGET_PADDING = 10
    EXPORT_WIDTH = 1920
    EXPORT_HEIGHT = 1080
    EXPORT_SCALE = 2


class MathOutputWidget(QTextEdit):
    """Виджет для отображения математических результатов"""

    def __init__(self):
        super().__init__()
        self.setReadOnly(True)

        font = QFont("JetBrains Mono", 12)
        if not font.exactMatch():
            font = QFont("Consolas", 12)
            if not font.exactMatch():
                font = QFont("Courier New", 12)
        self.setFont(font)

        self.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setMinimumHeight(200)

    def display_result(self, result):
        from utils.math_formatter import format_solutions

        try:
            formatted_text = format_solutions(result)
            self.setPlainText(formatted_text)

            doc_height = self.document().size().height()
            new_height = min(int(doc_height) + 50, 900)
            new_height = max(new_height, 200)
            self.setMaximumHeight(new_height)

        except Exception as e:
            import traceback
            error_msg = f"❌ Ошибка отображения:\n{str(e)}\n\n{traceback.format_exc()}"
            self.setPlainText(error_msg)


class PlotWidget(QWidget):
    """Современный виджет для графиков на базе Plotly"""

    def __init__(self):
        super().__init__()

        # КРИТИЧНО: Устанавливаем политику размера
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(WIDGET_PADDING, WIDGET_PADDING, WIDGET_PADDING, WIDGET_PADDING)
        self.layout.setSpacing(0)
        # ЦЕНТРИРОВАНИЕ: выравниваем по центру
        self.layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setLayout(self.layout)

        # WebView для отображения интерактивных графиков Plotly
        self.web_view = QWebEngineView()

        # КРИТИЧНО: Фиксированная политика размера для центрирования
        self.web_view.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self.layout.addWidget(self.web_view)

        # КРИТИЧНО: Минимальная высота из настроек
        self.setMinimumHeight(WIDGET_MIN_HEIGHT)

        self.temp_files = []  # Для очистки временных файлов

    def clear_plot(self):
        """Очистка графика"""
        # Очищаем временные файлы
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass
        self.temp_files = []

        # Очищаем WebView
        self.web_view.setHtml("")

    def plot_2d(self, expr_str, var='x', x_range=None, num_points=None):
        """Построение 2D графика y = f(x)"""
        try:
            self.clear_plot()

            # Используем настройки из конфига
            if x_range is None:
                x_range = DEFAULT_2D_RANGE
            if num_points is None:
                num_points = PLOT_2D_POINTS

            # АДАПТИВНАЯ ШИРИНА: берём ширину виджета
            plot_width = PLOT_2D_WIDTH if PLOT_2D_WIDTH else self.width() - 100
            plot_height = PLOT_2D_HEIGHT

            # Парсим выражение
            x_sym = symbols(var)
            expr = sympify(expr_str)
            f = lambdify(x_sym, expr, modules=['numpy'])

            # Генерируем точки
            x_vals = np.linspace(x_range[0], x_range[1], num_points)

            try:
                y_vals = f(x_vals)
                if np.iscomplexobj(y_vals):
                    y_vals = np.real(y_vals)
                mask = np.isfinite(y_vals)
                x_vals = x_vals[mask]
                y_vals = y_vals[mask]
            except:
                y_vals = []
                valid_x = []
                for x_val in x_vals:
                    try:
                        y_val = complex(f(x_val))
                        if np.isfinite(y_val.real):
                            y_vals.append(y_val.real)
                            valid_x.append(x_val)
                    except:
                        pass
                x_vals = np.array(valid_x)
                y_vals = np.array(y_vals)

            if len(x_vals) == 0:
                raise ValueError("Не удалось вычислить функцию")

            # Создаем Plotly фигуру
            fig = go.Figure()

            # Основная кривая
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                mode='lines',
                name=f'y = {expr_str}',
                line=dict(color=LINE_COLOR_2D, width=LINE_WIDTH_2D),
                hovertemplate='x: %{x:.4f}<br>y: %{y:.4f}<extra></extra>'
            ))

            # Координатные оси
            fig.add_hline(y=0, line_dash="solid", line_color=AXIS_COLOR, line_width=1)
            fig.add_vline(x=0, line_dash="solid", line_color=AXIS_COLOR, line_width=1)

            # Стильное оформление
            fig.update_layout(
                title=dict(
                    text=f'<b>График: y = {expr_str}</b>',
                    font=dict(size=TITLE_FONT_SIZE, color=TITLE_FONT_COLOR)
                ),
                xaxis=dict(
                    title=dict(text=var, font=dict(size=AXIS_TITLE_FONT_SIZE, color=AXIS_TITLE_FONT_COLOR)),
                    showgrid=True,
                    gridwidth=1,
                    gridcolor=GRID_COLOR,
                    zeroline=False,
                    tickfont=dict(size=TICK_FONT_SIZE)
                ),
                yaxis=dict(
                    title=dict(text='y', font=dict(size=AXIS_TITLE_FONT_SIZE, color=AXIS_TITLE_FONT_COLOR)),
                    showgrid=True,
                    gridwidth=1,
                    gridcolor=GRID_COLOR,
                    zeroline=False,
                    tickfont=dict(size=TICK_FONT_SIZE)
                ),
                plot_bgcolor=PLOT_BACKGROUND,
                paper_bgcolor=PAPER_BACKGROUND,
                hovermode=HOVER_MODE_2D,
                showlegend=SHOW_LEGEND,
                legend=dict(
                    font=dict(size=12),
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='rgba(148,163,184,0.3)',
                    borderwidth=1
                ),
                width=plot_width,
                height=plot_height,
                margin=dict(l=MARGIN_LEFT, r=MARGIN_RIGHT, b=MARGIN_BOTTOM, t=MARGIN_TOP)
            )

            # КРИТИЧНО: Устанавливаем размер WebView под график
            self.web_view.setFixedSize(plot_width, plot_height)

            # Сохраняем и отображаем
            temp_file = self._save_and_display(fig)
            self.temp_files.append(temp_file)

            return True, "График успешно построен"

        except Exception as e:
            error_msg = f"Ошибка построения 2D графика: {str(e)}"
            self._show_error(error_msg)
            return False, error_msg

    def plot_2d_implicit(self, expr_str, var1='x', var2='y', x_range=None, y_range=None, num_points=None):
        """
        Построение неявного графика F(x,y) = 0
        ИСПРАВЛЕНО: Только одна чистая линия без заливки!
        """
        try:
            self.clear_plot()

            # Используем настройки из конфига
            if num_points is None:
                num_points = PLOT_2D_IMPLICIT_POINTS

            # АДАПТИВНЫЙ РАЗМЕР: квадрат 60% от высоты окна
            window_height = self.height()
            plot_size = int(window_height * PLOT_2D_IMPLICIT_SIZE)
            # Минимум 500, максимум 900
            plot_size = max(500, min(plot_size, 900))

            # Парсим выражение
            x_sym, y_sym = symbols(f'{var1} {var2}')
            equation = sympify(expr_str)

            # Автоопределение диапазона
            if x_range is None or y_range is None:
                try:
                    x_intercepts = solve(equation.subs(y_sym, 0), x_sym)
                    y_intercepts = solve(equation.subs(x_sym, 0), y_sym)

                    x_vals = [float(val.evalf()) for val in x_intercepts if val.is_real]
                    y_vals = [float(val.evalf()) for val in y_intercepts if val.is_real]

                    if x_vals and y_vals:
                        x_max = max(abs(v) for v in x_vals) * 1.3
                        y_max = max(abs(v) for v in y_vals) * 1.3
                    else:
                        x_max = y_max = 10
                except:
                    x_max = y_max = 10

                x_range = (-x_max, x_max)
                y_range = (-y_max, y_max)

            # Создаём сетку
            x_grid = np.linspace(x_range[0], x_range[1], num_points)
            y_grid = np.linspace(y_range[0], y_range[1], num_points)
            X, Y = np.meshgrid(x_grid, y_grid)

            # Вычисляем F(x,y)
            f = lambdify((x_sym, y_sym), equation, 'numpy')

            try:
                Z = f(X, Y)
            except Exception as e:
                raise ValueError(f"Не удалось вычислить функцию: {e}")

            # Создаем Plotly фигуру
            fig = go.Figure()

            # КРИТИЧНО: Только ОДНА кривая F(x,y) = 0 БЕЗ заливки!
            fig.add_trace(go.Contour(
                x=x_grid,
                y=y_grid,
                z=Z,
                contours=dict(
                    start=0,
                    end=0,
                    size=1,
                    coloring='none',  # БЕЗ ЗАЛИВКИ!
                    showlabels=CONTOUR_SHOW_LABELS
                ),
                line=dict(
                    color=LINE_COLOR_IMPLICIT,
                    width=LINE_WIDTH_IMPLICIT,
                    smoothing=CONTOUR_LINE_SMOOTHING
                ),
                name=f'{expr_str} = 0',
                showscale=False,
                hovertemplate='x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
            ))

            # Координатные оси
            fig.add_hline(y=0, line_dash="solid", line_color=AXIS_COLOR, line_width=1)
            fig.add_vline(x=0, line_dash="solid", line_color=AXIS_COLOR, line_width=1)

            # Стильное оформление
            fig.update_layout(
                title=dict(
                    text=f'<b>График: {expr_str} = 0</b>',
                    font=dict(size=TITLE_FONT_SIZE, color=TITLE_FONT_COLOR)
                ),
                xaxis=dict(
                    title=dict(text=var1, font=dict(size=AXIS_TITLE_FONT_SIZE, color=AXIS_TITLE_FONT_COLOR)),
                    showgrid=True,
                    gridwidth=1,
                    gridcolor=GRID_COLOR,
                    scaleanchor='y',  # КРИТИЧНО: равные масштабы
                    scaleratio=1,
                    tickfont=dict(size=TICK_FONT_SIZE),
                    constrain='domain'
                ),
                yaxis=dict(
                    title=dict(text=var2, font=dict(size=AXIS_TITLE_FONT_SIZE, color=AXIS_TITLE_FONT_COLOR)),
                    showgrid=True,
                    gridwidth=1,
                    gridcolor=GRID_COLOR,
                    tickfont=dict(size=TICK_FONT_SIZE),
                    constrain='domain'
                ),
                plot_bgcolor=PLOT_BACKGROUND,
                paper_bgcolor=PAPER_BACKGROUND,
                hovermode=HOVER_MODE_IMPLICIT,
                showlegend=SHOW_LEGEND,
                legend=dict(
                    font=dict(size=12),
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='rgba(148,163,184,0.3)',
                    borderwidth=1
                ),
                width=plot_size,
                height=plot_size,
                margin=dict(l=MARGIN_LEFT, r=MARGIN_RIGHT, b=MARGIN_BOTTOM, t=MARGIN_TOP)
            )

            # КРИТИЧНО: Устанавливаем фиксированный размер WebView для центрирования
            self.web_view.setFixedSize(plot_size, plot_size)

            # Сохраняем и отображаем
            temp_file = self._save_and_display(fig)
            self.temp_files.append(temp_file)

            return True, "Неявный график успешно построен"

        except Exception as e:
            error_msg = f"Ошибка построения неявного графика: {str(e)}"
            self._show_error(error_msg)
            return False, error_msg

    def plot_3d(self, expr_str, var1='x', var2='y', x_range=None, y_range=None, num_points=None):
        """3D график z = f(x,y)"""
        try:
            self.clear_plot()

            # Используем настройки из конфига
            if x_range is None:
                x_range = DEFAULT_3D_X_RANGE
            if y_range is None:
                y_range = DEFAULT_3D_Y_RANGE
            if num_points is None:
                num_points = PLOT_3D_POINTS

            # АДАПТИВНАЯ ШИРИНА
            plot_width = PLOT_3D_WIDTH if PLOT_3D_WIDTH else self.width() - 100
            plot_height = PLOT_3D_HEIGHT

            # Парсим выражение
            x_sym, y_sym = symbols(f'{var1} {var2}')
            expr = sympify(expr_str)
            f = lambdify((x_sym, y_sym), expr, modules=['numpy'])

            # Генерируем сетку
            x_vals = np.linspace(x_range[0], x_range[1], num_points)
            y_vals = np.linspace(y_range[0], y_range[1], num_points)
            X, Y = np.meshgrid(x_vals, y_vals)

            # Вычисляем Z
            try:
                Z = f(X, Y)
                if np.iscomplexobj(Z):
                    Z = np.real(Z)
                Z[~np.isfinite(Z)] = np.nan
            except:
                Z = np.zeros_like(X)
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        try:
                            val = complex(f(X[i, j], Y[i, j]))
                            Z[i, j] = val.real if np.isfinite(val.real) else np.nan
                        except:
                            Z[i, j] = np.nan

            # Создаем Plotly фигуру
            fig = go.Figure(data=[go.Surface(
                x=X, y=Y, z=Z,
                colorscale=COLORSCALE_3D,
                showscale=True,
                colorbar=dict(
                    title=dict(text="z", font=dict(size=13)),
                    tickfont=dict(size=11)
                ),
                contours=dict(
                    z=dict(
                        show=True,
                        usecolormap=True,
                        highlightcolor="limegreen",
                        project=dict(z=True)
                    )
                ),
                hovertemplate='x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}<extra></extra>'
            )])

            # Стильное 3D оформление
            fig.update_layout(
                title=dict(
                    text=f'<b>3D График: z = {expr_str}</b>',
                    font=dict(size=TITLE_FONT_SIZE, color=TITLE_FONT_COLOR)
                ),
                scene=dict(
                    xaxis=dict(
                        title=dict(text=var1, font=dict(size=AXIS_TITLE_FONT_SIZE)),
                        backgroundcolor="rgb(250, 250, 250)",
                        gridcolor="rgb(220, 220, 220)",
                        showbackground=True,
                        tickfont=dict(size=TICK_FONT_SIZE)
                    ),
                    yaxis=dict(
                        title=dict(text=var2, font=dict(size=AXIS_TITLE_FONT_SIZE)),
                        backgroundcolor="rgb(250, 250, 250)",
                        gridcolor="rgb(220, 220, 220)",
                        showbackground=True,
                        tickfont=dict(size=TICK_FONT_SIZE)
                    ),
                    zaxis=dict(
                        title=dict(text='z', font=dict(size=AXIS_TITLE_FONT_SIZE)),
                        backgroundcolor="rgb(250, 250, 250)",
                        gridcolor="rgb(220, 220, 220)",
                        showbackground=True,
                        tickfont=dict(size=TICK_FONT_SIZE)
                    ),
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.3)
                    )
                ),
                paper_bgcolor=PAPER_BACKGROUND,
                width=plot_width,
                height=plot_height,
                margin=dict(l=0, r=0, b=0, t=MARGIN_TOP)
            )

            # КРИТИЧНО: Устанавливаем размер WebView под график
            self.web_view.setFixedSize(plot_width, plot_height)

            # Сохраняем и отображаем
            temp_file = self._save_and_display(fig)
            self.temp_files.append(temp_file)

            return True, "3D график успешно построен"

        except Exception as e:
            error_msg = f"Ошибка построения 3D графика: {str(e)}"
            self._show_error(error_msg)
            return False, error_msg

    def _save_and_display(self, fig):
        """Сохраняет Plotly фигуру в HTML и отображает в WebView"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            fig.write_html(f.name, config={
                'displayModeBar': SHOW_TOOLBAR,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'graph',
                    'height': EXPORT_HEIGHT,
                    'width': EXPORT_WIDTH,
                    'scale': EXPORT_SCALE
                }
            })
            temp_file = f.name

        self.web_view.setUrl(QUrl.fromLocalFile(temp_file))
        return temp_file

    def _show_error(self, message):
        """Показать сообщение об ошибке"""
        self.clear_plot()

        error_html = f"""
        <html>
        <head>
            <style>
                body {{
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                }}
                .error-box {{
                    background: white;
                    padding: 40px;
                    border-radius: 20px;
                    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                    max-width: 600px;
                    text-align: center;
                }}
                .error-icon {{
                    font-size: 64px;
                    margin-bottom: 20px;
                }}
                .error-title {{
                    color: #ef4444;
                    font-size: 24px;
                    font-weight: bold;
                    margin-bottom: 15px;
                }}
                .error-message {{
                    color: #64748b;
                    font-size: 16px;
                    line-height: 1.6;
                }}
            </style>
        </head>
        <body>
            <div class="error-box">
                <div class="error-icon">⚠️</div>
                <div class="error-title">Ошибка построения графика</div>
                <div class="error-message">{message}</div>
            </div>
        </body>
        </html>
        """

        self.web_view.setHtml(error_html)


class ModernButton(QPushButton):
    """Современная кнопка"""

    def __init__(self, text="", icon_text=""):
        super().__init__()
        if icon_text:
            self.setText(f"{icon_text}  {text}")
        else:
            self.setText(text)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(40)

        self.opacity_effect = QGraphicsOpacityEffect()
        self.opacity_effect.setOpacity(1.0)
        self.setGraphicsEffect(self.opacity_effect)

    def enterEvent(self, event):
        self.animate_opacity(0.85)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.animate_opacity(1.0)
        super().leaveEvent(event)

    def animate_opacity(self, target_opacity):
        self.animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.animation.setDuration(200)
        self.animation.setStartValue(self.opacity_effect.opacity())
        self.animation.setEndValue(target_opacity)
        self.animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self.animation.start()


class IconButton(QPushButton):
    """Круглая кнопка-иконка"""

    def __init__(self, text="", tooltip=""):
        super().__init__(text)
        self.setFixedSize(45, 45)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        if tooltip:
            self.setToolTip(tooltip)

        self.opacity_effect = QGraphicsOpacityEffect()
        self.opacity_effect.setOpacity(1.0)
        self.setGraphicsEffect(self.opacity_effect)

    def enterEvent(self, event):
        self.animate_opacity(0.8)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.animate_opacity(1.0)
        super().leaveEvent(event)

    def animate_opacity(self, target_opacity):
        self.animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.animation.setDuration(200)
        self.animation.setStartValue(self.opacity_effect.opacity())
        self.animation.setEndValue(target_opacity)
        self.animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self.animation.start()


"""
ui/widgets.py - Кастомные виджеты
"""

from PyQt6.QtWidgets import QPushButton


class IconButton(QPushButton):
    """Кнопка с иконкой/эмодзи"""

    def __init__(self, icon_text, tooltip=""):
        super().__init__(icon_text)
        self.setToolTip(tooltip)

        # Увеличенные размеры для полного отображения эмодзи
        self.setFixedSize(50, 50)  # Было меньше

        # Стили с правильными отступами
        self.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 0.1);
                border: none;
                border-radius: 25px;
                font-size: 22px;  /* Увеличен размер шрифта */
                padding: 0px;     /* Убраны лишние отступы */
                margin: 0px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.2);
            }
            QPushButton:pressed {
                background-color: rgba(255, 255, 255, 0.3);
            }
        """)


class TextIconButton(QPushButton):
    """Кнопка с текстом (для языка EN/RU)"""

    def __init__(self, text, tooltip=""):
        super().__init__(text)
        self.setToolTip(tooltip)

        # Размеры для текстовых кнопок
        self.setFixedSize(55, 50)

        self.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 0.1);
                border: none;
                border-radius: 25px;
                font-size: 14px;
                font-weight: bold;
                padding: 0px;
                margin: 0px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.2);
            }
            QPushButton:pressed {
                background-color: rgba(255, 255, 255, 0.3);
            }
        """)
