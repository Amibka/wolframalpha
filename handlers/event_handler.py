from core.parser import get_text


def hide_widget(widget):
    widget.hide()


def show_widget(widget):
    widget.show()


def on_enter_pressed(input_widget, output_widget, *extra_widgets):
    user_input = input_widget.text()

    for widget in extra_widgets:
        widget.show()

    output_widget.setText(f"{get_text(user_input)}")
