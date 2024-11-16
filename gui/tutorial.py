from edifice import App, Label, TextInput, View, Window, component, use_state

METERS_TO_FEET = 3.28084

def str_to_float(s):
    try:
        return float(s)
    except ValueError:
        return 0.0

@component
def MyApp2(self):

    meters, meters_set = use_state("0.0")

    feet = "%.3f" % (str_to_float(meters) * METERS_TO_FEET)

    meters_label_style = {"width": 170}
    feet_label_style = {"margin-left": 20, "width": 200}
    input_style = {"padding": 2, "width": 120}

    with View(layout="row", style={"margin": 10, "width": 560}):
        Label("Measurement in meters:", style=meters_label_style)
        TextInput(meters, style=input_style, on_change=meters_set)
        Label(f"Measurement in feet: {feet}", style=feet_label_style)

@component
def MyApp(self):
    with Window():
        with View():
            MyApp2()

if __name__ == "__main__":
    App(MyApp()).start()
