import tkinter as tk


def press(key):
    current = display_var.get()
    if key == "C":
        display_var.set("")
    elif key == "=":
        try:
            display_var.set(str(eval(current)))
        except Exception:
            display_var.set("Error")
    else:
        display_var.set(current + key)


root = tk.Tk()
root.title("Calculator")

display_var = tk.StringVar()
display = tk.Entry(root, textvariable=display_var, font=("Arial", 24), justify="right")
display.grid(row=0, column=0, columnspan=4, padx=5, pady=5, sticky="nsew")

buttons = [
    ("7", "8", "9", "/"),
    ("4", "5", "6", "*"),
    ("1", "2", "3", "-"),
    ("C", "0", "=", "+"),
]

for r, row in enumerate(buttons, start=1):
    for c, text in enumerate(row):
        tk.Button(root, text=text, font=("Arial", 18), command=lambda t=text: press(t)).grid(
            row=r, column=c, padx=2, pady=2, sticky="nsew"
        )

root.mainloop()
