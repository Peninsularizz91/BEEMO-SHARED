import tkinter as tk
from PIL import Image, ImageTk, ImageSequence
import os

def show_loading_screen():
    root = tk.Tk()
    root.title("Loading Beemo 2.0")
    root.geometry("800x600")  # Adjust the size as needed

    # Load the GIF background
    base_path = os.path.dirname(__file__)
    gif_path = os.path.join(base_path, 'Icon', 'world-bee-day-save-the-bees-1732550224707.gif')
    gif_image = Image.open(gif_path)

    # Create a label to display the GIF
    gif_label = tk.Label(root)
    gif_label.place(x=0, y=0, relwidth=1, relheight=1)

    # Function to animate the GIF
    def animate_gif(counter):
        try:
            gif_frame = next(gif_sequence)
            gif_photo = ImageTk.PhotoImage(gif_frame)
            gif_label.config(image=gif_photo)
            gif_label.image = gif_photo
            root.after(50, lambda: animate_gif(counter + 1))
        except Exception as e:
            print(f"Error animating GIF: {e}")

    # Convert the GIF into a sequence of frames
    gif_sequence = ImageSequence.Iterator(gif_image)
    animate_gif(0)

    # Load the PNG for the foreground
    png_path = os.path.join(base_path, 'Icon', 'icon4.png')
    png_image = Image.open(png_path)
    png_photo = ImageTk.PhotoImage(png_image)

    # Create a label to display the PNG
    png_label = tk.Label(root, image=png_photo, bg='yellow')
    png_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    # Display the loading screen for 5 seconds
    root.after(5000, root.destroy)
    root.mainloop()