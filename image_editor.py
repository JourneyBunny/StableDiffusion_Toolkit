import tkinter as tk
from tkinter import Text, filedialog, StringVar
from PIL import Image, ImageTk
import numpy as np
import os
import make_picture  # Import the make_picture.py file

class BrushLabelingApp:
    def __init__(self, root, brush_radius=10):
        self.root = root
        self.root.title("Stable Diffusion Image Tool")

        # Model selection setup
        self.default_model_name = "SDXL"  # Initial default model
        self.selected_model = StringVar(value=self.default_model_name)

        # Fetch available model names from the keys of make_picture.model_info
        self.model_names = list(make_picture.model_info.keys())

        # Set initial blank canvas size
        self.canvas_width = 1024
        self.canvas_height = 1024
        self.brush_radius = brush_radius
        self.drawing = False  # Flag to check if user is actively drawing
        self.brush_circle = None  # Brush outline circle
        self.mask_visible = True  # Flag to track visibility of mask
        self.mask_states = []  # List to keep track of mask states for undo functionality
        self.mask_cleared = False  # Flag to indicate if the mask was cleared by generate or inpaint
        self.base_image_path = None  # Path to the base image for reload functionality

        # Batch generation variables
        self.batch_amount = tk.IntVar(value=1)
        self.batch_filename = tk.StringVar(value="batch_output")

        # Set up labeled pixels mask
        self.labeled_pixels_mask = np.zeros((self.canvas_height, self.canvas_width), dtype=bool)
        
        # Create blank image to start with
        self.image = Image.new("RGBA", (self.canvas_width, self.canvas_height), (255, 255, 255, 255))
        self.photo = ImageTk.PhotoImage(self.image)

        # Set up canvas to display the image
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height)
        self.canvas.grid(row=0, column=0, columnspan=3)
        self.canvas_image_id = self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

        # Overlay for showing labeled pixels
        self.overlay = Image.new("RGBA", (self.canvas_width, self.canvas_height))
        self.overlay_tk = ImageTk.PhotoImage(self.overlay)
        self.overlay_canvas = self.canvas.create_image(0, 0, anchor="nw", image=self.overlay_tk)

        # UI Elements at the bottom
        control_frame = tk.Frame(root)
        control_frame.grid(row=1, column=0, columnspan=3, pady=10, sticky="w")

        # Model selection row
        model_label = tk.Label(control_frame, text="Select Model:")
        model_label.grid(row=0, column=0, padx=(10, 5), pady=5, sticky="w")

        self.model_dropdown = tk.OptionMenu(control_frame, self.selected_model, *self.model_names)
        self.model_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # Textbox for prompt in the same column as dropdown, spanning both columns 0 and 1
        self.prompt_textbox = Text(control_frame, height=3, width=50)
        self.prompt_textbox.grid(row=1, column=0, padx=10, pady=5, columnspan=2, sticky="w")

        # Button frame in the second column (Generate, Hide mask, Inpaint)
        button_frame = tk.Frame(control_frame)
        button_frame.grid(row=0, column=2, rowspan=2, padx=10, pady=5, sticky="n")

        self.generate_button = tk.Button(button_frame, text="Generate", command=self.generate_image)
        self.generate_button.pack(side=tk.TOP, fill=tk.X, pady=2)

        self.toggle_mask_button = tk.Button(button_frame, text="Hide mask", command=self.toggle_mask, state="disabled")
        self.toggle_mask_button.pack(side=tk.TOP, fill=tk.X, pady=2)

        self.inpaint_button = tk.Button(button_frame, text="Inpaint", command=self.inpaint_image, state="disabled")
        self.inpaint_button.pack(side=tk.TOP, fill=tk.X, pady=2)

        # Undo, Reload, Open, Save buttons in the third column
        undo_frame = tk.Frame(control_frame)
        undo_frame.grid(row=0, column=3, rowspan=2, padx=10, pady=5, sticky="n")

        self.undo_button = tk.Button(undo_frame, text="Undo", command=self.undo_mask, state="disabled")
        self.undo_button.pack(side=tk.TOP, fill=tk.X, pady=2)

        self.reload_button = tk.Button(undo_frame, text="Reload", command=self.reload_image, state="disabled")
        self.reload_button.pack(side=tk.TOP, fill=tk.X, pady=2)

        self.open_button = tk.Button(undo_frame, text="Open", command=self.open_image)
        self.open_button.pack(side=tk.TOP, fill=tk.X, pady=2)

        self.save_button = tk.Button(undo_frame, text="Save", command=self.save_image, state="disabled")
        self.save_button.pack(side=tk.TOP, fill=tk.X, pady=2)

        # Batch frame in the fourth column
        batch_frame = tk.Frame(control_frame)
        batch_frame.grid(row=0, column=4, rowspan=2, padx=10, pady=5, sticky="n")

        batch_label = tk.Label(batch_frame, text="Batch")
        batch_label.pack(side=tk.TOP, fill=tk.X, pady=2)

        self.batch_spinbox = tk.Spinbox(batch_frame, from_=1, to=100, textvariable=self.batch_amount, width=5)
        self.batch_spinbox.pack(side=tk.TOP, fill=tk.X, pady=2)

        self.batch_filename_entry = tk.Entry(batch_frame, textvariable=self.batch_filename, state="disabled")
        self.batch_filename_entry.pack(side=tk.TOP, fill=tk.X, pady=2)
        self.batch_filename_entry.insert(0, "batch_output")

        # Now set up the trace after batch_filename_entry is defined
        self.batch_amount.trace('w', self.batch_amount_changed)

        # Bind events
        self.canvas.bind("<Motion>", self.update_brush_position)
        self.canvas.bind("<Button-1>", self.start_brush)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.stop_brush)
        self.root.bind("<MouseWheel>", self.adjust_brush_radius)  # Bind scroll wheel
        self.root.bind("<KeyPress-[>", self.decrease_radius)       # Bind [ key
        self.root.bind("<KeyPress-]>", self.increase_radius)       # Bind ] key
        self.root.bind("<KeyPress-Up>", self.increase_radius)      # Bind Up arrow key
        self.root.bind("<KeyPress-Down>", self.decrease_radius)    # Bind Down arrow key

    def batch_amount_changed(self, *args):
        try:
            batch_value = int(self.batch_amount.get())
            if batch_value > 1:
                self.batch_filename_entry.config(state="normal")
            else:
                self.batch_filename_entry.config(state="disabled")
        except ValueError:
            # Invalid input, disable the batch filename entry
            self.batch_filename_entry.config(state="disabled")

    def update_brush_position(self, event):
        if not self.drawing:
            x, y = event.x, event.y
            self.draw_brush_circle(x, y)
        
    def draw_brush_circle(self, x, y):
        if self.brush_circle:
            self.canvas.delete(self.brush_circle)
        
        r = self.brush_radius
        self.brush_circle = self.canvas.create_oval(
            x - r, y - r, x + r, y + r,
            outline="red", width=2, tags="brush_outline"
        )

    def start_brush(self, event):
        self.drawing = True
        self.canvas.delete("brush_outline")  # Hide brush outline
        self.paint(event)  # Start painting immediately
        
    def paint(self, event):
        x, y = event.x, event.y
        self.update_mask_and_overlay(x, y)

    def stop_brush(self, event):
        self.drawing = False
        self.update_brush_position(event)  # Restore outline at current position
        # Save the current mask state for undo functionality
        self.save_mask_state()

    def save_mask_state(self):
        # Save a copy of the current mask
        self.mask_states.append(self.labeled_pixels_mask.copy())
        self.mask_cleared = False  # New mask state added, so mask is not cleared
        self.update_undo_button_state()

    def update_mask_and_overlay(self, cx, cy):
        y_grid, x_grid = np.ogrid[-self.brush_radius:self.brush_radius+1, -self.brush_radius:self.brush_radius+1]
        mask = x_grid**2 + y_grid**2 <= self.brush_radius**2

        x_start, x_end = max(0, cx - self.brush_radius), min(self.canvas_width, cx + self.brush_radius + 1)
        y_start, y_end = max(0, cy - self.brush_radius), min(self.canvas_height, cy + self.brush_radius + 1)

        self.labeled_pixels_mask[y_start:y_end, x_start:x_end][mask[:y_end-y_start, :x_end-x_start]] = True
        self.redraw_overlay()
        self.update_toggle_button_state()  # Update toggle button based on label data
        self.update_inpaint_button_state()  # Update inpaint button based on label data
        self.update_generate_button_state()  # Update generate button based on label data

    def redraw_overlay(self):
        if self.mask_visible:
            overlay_data = np.array(self.overlay)
            overlay_data[..., :3] = 0  # Clear previous color (RGB channels)
            overlay_data[self.labeled_pixels_mask] = [0, 0, 255, 128]  # RGBA blue with transparency

            self.overlay = Image.fromarray(overlay_data, "RGBA")
            self.overlay_tk = ImageTk.PhotoImage(self.overlay)
            self.canvas.itemconfig(self.overlay_canvas, image=self.overlay_tk)
        else:
            # If mask is hidden, display a transparent overlay
            self.overlay = Image.new("RGBA", (self.canvas_width, self.canvas_height), (255, 255, 255, 0))
            self.overlay_tk = ImageTk.PhotoImage(self.overlay)
            self.canvas.itemconfig(self.overlay_canvas, image=self.overlay_tk)

    def adjust_brush_radius(self, event):
        if event.delta > 0:
            self.brush_radius += 1
        elif event.delta < 0 and self.brush_radius > 1:
            self.brush_radius -= 1
        self.update_brush_position(event)  # Update outline with new radius

    def increase_radius(self, event=None):
        self.brush_radius += 5
        self.update_brush_position(event)

    def decrease_radius(self, event=None):
        if self.brush_radius > 5:
            self.brush_radius -= 5
            self.update_brush_position(event)

    def generate_image(self):
        # Get prompt text from the textbox
        prompt = self.prompt_textbox.get("1.0", "end-1c").strip()
        selected_model = self.selected_model.get()  # Get the selected model
        if not prompt:
            print("Please enter a prompt.")
            return

        # Get batch amount
        try:
            batch_value = int(self.batch_amount.get())
        except ValueError:
            batch_value = 1

        if batch_value <= 1:
            # Existing logic
            output_path = "generated_image.png"
            make_picture.generate_image(prompt=prompt, output=output_path, model_name=selected_model)

            # Load the generated image
            self.image = Image.open(output_path).convert("RGBA")
            self.photo = ImageTk.PhotoImage(self.image)
            
            # Display the generated image on the canvas by updating the image item
            self.canvas.itemconfig(self.canvas_image_id, image=self.photo)
            
            # Reset the overlay and labeled pixels mask for a new image
            self.labeled_pixels_mask = np.zeros((self.canvas_height, self.canvas_width), dtype=bool)
            # Keep only the last state
            if self.mask_states:
                self.mask_states = [self.mask_states[-1]]
            else:
                self.mask_states = []
            self.mask_cleared = True
            self.undo_button.config(text="Restore mask")
            self.update_undo_button_state()

            self.overlay = Image.new("RGBA", (self.canvas_width, self.canvas_height))
            self.overlay_tk = ImageTk.PhotoImage(self.overlay)
            self.canvas.itemconfig(self.overlay_canvas, image=self.overlay_tk)
            
            # Update base image path
            self.base_image_path = output_path
            # Disable Reload button since we just generated a new image
            self.reload_button.config(state="disabled")
            # Enable Save button
            self.save_button.config(state="normal")

            # Disable the toggle mask button and inpaint button since there's no label data
            self.update_toggle_button_state()
            self.update_inpaint_button_state()
            self.update_generate_button_state()
        else:
            # Batch generation logic
            batch_name = self.batch_filename.get().strip()
            if not batch_name:
                batch_name = "batch_output"

            for i in range(1, batch_value + 1):
                output_path = f"{batch_name}_{i}.png"
                make_picture.generate_image(prompt=prompt, output=output_path, model_name=selected_model)
                print(f"Generated image saved to {output_path}")

            # Optionally, load the last generated image into the canvas
            last_output_path = f"{batch_name}_{batch_value}.png"
            if os.path.exists(last_output_path):
                self.image = Image.open(last_output_path).convert("RGBA")
                self.photo = ImageTk.PhotoImage(self.image)
                self.canvas.itemconfig(self.canvas_image_id, image=self.photo)
                
                # Reset the overlay and labeled pixels mask for a new image
                self.labeled_pixels_mask = np.zeros((self.canvas_height, self.canvas_width), dtype=bool)
                # Keep only the last state
                if self.mask_states:
                    self.mask_states = [self.mask_states[-1]]
                else:
                    self.mask_states = []
                self.mask_cleared = True
                self.undo_button.config(text="Restore mask")
                self.update_undo_button_state()

                self.overlay = Image.new("RGBA", (self.canvas_width, self.canvas_height))
                self.overlay_tk = ImageTk.PhotoImage(self.overlay)
                self.canvas.itemconfig(self.overlay_canvas, image=self.overlay_tk)
                
                # Update base image path
                self.base_image_path = last_output_path
                # Disable Reload button since we just generated a new image
                self.reload_button.config(state="disabled")
                # Enable Save button
                self.save_button.config(state="normal")

                # Disable the toggle mask button and inpaint button since there's no label data
                self.update_toggle_button_state()
                self.update_inpaint_button_state()
                self.update_generate_button_state()
            else:
                print(f"Failed to load the last generated image: {last_output_path}")

    def toggle_mask(self):
        # Toggle mask visibility
        self.mask_visible = not self.mask_visible
        self.redraw_overlay()
        
        # Update button text
        self.toggle_mask_button.config(text="Show mask" if not self.mask_visible else "Hide mask")
        # Update inpaint and generate button state
        self.update_inpaint_button_state()
        self.update_generate_button_state()

    def update_toggle_button_state(self):
        # Enable the toggle mask button if there's label data; otherwise, disable it
        if np.any(self.labeled_pixels_mask):
            self.toggle_mask_button.config(state="normal")
        else:
            self.toggle_mask_button.config(state="disabled")

    def update_inpaint_button_state(self):
        # Enable the inpaint button only if there's label data and the mask is visible
        if np.any(self.labeled_pixels_mask) and self.mask_visible:
            self.inpaint_button.config(state="normal")
        else:
            self.inpaint_button.config(state="disabled")

    def update_generate_button_state(self):
        # Disable the generate button if there's label data visible
        if np.any(self.labeled_pixels_mask) and self.mask_visible:
            self.generate_button.config(state="disabled")
        else:
            self.generate_button.config(state="normal")

    def update_undo_button_state(self):
        if self.mask_cleared and self.mask_states:
            self.undo_button.config(state="normal")
        elif self.mask_states:
            self.undo_button.config(state="normal", text="Undo")
        else:
            self.undo_button.config(state="disabled")

    def undo_mask(self):
        if self.undo_button['text'] == "Undo":
            if len(self.mask_states) > 1:
                # Remove the last state
                self.mask_states.pop()
                # Restore the previous state
                self.labeled_pixels_mask = self.mask_states[-1].copy()
            elif len(self.mask_states) == 1:
                # Only one state left, clear the mask
                self.mask_states.pop()
                self.labeled_pixels_mask = np.zeros((self.canvas_height, self.canvas_width), dtype=bool)
            self.redraw_overlay()
            self.update_toggle_button_state()
            self.update_inpaint_button_state()
            self.update_generate_button_state()
            self.update_undo_button_state()
        elif self.undo_button['text'] == "Restore mask":
            if self.mask_states:
                # Restore the last mask state
                self.labeled_pixels_mask = self.mask_states[-1].copy()
                self.redraw_overlay()
                self.mask_cleared = False
                self.undo_button.config(text="Undo")
                self.update_toggle_button_state()
                self.update_inpaint_button_state()
                self.update_generate_button_state()
                self.update_undo_button_state()

    def inpaint_image(self):
        # Get prompt text from the textbox
        prompt = self.prompt_textbox.get("1.0", "end-1c").strip()
        selected_model = self.selected_model.get()  # Get the selected model

        if not prompt:
            prompt = "Inpaint the masked areas"

        # Create a copy of the image with the alpha channel set to zero where the mask is painted
        inpaint_mask = self.image.copy()
        inpaint_mask_data = np.array(inpaint_mask)

        # Ensure labeled_pixels_mask matches image dimensions
        if inpaint_mask_data.shape[:2] != self.labeled_pixels_mask.shape:
            self.labeled_pixels_mask = np.zeros(inpaint_mask_data.shape[:2], dtype=bool)

        try:
            inpaint_mask_data[self.labeled_pixels_mask, 3] = 0  # Set alpha channel to 0 where the mask is painted
        except IndexError as e:
            print(f"Error creating inpaint mask: {e}")
            return

        inpaint_mask = Image.fromarray(inpaint_mask_data, "RGBA")
        inpaint_mask_path = "inpaint_mask.png"
        inpaint_mask.save(inpaint_mask_path)

        # Call the inpaint function from make_picture.py
        output_path = "inpainted_image.png"
        make_picture.inpaint_image(prompt=prompt, image_path=inpaint_mask_path, output=output_path, model_name=selected_model)

        # Check if inpainting output was created (i.e., model supports inpainting)
        if os.path.exists(output_path):
            # Load the inpainted image
            self.image = Image.open(output_path).convert("RGBA")
            self.photo = ImageTk.PhotoImage(self.image)

            # Display the inpainted image on the canvas by updating the image item
            self.canvas.itemconfig(self.canvas_image_id, image=self.photo)

            # Clean up
            os.remove(output_path)
            print(f"Inpainted image saved and loaded successfully.")
        else:
            print(f"Inpainting not supported for the selected model '{selected_model}'. Displaying the original image.")

        # Delete the temporary inpaint mask file
        os.remove(inpaint_mask_path)

        # Reset the labeled pixels mask since the image is now inpainted
        self.labeled_pixels_mask = np.zeros((self.canvas_height, self.canvas_width), dtype=bool)
        # Keep only the last state
        if self.mask_states:
            self.mask_states = [self.mask_states[-1]]
        else:
            self.mask_states = []
        self.mask_cleared = True
        self.undo_button.config(text="Restore mask")
        self.update_undo_button_state()

        self.overlay = Image.new("RGBA", (self.canvas_width, self.canvas_height))
        self.overlay_tk = ImageTk.PhotoImage(self.overlay)
        self.canvas.itemconfig(self.overlay_canvas, image=self.overlay_tk)

        # Update button states
        self.update_toggle_button_state()
        self.update_inpaint_button_state()
        self.update_generate_button_state()
        # Enable Reload button after inpainting
        self.reload_button.config(state="normal")
        # Enable Save button
        self.save_button.config(state="normal")

    def open_image(self):
        file_path = filedialog.askopenfilename(
            title="Open Image",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
        )
        if file_path:
            # Load the image
            self.image = Image.open(file_path).convert("RGBA")
            # Resize to 1024x1024
            self.image = self.image.resize((self.canvas_width, self.canvas_height), Image.LANCZOS)
            self.photo = ImageTk.PhotoImage(self.image)
            
            # Display the image on the canvas
            self.canvas.itemconfig(self.canvas_image_id, image=self.photo)
            
            # Reset the overlay and labeled pixels mask
            self.labeled_pixels_mask = np.zeros((self.canvas_height, self.canvas_width), dtype=bool)
            self.overlay = Image.new("RGBA", (self.canvas_width, self.canvas_height))
            self.overlay_tk = ImageTk.PhotoImage(self.overlay)
            self.canvas.itemconfig(self.overlay_canvas, image=self.overlay_tk)
            
            # Update base image path
            self.base_image_path = file_path
            # Disable Reload button since we just opened a new image
            self.reload_button.config(state="disabled")
            # Enable Save button
            self.save_button.config(state="normal")
            
            # Update button states
            self.update_toggle_button_state()
            self.update_inpaint_button_state()
            self.update_generate_button_state()
            self.update_undo_button_state()

    def save_image(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png")],
            title="Save Image"
        )
        if file_path:
            self.image.save(file_path)
            # Update base image path
            self.base_image_path = file_path
            # Enable Reload button
            self.reload_button.config(state="normal")
            # Enable Save button (in case it was disabled)
            self.save_button.config(state="normal")

    def reload_image(self):
        if self.base_image_path and os.path.exists(self.base_image_path):
            # Load the base image
            self.image = Image.open(self.base_image_path).convert("RGBA")
            self.photo = ImageTk.PhotoImage(self.image)
            
            # Display the image on the canvas
            self.canvas.itemconfig(self.canvas_image_id, image=self.photo)
            
            # Reset the overlay and labeled pixels mask
            self.labeled_pixels_mask = np.zeros((self.canvas_height, self.canvas_width), dtype=bool)
            self.overlay = Image.new("RGBA", (self.canvas_width, self.canvas_height))
            self.overlay_tk = ImageTk.PhotoImage(self.overlay)
            self.canvas.itemconfig(self.overlay_canvas, image=self.overlay_tk)
            
            # Update button states
            self.update_toggle_button_state()
            self.update_inpaint_button_state()
            self.update_generate_button_state()
            self.update_undo_button_state()
            
            # Disable reload button since we have reloaded the base image
            self.reload_button.config(state="disabled")
        else:
            print("No image to reload.")

# Set up main window and run
root = tk.Tk()
app = BrushLabelingApp(root, brush_radius=20)
root.mainloop()
