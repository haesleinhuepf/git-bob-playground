from kivy.app import App
from kivy.uix.scatter import Scatter
from kivy.uix.image import Image
from kivy.core.window import Window
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import Line, Color
from kivy.metrics import dp
import numpy as np
import pandas as pd
from kivy.graphics.texture import Texture

class ImageScatter(Scatter):
    """Widget that handles touch interaction (drag, rotate, zoom) for a single image"""
    def __init__(self, img_array, image_id, **kwargs):
        """
        Parameters
        ----------
        img_array : ndarray
            Image data as numpy array
        image_id : int
            Unique identifier for the image
        """
        super().__init__(**kwargs)
        h, w = img_array.shape[:2]
        texture = Texture.create(size=(w, h))
        texture.blit_buffer(img_array.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        self.img = Image(texture=texture)
        self.add_widget(self.img)
        self.image_id = image_id
        
    def on_transform_with_touch(self, touch):
        """Update transform data when image is manipulated"""
        super().on_transform_with_touch(touch)
        app = App.get_running_app()
        app.transforms.loc[self.image_id] = [
            self.pos[0], self.pos[1],
            self.rotation,
            self.scale,
            app.transforms.loc[self.image_id, 'selected']
        ]

class SelectionBox:
    """Handles drawing selection rectangle"""
    def __init__(self, widget):
        self.start = None
        self.end = None
        self.widget = widget
        
    def draw(self):
        """Draw selection rectangle on canvas"""
        if self.start and self.end:
            self.widget.canvas.clear()
            with self.widget.canvas:
                Color(0, 1, 0, 0.3)
                Line(rectangle=(
                    min(self.start[0], self.end[0]),
                    min(self.start[1], self.end[1]),
                    abs(self.end[0] - self.start[0]),
                    abs(self.end[1] - self.start[1])
                ))

class TouchApp(App):
    """Main application class"""
    def build(self):
        """Initialize application UI"""
        Window.fullscreen = True
        layout = BoxLayout(orientation='vertical')
        
        # Setup toolbar
        toolbar = BoxLayout(size_hint_y=None, height=dp(50))
        for text in ['Load', 'Save', 'Delete', 'Move', 'Select']:
            btn = Button(text=text)
            btn.bind(on_press=self.on_button_press)
            toolbar.add_widget(btn)
        
        self.root = BoxLayout()
        self.mode = 'move'
        self.selection = SelectionBox(self.root)
        self.transforms = pd.DataFrame(columns=['x', 'y', 'rotation', 'scale', 'selected'])
        
        layout.add_widget(toolbar)
        layout.add_widget(self.root)
        return layout
    
    def on_button_press(self, button):
        """Handle toolbar button presses"""
        if button.text == 'Save':
            selected = self.transforms[self.transforms.selected == 1]
            df_to_save = selected if len(selected) > 0 else self.transforms
            df_to_save.to_csv('image_positions.csv')
        elif button.text == 'Load':
            self.transforms = pd.read_csv('image_positions.csv')
            self.reload_images()
        elif button.text == 'Delete':
            to_delete = self.transforms[self.transforms.selected == 1].index
            self.transforms.drop(to_delete, inplace=True)
            self.reload_images()
        else:
            self.mode = button.text.lower()

if __name__ == '__main__':
    TouchApp().run()
