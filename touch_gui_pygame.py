import pygame
import numpy as np
import pandas as pd
from pathlib import Path

class TouchGUI:
    """Main GUI class for handling touch-based image manipulation"""
    
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self.width, self.height = self.screen.get_size()
        
        # Initialize data storage
        self.images = {}  # id -> surface mapping
        self.transforms = pd.DataFrame(columns=['x', 'y', 'rotation', 'scale', 'selected'])
        
        # GUI state
        self.mode = 'move'
        self.selection_start = None
        self.active_touches = {}
        self.buttons = self._create_buttons()
        
    def _create_buttons(self):
        """Create toolbar buttons"""
        buttons = {}
        texts = ['Load', 'Save', 'Delete', 'Move', 'Select']
        x = 10
        for text in texts:
            rect = pygame.Rect(x, 10, 100, 40)
            buttons[text] = rect
            x += 120
        return buttons
    
    def add_image(self, img_array, image_id):
        """
        Add new image to GUI
        
        Parameters
        ----------
        img_array : ndarray
            Image as numpy array
        image_id : int
            Unique identifier for image
        """
        surface = pygame.surfarray.make_surface(img_array)
        self.images[image_id] = surface
        self.transforms.loc[image_id] = [
            self.width//2, self.height//2,  # x, y
            0, 1, 0  # rotation, scale, selected
        ]

    def handle_touch(self, event):
        """
        Process touch events for image manipulation
        
        Parameters
        ----------
        event : pygame.event
            Touch event to process
        """
        if event.type == pygame.FINGERDOWN:
            pos = (event.x * self.width, event.y * self.height)
            
            # Check button clicks
            for text, rect in self.buttons.items():
                if rect.collidepoint(pos):
                    self._handle_button(text)
                    return
                    
            if self.mode == 'select':
                self.selection_start = pos
            elif self.mode == 'move':
                self.active_touches[event.finger_id] = pos
                
        elif event.type == pygame.FINGERUP:
            self.active_touches.pop(event.finger_id, None)
            if self.mode == 'select':
                self._finalize_selection()
                
        elif event.type == pygame.FINGERMOTION and self.mode == 'move':
            pos = (event.x * self.width, event.y * self.height)
            if len(self.active_touches) == 1:  # Move
                dx = pos[0] - self.active_touches[event.finger_id][0]
                dy = pos[1] - self.active_touches[event.finger_id][1]
                self._move_selected(dx, dy)
            elif len(self.active_touches) == 2:  # Scale/Rotate
                self._handle_transform(event)
            self.active_touches[event.finger_id] = pos

    def _handle_button(self, text):
        """Handle toolbar button presses"""
        if text in ['Move', 'Select']:
            self.mode = text.lower()
        elif text == 'Save':
            self._save_transforms()
        elif text == 'Load':
            self._load_transforms()
        elif text == 'Delete':
            self._delete_selected()

    def _save_transforms(self):
        """Save transformation data to CSV"""
        self.transforms.to_csv('image_positions.csv')

    def _load_transforms(self):
        """Load transformation data from CSV"""
        if Path('image_positions.csv').exists():
            self.transforms = pd.read_csv('image_positions.csv', index_col=0)

    def _delete_selected(self):
        """Delete selected images"""
        to_delete = self.transforms[self.transforms.selected == 1].index
        self.transforms.drop(to_delete, inplace=True)
        for idx in to_delete:
            self.images.pop(idx, None)

    def run(self):
        """Main event loop"""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type in [pygame.FINGERDOWN, pygame.FINGERUP, pygame.FINGERMOTION]:
                    self.handle_touch(event)
                    
            self._draw()
            pygame.display.flip()
            
        pygame.quit()

    def _draw(self):
        """Draw all GUI elements"""
        self.screen.fill((200, 200, 200))
        
        # Draw images
        for img_id, surf in self.images.items():
            transform = self.transforms.loc[img_id]
            scaled = pygame.transform.rotozoom(surf, transform.rotation, transform.scale)
            rect = scaled.get_rect(center=(transform.x, transform.y))
            self.screen.blit(scaled, rect)
            
        # Draw buttons
        for text, rect in self.buttons.items():
            color = (100, 100, 250) if self.mode == text.lower() else (150, 150, 150)
            pygame.draw.rect(self.screen, color, rect)
            font = pygame.font.Font(None, 36)
            text_surf = font.render(text, True, (0, 0, 0))
            text_rect = text_surf.get_rect(center=rect.center)
            self.screen.blit(text_surf, text_rect)

if __name__ == '__main__':
    gui = TouchGUI()
    # Add test image
    test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    gui.add_image(test_img, 0)
    gui.run()
