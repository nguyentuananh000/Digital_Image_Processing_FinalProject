import pygame
import os

class AudioManager:
    def __init__(self, sound_path):
        pygame.mixer.init()
        self.is_playing = False
        self.sound_loaded = False
        
        if os.path.exists(sound_path):
            pygame.mixer.music.load(sound_path)
            self.sound_loaded = True
        else:
            print(f"[CANH BAO] Khong tim thay file am thanh tai: {sound_path}")

    def play(self):
        if self.sound_loaded and not self.is_playing:
            pygame.mixer.music.play(-1)
            self.is_playing = True

    def stop(self):
        if self.sound_loaded and self.is_playing:
            pygame.mixer.music.stop()
            self.is_playing = False