import pygame
import os

class AlertSystem:
    def __init__(self, method="speaker", sound_path=""):
        self.method = method
        self.is_alerting = False
        
        # Nếu phương thức là loa (laptop hiện tại)
        if self.method == "speaker":
            pygame.mixer.init()
            self.sound_loaded = False
            if os.path.exists(sound_path):
                pygame.mixer.music.load(sound_path)
                self.sound_loaded = True
            else:
                print(f"[CANH BAO] Khong tim thay file am thanh tai: {sound_path}")
                
        # Sau này bạn có thể thêm logic cho phương thức khác tại đây:
        # elif self.method == "gpio_vibration":
        #     import RPi.GPIO as GPIO
        #     ...

    def trigger(self):
        if not self.is_alerting:
            if self.method == "speaker" and self.sound_loaded:
                pygame.mixer.music.play(-1)
            # elif self.method == "gpio_vibration":
            #     Kích hoạt chân GPIO làm rung ghế
            
            self.is_alerting = True

    def stop(self):
        if self.is_alerting:
            if self.method == "speaker" and self.sound_loaded:
                pygame.mixer.music.stop()
            # elif self.method == "gpio_vibration":
            #     Tắt chân GPIO
            
            self.is_alerting = False