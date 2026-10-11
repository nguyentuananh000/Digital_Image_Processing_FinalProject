import os
import time
from threading import Thread
import numpy as np
import pygame


class AlertSystem:
    """
    Hệ thống cảnh báo âm thanh đa cấp độ (Multi-level Escalation Alert):
    - Cấp độ 0 (SAFE): Tắt báo động.
    - Cấp độ 1 (WARNING - Nhắc nhở sớm): Phát âm báo nhắc nhở ngắn (Beep kép) định kỳ mỗi 4 giây
      khi tài xế ngáp nhiều, chợp mắt ngắn (Micro-sleep), hoặc quay đầu mất tập trung.
    - Cấp độ 2 (DANGER - Báo động khẩn cấp): Phát còi hú liên tục (alarm.wav) khi ngủ gật,
      gục đầu hoặc chỉ số mệt mỏi PERCLOS vượt ngưỡng nguy hiểm.
    """

    def __init__(self, method="speaker", sound_path="", warning_cooldown=4.0):
        self.method = method
        self.is_alerting = False
        self.current_level = 0
        self.sound_loaded = False
        self.warning_sound = None
        self.warning_cooldown = float(warning_cooldown)
        self._last_warning_time = 0.0

        if self.method == "speaker":
            try:
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
                if sound_path and os.path.exists(sound_path):
                    pygame.mixer.music.load(sound_path)
                    self.sound_loaded = True
                else:
                    print(f"[CANH BAO] Khong tim thay file am thanh tai: {sound_path}")

                # Tạo âm thanh nhắc nhở (Warning Tone 880Hz ngắn) bằng tín hiệu sóng sin
                self.warning_sound = self._synthesize_warning_beep()
            except Exception as e:
                print(f"[CANH BAO] Khong the khoi tao he thong am thanh pygame: {e}")

    @staticmethod
    def _synthesize_warning_beep(freq: int = 880, duration: float = 0.22, sample_rate: int = 44100):
        """Tạo âm báo nhắc nhở nhẹ nhàng (Cấp độ 1) không cần file âm thanh ngoài."""
        try:
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            wave = (np.sin(2 * np.pi * freq * t) * 16000).astype(np.int16)
            # Làm mượt 2 đầu (fade in/out) để không bị tiếng bụp loa
            fade_len = min(400, len(wave) // 4)
            wave[:fade_len] = (wave[:fade_len] * np.linspace(0, 1, fade_len)).astype(np.int16)
            wave[-fade_len:] = (wave[-fade_len:] * np.linspace(1, 0, fade_len)).astype(np.int16)
            stereo_wave = np.column_stack((wave, wave))
            return pygame.sndarray.make_sound(stereo_wave)
        except Exception:
            return None

    def update_level(self, level: int):
        """
        Cập nhật mức cảnh báo:
        - level = 0: An toàn (dừng còi)
        - level = 1: Cảnh báo sớm (phát tiếng nhắc nhở định kỳ)
        - level = 2: Báo động khẩn cấp (hú còi liên tục)
        """
        if level >= 2:
            self.trigger()
            self.current_level = 2
        elif level == 1:
            if self.current_level == 2:
                self.stop()
            self.current_level = 1
            now = time.time()
            if now - self._last_warning_time >= self.warning_cooldown:
                self._last_warning_time = now
                self._play_warning_nonblocking()
        else:
            self.stop()
            self.current_level = 0

    def _play_warning_nonblocking(self):
        if self.method == "speaker":
            if self.warning_sound is not None:
                try:
                    self.warning_sound.play()
                    return
                except Exception:
                    pass
            # Fallback cho Windows nếu sndarray không khả dụng
            def _beep():
                try:
                    import winsound
                    winsound.Beep(880, 200)
                except Exception:
                    pass
            Thread(target=_beep, daemon=True).start()

    def trigger(self):
        """Kích hoạt báo động khẩn cấp (Cấp độ 2)."""
        if not self.is_alerting:
            if self.method == "speaker" and self.sound_loaded:
                try:
                    pygame.mixer.music.play(-1)
                except Exception as e:
                    print(f"[LOI AM THANH] {e}")
            self.is_alerting = True

    def stop(self):
        """Dừng báo động khẩn cấp."""
        if self.is_alerting:
            if self.method == "speaker" and self.sound_loaded:
                try:
                    pygame.mixer.music.stop()
                except Exception:
                    pass
            self.is_alerting = False
