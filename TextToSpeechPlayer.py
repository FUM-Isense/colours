import io
import queue
import threading
from gtts import gTTS
import numpy as np
import pygame


class TextToSpeechPlayer:
    def __init__(self, language='en'):
        self.language = language
        pygame.mixer.init()
        self.text_queue = queue.Queue()
        self.stop_flag = threading.Event()
        self.speech_th = threading.Thread(target=self.__speak)
        self.speech_th.start()

    def __speak(self):
        stop_called = False
        
        while True and not stop_called:
            if not self.text_queue.empty():
                next_text = self.text_queue.get()
                tts = gTTS(text=next_text, lang=self.language, slow=False)
                mp3_fp = io.BytesIO()
                tts.write_to_fp(mp3_fp)
                mp3_fp.seek(0)
                pygame.mixer.music.load(mp3_fp, 'mp3')
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                    
            else:
                if self.stop_flag.is_set():
                    stop_called = True

    def say(self, text):
        self.text_queue.put(text)
        
    def stop(self):
        self.stop_flag.set()
        self.speech_th.join()