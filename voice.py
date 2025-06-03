import os
import pygame
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS

class VoiceService:
    def __init__(self):
        self._ckpt_converter = 'modules/OpenVoice/checkpoints_v2/converter'
        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._output_dir = 'outputs_v2'

        self._tone_color_converter = ToneColorConverter(f'{self._ckpt_converter}/config.json', device=self._device)
        self._tone_color_converter.load_ckpt(f'{self._ckpt_converter}/checkpoint.pth')

        os.makedirs(self._output_dir, exist_ok=True)

    def openvoice(self, text, reference_speaker_path):
    # Reference voice (clone target)
        target_se, _ = se_extractor.get_se(reference_speaker_path, self._tone_color_converter, vad=True)

    # TTS model and fixed speaker_id
        model = TTS(language="EN", device=self._device)
        speaker_id = 2  # en-india speaker

        # Temporary TTS output
        src_path = f'{self._output_dir}/tmp.wav'
        speed = 1.0

        # Generate source voice
        model.tts_to_file(text, speaker_id, src_path, speed=speed)

        # Source speaker embedding
        source_se = torch.load(
            'modules/OpenVoice/checkpoints_v2/base_speakers/ses/en-india.pth',
            map_location=self._device
        )

    # Convert voice
        save_path = f'{self._output_dir}/output_en-india.wav'
        self._tone_color_converter.convert(
            audio_src_path=src_path,
            src_se=source_se,
            tgt_se=target_se,
            output_path=save_path,
            message="@MyShell"
        )

        self.play(save_path)

    def play(self, temp_audio_file):
        pygame.mixer.init()
        pygame.mixer.music.load(temp_audio_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        pygame.mixer.music.stop()
        pygame.mixer.quit()
