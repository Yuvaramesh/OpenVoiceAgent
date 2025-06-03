import time
from voice import VoiceService
import nltk

# Download NLTK resources
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

# Create VoiceService instance
vs = VoiceService()

# Set reference speaker audio path
reference_speaker_path = "modules/OpenVoice/resources/yuva.mp3"

# Call openvoice with text and reference path
vs.openvoice(
    "Accessing alarm and interface settings. Hello Sir, What can I do for you today?",
    reference_speaker_path
)
