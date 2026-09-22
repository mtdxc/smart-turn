import numpy as np
import soundfile as sf
import time
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# input可以是URL，也可以是本地文件路径；音频必须为8 kHz单声道wav
input = 'mix_speech.wav'
model_id = 'damo/speech_flatflocoformer_separation_timefrequency_8k_middle_libri2mix360'
model_id = 'iic/speech_flatsepreformer_separation_temporal_8k_base_libri2mix100'

separation = pipeline(Tasks.speech_separation, model=model_id)
start_time = time.perf_counter()
result = separation(input)
elapsed_seconds = time.perf_counter() - start_time
signals = result[OutputKeys.OUTPUT_PCM_LIST]
audio_seconds = len(signals[0]) / np.dtype(np.int16).itemsize / 8000
print(f'转换倍率: {audio_seconds / elapsed_seconds:.2f}x')

for i, signal in enumerate(signals):
    save_file = f'output_spk{i + 1}.wav'
    sf.write(save_file, np.frombuffer(signal, dtype=np.int16), 8000)