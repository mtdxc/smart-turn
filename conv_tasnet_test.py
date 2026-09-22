# model from https://huggingface.co/Mazino0/conv-tasnet-onnx/tree/main
import onnxruntime as ort
import numpy as np
import soundfile as sf
import time

mixture, sample_rate = sf.read('mix_speech1.wav', dtype='float32')
if sample_rate != 8000:
    raise ValueError(f'音频采样率必须为8000 Hz，当前为{sample_rate} Hz')
if mixture.ndim != 1:
    raise ValueError(f'音频必须为单声道，当前shape为{mixture.shape}')

# ONNX model input expects shape: [batch, time]. Add batch dimension.
mixture = mixture.astype(np.float32, copy=False).reshape(1, -1)

# Use int8 model (recommended)
sess = ort.InferenceSession('conv_tasnet_libri2mix_int8.onnx', providers=['CPUExecutionProvider'])
start_time = time.perf_counter()
separated = sess.run(None, {'mixture': mixture})[0]
elapsed_seconds = time.perf_counter() - start_time

# separated.shape = (1, 2, 8000) — two speaker sources
source_1 = separated[0, 0, :]
source_2 = separated[0, 1, :]

audio_seconds = mixture.shape[1] / sample_rate
print(f'转换倍率: {audio_seconds / elapsed_seconds:.2f}x')

sf.write('source_1.wav', source_1, sample_rate)
sf.write('source_2.wav', source_2, sample_rate)