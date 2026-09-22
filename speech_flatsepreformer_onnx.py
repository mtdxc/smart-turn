# 下面的代码会读取ONNX元数据，自动判断模型使用波形输入还是STFT输入，并将分离结果保存为output_spk1.wav和output_spk2.wav。

import os
import time

import numpy as np
import onnxruntime
import soundfile as sf

ONNX_PATH = os.environ.get('ONNX_PATH', 'onnx_model.onnx')
AUDIO_PATH = os.environ.get('AUDIO_PATH', 'mix_speech1.wav')

mixture, sample_rate = sf.read(AUDIO_PATH, dtype='float32')
if sample_rate != 8000:
    raise ValueError(f'音频采样率必须为8000 Hz，当前为{sample_rate} Hz')
if mixture.ndim != 1:
    raise ValueError(f'音频必须为单声道，当前shape为{mixture.shape}')

session = onnxruntime.InferenceSession(
    ONNX_PATH, providers=['CPUExecutionProvider'])
metadata = session.get_modelmeta().custom_metadata_map

if metadata.get('input_domain') == 'waveform':
    start_time = time.perf_counter()
    sources = session.run(
        None, {'mixture': mixture[None].astype(np.float32)})[0]
    elapsed_seconds = time.perf_counter() - start_time
    sources = sources[0].transpose(1, 0)
    audio_seconds = len(mixture) / sample_rate
    print(f'转换倍率: {audio_seconds / elapsed_seconds:.2f}x')
elif metadata.get('input_domain') == 'stft':
    print('STFT输入尚未实现')
    # mixture_tensor = torch.from_numpy(mixture).unsqueeze(0)
    # mixture_std = torch.std(mixture_tensor, dim=1, keepdim=True)
    # if mixture_std.item() == 0:
    #     raise ValueError('输入音频不能是静音')
    # normalized = mixture_tensor / mixture_std
    # window = torch.hann_window(128, dtype=normalized.dtype)
    # spectrogram = torch.stft(
    #     normalized,
    #     n_fft=128,
    #     hop_length=64,
    #     win_length=128,
    #     window=window,
    #     center=True,
    #     normalized=False,
    #     onesided=True,
    #     return_complex=True,
    # ).transpose(1, 2)
    # sources_real, sources_imag = session.run(
    #     None,
    #     {
    #         'mixture_real': spectrogram.real.numpy(),
    #         'mixture_imag': spectrogram.imag.numpy(),
    #     },
    # )
    # source_specs = torch.complex(
    #     torch.from_numpy(sources_real), torch.from_numpy(sources_imag))
    # waves = [
    #     torch.istft(
    #         source_specs[:, speaker].transpose(1, 2),
    #         n_fft=128,
    #         hop_length=64,
    #         win_length=128,
    #         window=window,
    #         center=True,
    #         normalized=False,
    #         onesided=True,
    #         length=mixture.shape[0],
    #         return_complex=False,
    #     ) * mixture_std
    #     for speaker in range(source_specs.shape[1])
    # ]
    # sources = torch.cat(waves, dim=0).numpy()
else:
    raise ValueError('ONNX模型缺少有效的input_domain元数据')

for speaker, source in enumerate(sources, start=1):
    peak = np.max(np.abs(source))
    output = source if peak == 0 else source * (0.5 / peak)
    output_path = f'output_spk{speaker}.wav'
    sf.write(output_path, output.astype(np.float32), sample_rate)
    print(f'已保存：{output_path}')