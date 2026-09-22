import inspect
import os

import onnx
import torch
import torch.nn as nn
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

MODEL_ID = os.environ.get('MODEL_ID',
    'iic/speech_flatsepreformer_separation_temporal_8k_base_libri2mix100')
ONNX_PATH = os.environ.get('ONNX_PATH', 'onnx_model.onnx')

class TFLocoformerOnnxWrapper(nn.Module):
    """将复数频谱拆成ONNX支持的实部和虚部张量。"""

    def __init__(self, separator):
        super().__init__()
        self.separator = separator

    def forward(self, mixture_real, mixture_imag):
        batch = torch.stack((mixture_real, mixture_imag), dim=1)
        batch_size, _, frames, freqs = batch.shape

        with torch.autocast(device_type=batch.device.type, enabled=False):
            batch = self.separator.conv(batch)

        frequency_index = 0
        time_index = 0
        for layer_type in self.separator.layers_type:
            if layer_type == 'f':
                batch = self.separator.f_blocks[frequency_index](batch)
                frequency_index += 1
            else:
                batch = self.separator.t_blocks[time_index](batch)
                time_index += 1

        with torch.autocast(device_type=batch.device.type, enabled=False):
            batch = self.separator.deconv(batch)

        batch = batch.reshape(
            batch_size, self.separator.num_spk, 2, frames, freqs)
        return batch[:, :, 0], batch[:, :, 1]


separation = pipeline(
    Tasks.speech_separation, model=MODEL_ID, device='cpu')
model = separation.model.eval()

if 'flatflocoformer' in str(MODEL_ID):
    export_model = TFLocoformerOnnxWrapper(model.separator).eval()
    example_inputs = (
        torch.randn(1, 126, 65),
        torch.randn(1, 126, 65),
    )
    input_names = ['mixture_real', 'mixture_imag']
    output_names = ['sources_real', 'sources_imag']
    dynamic_axes = {
        'mixture_real': {0: 'batch', 1: 'frames'},
        'mixture_imag': {0: 'batch', 1: 'frames'},
        'sources_real': {0: 'batch', 2: 'frames'},
        'sources_imag': {0: 'batch', 2: 'frames'},
    }
    metadata = {
        'input_domain': 'stft',
        'n_fft': '128',
        'hop_length': '64',
    }
elif 'flatsepreformer' in str(MODEL_ID):
    export_model = model
    example_inputs = (torch.randn(1, 8000), )
    input_names = ['mixture']
    output_names = ['sources']
    dynamic_axes = {
        'mixture': {0: 'batch', 1: 'samples'},
        'sources': {0: 'batch', 1: 'samples'},
    }
    metadata = {'input_domain': 'waveform'}
else:
    raise ValueError(f'不支持导出该模型：{MODEL_ID}')

export_options = {}
if 'dynamo' in inspect.signature(torch.onnx.export).parameters:
    export_options['dynamo'] = False

torch.onnx.export(
    export_model,
    example_inputs,
    ONNX_PATH,
    opset_version=17,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes=dynamic_axes,
    **export_options,
)

onnx_model = onnx.load(ONNX_PATH)
metadata.update({'sample_rate': '8000', 'num_speakers': '2'})
for key, value in metadata.items():
    item = onnx_model.metadata_props.add()
    item.key = key
    item.value = value
onnx.checker.check_model(onnx_model)
onnx.save(onnx_model, ONNX_PATH)
print(f'ONNX模型已导出到：{ONNX_PATH}')