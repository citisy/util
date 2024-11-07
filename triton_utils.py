import json
import re
from typing import Optional, Any

import numpy as np
import requests
import tritonclient  # pip install tritonclient[all]
import tritonclient.grpc
import tritonclient.http

from . import converter, log_utils

# https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#datatypes
datatypes = {
    'Config': ['TYPE_BOOL', 'TYPE_UINT8', 'TYPE_UINT16', 'TYPE_UINT32', 'TYPE_UINT64', 'TYPE_INT8', 'TYPE_INT16', 'TYPE_INT32', 'TYPE_INT64', 'TYPE_FP16', 'TYPE_FP32', 'TYPE_FP64', 'TYPE_STRING', 'TYPE_BF16'],
    'API': ['BOOL', 'UINT8', 'UINT16', 'UINT32', 'UINT64', 'INT8', 'INT16', 'INT32', 'INT64', 'FP16', 'FP32', 'FP64', 'BYTES', 'BF16'],
    'TensorRT': ['kBOOL', 'kUINT8', '', '', '', 'kINT8', '', 'kINT32', '', 'kHALF', 'kFLOAT', '', '', ''],
    'TensorFlow': ['DT_BOOL', 'DT_UINT8', 'DT_UINT16', 'DT_UINT32', 'DT_UINT64', 'DT_INT8', 'DT_INT16', 'DT_INT32', 'DT_INT64', 'DT_HALF', 'DT_FLOAT', 'DT_DOUBLE', 'DT_STRING', ''],
    'ONNX': ['BOOL', 'UINT8', 'UINT16', 'UINT32', 'UINT64', 'INT8', 'INT16', 'INT32', 'INT64', 'FLOAT16', 'FLOAT', 'DOUBLE', 'STRING', ''],
    'PyTorch': ['kBool', 'kByte', '', '', '', 'kChar', 'kShort', 'kInt', 'kLong', '', 'kFloat', 'kDouble', '', ''],
    'NumPy': ['BOOL', 'uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64', 'BYTES', '']
}


class BaseClient:
    def __init__(
            self,
            client: Optional[tritonclient.http.InferenceServerClient | tritonclient.grpc.InferenceServerClient],
            verbose=False, logger=None, **kwargs
    ):
        self.client = client
        self.verbose = verbose
        self.logger = log_utils.get_logger(logger)
        self.model_configs = {}
        self.model_versions = {}

    def init(self):
        raise NotImplementedError()

    def _init(self, model_info):
        for info in model_info:
            name = info['name']
            if 'version' not in info:
                continue
            version = info['version']
            state = info['state']

            if state != 'READY':
                self.logger.warning(f'{name}:{version}({state}) is not ready, pls check!')
                continue

            self.model_versions[name] = version
            if isinstance(self.client, tritonclient.http.InferenceServerClient):
                self.model_configs[name, version] = self.client.get_model_config(name, version)
            else:
                self.model_configs[name, version] = self.client.get_model_config(name, version, as_json=True)['config']

        if self.verbose:
            self.logger.info(self.model_configs)

    def load(self, model_name):
        """must work on explicit model
        run tritonserver with `--model-control-mode explicit`"""
        self.client.load_model(model_name)
        self.init()

    def unload(self, model_name):
        self.client.unload_model(model_name)
        self.init()

    def _get_model_configs(self, model_name, model_version=None):
        if not self.model_versions:
            # note, in some version of triton, it will be got some unknown exceptions when init early
            # so init when using
            self.init()

        model_version = model_version or self.model_versions.get(model_name)
        assert (model_name, model_version) in self.model_configs, \
            f'Got {model_name = } and {model_version = }, where the keys is {self.model_configs.keys()}, pls check'

        model_config = self.model_configs[model_name, model_version]

        return model_version, model_config

    def _parse_input(self, i, cfg):
        dtype = cfg['data_type']
        if dtype == 'TYPE_STRING':
            if isinstance(i, list):
                i = np.array([_.encode('utf8') for _ in i], dtype=np.object_)
            else:
                i = np.array([i.encode('utf8')], dtype=np.object_)

        elif dtype == 'TYPE_BOOL':
            if isinstance(i, list):
                i = np.array(i)
            else:
                i = np.array([i])

        dtype = datatypes['API'][datatypes['Config'].index(dtype)]
        return i, dtype

    def _parse_output(self, result):
        if isinstance(result, tritonclient.http.InferResult):
            response = result.get_response()
        else:
            response = result.get_response(as_json=True)

        outputs = {}
        for output in response['outputs']:
            name = output['name']
            datatype = output['datatype']
            o = result.as_numpy(name)
            if datatype == 'BYTES':
                o = o[0].decode('utf-8')
            outputs[name] = o

        return outputs


class HttpClient(BaseClient):
    """
    note, InferenceServerClient intended to be used by a single thread,
    if using in a multi-thread like flask app, usually got errors like
    `greenlet.error: cannot switch to a different thread (which happens to have exited)`
    so using the following scripts to initialize the client to avoid the errors:
        ```
        class App:
            @property
            def trt_client():
                return HttpClient()
        ```
    instead of:
        ```
        class App:
            def __init__():
                self.trt_client = HttpClient()
        ```

    """

    def __init__(self, url='127.0.0.1:8000', verbose=False, **kwargs):
        self.url = url

        client = tritonclient.http.InferenceServerClient(url=url, verbose=False, **kwargs)
        super().__init__(client, verbose=verbose, **kwargs)

    def init(self):
        model_info = self.client.get_model_repository_index()
        self._init(model_info)

    def async_infer(self, *inputs: Optional['np.ndarray'], model_name, model_version=None):
        model_version, model_config = self._get_model_configs(model_name, model_version)

        _inputs = []
        for cfg, i in zip(model_config['input'], inputs):
            i, dtype = self._parse_input(i, cfg)

            _input = tritonclient.http.InferInput(cfg['name'], i.shape, dtype)
            _input.set_data_from_numpy(i)
            _inputs.append(_input)

        _outputs = []
        for cfg in model_config['output']:
            _output = tritonclient.http.InferRequestedOutput(cfg['name'])
            _outputs.append(_output)

        async_req = self.client.async_infer(
            model_name=model_name,
            model_version=model_version,
            inputs=_inputs,
            outputs=_outputs
        )

        return async_req

    def async_get(self, async_req: tritonclient.http.InferAsyncRequest):
        result = async_req.get_result()
        outputs = self._parse_output(result)

        return outputs

    def generate(self, *inputs: Any, model_name, model_version=None):
        """official http triton client do not support generate endpoint called
        see https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2450/user-guide/docs/protocol/extension_generate.html"""
        model_version, model_config = self._get_model_configs(model_name, model_version)

        if model_version:
            model_version = f'/versions/{model_version}'
        else:
            model_version = ''

        url = f'http://{self.url}/v2/models/{model_name}{model_version}/generate_stream'

        _inputs = {}
        for cfg, i in zip(model_config['input'], inputs):
            _inputs[cfg['name']] = converter.DataConvert.custom_to_constant(i)

        _inputs = json.dumps(_inputs, ensure_ascii=False)
        r = requests.post(url, data=_inputs, stream=True)

        for chunk in r.iter_lines():
            if chunk:
                data = chunk.decode('utf8')
                r = re.search(r'^data: (.*)$', data)
                if r:
                    data = r.group(1)
                    data = json.loads(data)
                    yield data


class GrpcClient(BaseClient):
    def __init__(self, url='127.0.0.1:8001', verbose=False, **kwargs):
        self.url = url

        client = tritonclient.grpc.InferenceServerClient(url=url, verbose=False, **kwargs)
        super().__init__(client, verbose=verbose, **kwargs)

    def init(self):
        model_info = self.client.get_model_repository_index(as_json=True)['models']
        self._init(model_info)

    def generate(self, *inputs: Any, model_name, model_version=None, callback=None):
        model_version, model_config = self._get_model_configs(model_name, model_version)

        _inputs = []
        for cfg, i in zip(model_config['input'], inputs):
            i, dtype = self._parse_input(i, cfg)

            _input = tritonclient.grpc.InferInput(cfg['name'], i.shape, dtype)
            _input.set_data_from_numpy(i)
            _inputs.append(_input)

        _outputs = []
        for cfg in model_config['output']:
            _output = tritonclient.grpc.InferRequestedOutput(cfg['name'])
            _outputs.append(_output)

        total_outputs = []

        def _callback(result: tritonclient.grpc.InferResult, error):
            outputs = self._parse_output(result)
            total_outputs.append(outputs)

        self.client.start_stream(callback or _callback)
        self.client.async_stream_infer(
            model_name=model_name,
            model_version=model_version,
            inputs=_inputs,
            outputs=_outputs
        )

        self.client.stop_stream()
        return total_outputs


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    this is an example
    refer to: https://github.com/triton-inference-server/python_backend
    """

    def initialize(self, args):
        import triton_python_backend_utils as pb_utils
        import json

        # get configs
        # configs can be found in `config.pbtxt`
        self.model_config = model_config = json.loads(args['model_config'])

        self.input0_config = pb_utils.get_output_config_by_name(model_config, "INPUT0")
        self.input0_dtype = pb_utils.triton_string_to_numpy(self.input0_config['data_type'])
        self.output0_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT0")
        self.output0_dtype = pb_utils.triton_string_to_numpy(self.output0_config['data_type'])
        ...

        # init model
        self.model = ...

    def execute(self, requests):
        import triton_python_backend_utils as pb_utils

        responses = []
        for request in requests:
            # get inputs
            in_0 = pb_utils.get_input_tensor_by_name(request, 'INPUT0')
            ...

            # get outputs from model inference
            out_0, *outs = self.model(in_0, ...)

            out_tensor_0 = pb_utils.Tensor('OUTPUT0', out_0.astype(self.output0_dtype))
            ...

            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0, ...])

            responses.append(inference_response)
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        pass
