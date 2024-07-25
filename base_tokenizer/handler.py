from typing import Dict, List, Any
from transformers import pipeline

class EndpointHandler():
    def __init__(self, path=""):
        self.pipeline = pipeline("text-generation",model=path, device='cuda')

    def __call__(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
       data args:
            inputs (:obj: `str`)
            date (:obj: `str`)
      Return:
            A :obj:`list` | `dict`: will be serialized and returned
        """
        # get inputs
        inputs = data.pop("inputs",data)
        config = data.pop("config", None)
        # run normal prediction
        prediction = self.pipeline(inputs, **config)
        return prediction