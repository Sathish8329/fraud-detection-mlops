from kserve import Model, ModelServer
from kserve.protocol.infer_type import InferRequest
import logging

logging.basicConfig(level=logging.INFO)


class FraudTransformer(Model):

    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = True

    async def preprocess(self, payload: InferRequest, headers=None):
        # Extract raw data from the InferRequest object
        infer_input = payload.inputs[0]
        data = list(infer_input.data)

        country_map = {
            "India": "IN",
            "USA": "US",
            "United States": "US",
            "UK": "UK",
        }

        # Normalize country field (index 1)
        data[1] = country_map.get(data[1], data[1])

        # Return KServe v1 dict format that the predictor's /v1/models/fraud-model:predict expects
        return {
            "instances": [
                {
                    "amount": data[0],
                    "country": data[1],
                    "hour": data[2],
                    "previous_txns": data[3],
                }
            ]
        }


model = FraudTransformer("fraud-model")

if __name__ == "__main__":
    ModelServer().start([model])