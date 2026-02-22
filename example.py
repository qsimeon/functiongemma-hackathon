
import sys, os
sys.path.insert(0, "cactus/python/src")
os.environ["CACTUS_NO_CLOUD_TELE"] = "1"

import json
from cactus import cactus_init, cactus_complete, cactus_destroy

# model = cactus_init("cactus/weights/lfm2-vl-450m")
model = cactus_init("cactus/weights/functiongemma-270m-it")
messages = [{"role": "user", "content": "What is 2+2?"}]
response = json.loads(cactus_complete(model, messages))
print(response["response"])

cactus_destroy(model)