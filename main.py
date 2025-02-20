from typing import Union
from fastapi import FastAPI

# model.py를 가져온다
import model
import tmodel

# 그 안에 있는 AndModel 클래스의 인스턴스를 생성한다
andmodel = model.AndModel()
ormodel = model.ORModel()
notmodel = model.NotModel()
xormodel = tmodel.Xormodel()

app = FastAPI()

@app.get("/")
def read_root():
    # return ["Hello", "World"]
    return ("Hello", "World")

# /items/{item_id} 경로
# item_id 경로 매개변수(파라메터)
@app.get("/items/{item_id}") #endpoint 엔드포인트
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.get("/and/left/{left}/right/{right}") 
def predict_and(left: int, right: int):
    result = andmodel.predict([left, right])
    return {"result": result}

@app.post("/atrain")
def train_and():
    andmodel.train()
    return {"result": "OK"}

@app.get("/or/left/{left}/right/{right}") 
def predict_or(left: int, right: int):
    result = ormodel.predict([left, right])
    return {"result": result}

@app.post("/otrain")
def train_or():
    ormodel.train()
    return {"result": "OK"}

@app.get("/not/input/{input}") 
def predict_not(input : int):
    result = notmodel.predict(input)
    return {"result": result}

@app.post("/ntrain")
def train_not():
    notmodel.train()
    return {"result": "OK"}

@app.get("/xor/left/{left}/right/{right}") 
def predict_xor(left: int, right: int):
    result = xormodel.predict([left, right])
    return {"result": result}

@app.post("/xtrain")
def train_not():
    xormodel.train()
    return {"result": "OK"}