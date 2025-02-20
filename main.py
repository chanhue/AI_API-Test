from typing import Union
from fastapi import FastAPI

app = FastAPI() #클래스 생성자

@app.get("/")
def read_root():
    return {"Hello": "World"} #딕셔너리로 반환하면 JSON으로 응답한다.
    #return "hello world"

# 경로 "/items/{item_id}" -> 경로에 접근하여 사용 가능
# item_id 경로 매개변수(파라미터)
@app.get("/items/{item_id}") #endpoint 엔드포인트
def read_item(item_id: int, q: Union[str, None] = None): #기존의 파이썬과는 다르게 타입을 설정함(Type Hinting) - 검증에서 장점이 있다.
    return {"item_id": item_id, "q": q} 

