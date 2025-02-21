import torch
import torch.nn as nn
import torch.optim as optim

class Xormodel(nn.Module):
    def __init__(self):
        super(Xormodel, self).__init__()
        self.hidden1 = nn.Linear(2,4)
        self.hidden2 = nn.Linear(4,8)
        self.output = nn.Linear(8,1)
        self.activation = nn.ReLU()
        self.final_activation = nn.Sigmoid()
    
    def forward(self, x):
        x = self.activation(self.hidden1(x))  # 은닉층 활성화
        x = self.activation(self.hidden2(x))  # 은닉층 활성화
        x = self.final_activation(self.output(x))  # 출력층 활성화
        return x
    
    def train(self):
        X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
        y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
        criterion = nn.BCELoss()  # Binary Cross Entropy Loss
        optimizer = optim.SGD(self.parameters(), lr=0.1)  # 확률적 경사 하강법(SGD)

        # 모델 학습
        epochs = 500
        for epoch in range(epochs):
            optimizer.zero_grad()  # 기울기 초기화
            output = self(X)  # 순전파
            loss = criterion(output, y)  # 손실 계산
            loss.backward()  # 역전파
            optimizer.step()  # 가중치 업데이트

    def predict(self, input):
        with torch.no_grad():
            input = torch.tensor(input, dtype=torch.float32)
            predictions = self(input).round()
            return int(predictions.squeeze().tolist())
        
    def save(self):
        torch.save(self.state_dict(), "xormodel.pth")

    def load(self):
        self.load_state_dict(torch.load("xormodel.pth"))