import paddle
from paddlenlp.transformers import GPTModel, GPTTokenizer

class MyModel(paddle.nn.Layer):
    def __init__(self,vocab_size):
        super().__init__()
        #tokenizer = GPTTokenizer.from_pretrained('gpt2-medium-en')
        self.encoder = GPTModel.from_pretrained('gpt2-medium-en')
        self.linear = paddle.nn.Linear(1024,vocab_size)
    def forward(self,x):

        x = self.encoder(input_ids=x)
        x = self.linear(x)
        return x
