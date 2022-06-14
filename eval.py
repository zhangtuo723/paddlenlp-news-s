from net import *
from paddle import fluid
from paddlenlp.transformers import GPTModel, GPTTokenizer
from rouge import Rouge
from paddlenlp.datasets import load_dataset
from tqdm import tqdm

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocab size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.shape[-1])  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
        # ...表示其他维度由计算机自行推断
        indices_to_remove = logits < paddle.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷

    if top_p > 0.0:
        sorted_logits, sorted_indices = paddle.sort(logits, descending=True)  # 对logits进行递减排序
        cumulative_probs = paddle.cumsum(fluid.layers.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits



def eval(news,target):

    input_id = [tokenizer.sep_token_id]
    input_id.extend(tokenizer(news)["input_ids"][:400])
    input_id.append(tokenizer.sep_token_id)
    input_id = paddle.to_tensor([input_id])
    #logits = net(paddle.to_tensor([input_id]))
    response = []
    for _ in range(max_len):
        logits = net(input_id)


        next_token_logits = logits[0, -1, :]
        # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
        #print(next_token_logits.shape)

        for id in set(response):
            next_token_logits[id] /= repetition_penalty
        next_token_logits = next_token_logits / temperature
        # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
        #next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
        filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=5, top_p=0)
        # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
        next_token = paddle.multinomial(fluid.layers.softmax(filtered_logits, axis=-1), num_samples=1)
        if next_token == tokenizer.sep_token_id:  # 遇到[SEP]则表明response生成结束
            break
        response.append(next_token.item())
        input_id = paddle.concat((input_id, next_token.unsqueeze(0)), axis=1)
        # his_text = tokenizer.convert_ids_to_tokens(curr_input_tensor.tolist())
        # print("his_text:{}".format(his_text))
    #history.append(response)
    text = tokenizer.convert_ids_to_string(response)

    rouge_score = rouge.get_scores(text,target)
    score1 = rouge_score[0]["rouge-1"]["p"]
    score2 = rouge_score[0]["rouge-2"]["p"]
    score3 = rouge_score[0]["rouge-l"]["p"]
    return score1,score2,score3

tokenizer = GPTTokenizer.from_pretrained("gpt2-medium-en")
tokenizer.add_special_tokens({"sep_token":"<sep>"})
rouge = Rouge()
max_len = 50
repetition_penalty = 1.0
temperature=1


net = MyModel(tokenizer.vocab_size+1)
net_dic = paddle.load("./model/model_2_262500.pkl")
net.set_state_dict(net_dic)

train_set, dev_set, test_set = load_dataset("cnn_dailymail",  splits=["train", "dev", "test"])

s1_all,s2_all,s3_all = 0,0,0
for data in tqdm(test_set):
    content = data["article"].lower()
    title = data["highlights"].lower()
    s1,s2,s3=eval(content,title)
    s1_all+=s1
    s2_all+=s2
    s3_all+=s3
print(f"r1:{s1_all/len(test_set)},r2:{s2_all/len(test_set)},r3:{s3_all/len(test_set)}")




