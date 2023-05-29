from transformers import PegasusForConditionalGeneration
# Need to download tokenizers_pegasus.py and other Python script from Fengshenbang-LM github repo in advance,
# or you can download tokenizers_pegasus.py and data_utils.py in https://huggingface.co/IDEA-CCNL/Randeng_Pegasus_523M/tree/main
# Strongly recommend you git clone the Fengshenbang-LM repo:
# 1. git clone https://github.com/IDEA-CCNL/Fengshenbang-LM
# 2. cd Fengshenbang-LM/fengshen/examples/pegasus/
# and then you will see the tokenizers_pegasus.py and data_utils.py which are needed by pegasus model
from tokenizers_pegasus import PegasusTokenizer

import opencc

converter = opencc.OpenCC('t2s.json')
model = PegasusForConditionalGeneration.from_pretrained("IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese")
tokenizer = PegasusTokenizer.from_pretrained("IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese")

# text = "据微信公众号“界面”报道，4日上午10点左右，中国发改委反垄断调查小组突击查访奔驰上海办事处，调取数据材料，并对多名奔驰高管进行了约谈。截止昨日晚9点，包括北京梅赛德斯-奔驰销售服务有限公司东区总经理在内的多名管理人员仍留在上海办公室内"
def summarize(text):
    # text is a list of sentences
    doc = "，".join(text) 
    simplified_doc = converter.convert(doc)
    inputs = tokenizer(simplified_doc, max_length=1024, return_tensors="pt")

    # Generate Summary
    summary_ids = model.generate(inputs["input_ids"])
    return tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

if __name__ == '__main__':
    print(
        summarize([
            '嘉義市長選舉改至18日投票，衛福部前部長陳時中6日替民進黨嘉義市長參選人李俊俋站',
            '台。陳時中於臉書發文表示，「若對我的付出與努力感到疼惜、為我不捨，請您將這份感',
            '情轉化為相挺俊俋的心」，幫忙催票、固票。',
            '陳時中指出，李俊俋的從政生涯有紮實完整的資歷，從嘉義市副市長、立法委員，至總統',
            '府副秘書長，不論從地方到中央，均秉持專業嚴謹的態度。一方面能與中央聯結，建立良',
            '好的溝通管道；另一方面，也能與嘉義縣的五星縣長翁章梁串聯，嘉義縣市通力合作，打',
            '破縣市邊界做整體規劃，擴大經濟規模，讓各類型產業都能在嘉義多元蓬勃發展。',
            '陳時中表示，李俊俋能落實區域治理與政見的優秀人選。從小自嘉義長大的他，發自內心',
            '的表示「有產業才留得住人才、有人才未來才會發展。」因此，包括基礎交通建設、土地',
            '規劃都必須具備前瞻願景的思考。其次，對長者的照顧與福利，亦必須逐步增加並落實。',
            '這些都是李俊俋作為嘉義子弟想實現的的政見內容。透過自己過去的豐富經歷，李俊俋必',
            '能使嘉義市在全國的發展進程中免於被邊緣化，也讓嘉義的少年人能看到嘉義的希望。',
            '陳時中提到，在防疫期間或首都選戰中，若對他的付出與努力感到疼惜、為他不捨，請民',
            '眾將這份感情轉化為相挺李俊俋的心，幫忙催票、固票，最重要的是，在18日投給李俊俋',
            '1票。',
        ])
    )