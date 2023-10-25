import argparse
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer
import warnings

warnings.filterwarnings("ignore")


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="", type=str,
                        help="pretrained model name or path!", required=True)
    parser.add_argument("--lora_name_or_path", default="", type=str,
                        help="lora parameter path!", required=True)
    return parser.parse_args()


prompt_template = """文本分类任务：将一段用户给手机的评论进行分类。下面是一些范例：xxxxxx -> """


def get_prompt(text):
    return prompt_template.replace('xxxxxx', text)


def predict(model, tokenizer, text):
    response, history = model.chat(tokenizer, get_prompt(text), history=[], temperature=0.01)
    return response


if __name__ == "__main__":
    args = set_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_name_or_path,
                                      load_in_8bit=False,
                                      trust_remote_code=True,
                                      device_map='auto')
    model = PeftModel.from_pretrained(model, args.lora_name_or_path)
    model = model.merge_and_unload()  # merge

    predict(model, tokenizer, '手机发热很严重啊')
