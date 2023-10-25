import json
import torch
from utils.tools import build_inputs
from torch.utils.data import Dataset


class GLM2Dataset(Dataset):

    def __init__(self,
                 data_path: str = "train.txt",
                 tokenizer=None,
                 max_seq_length: int = 512,
                 is_skip: bool = True):
        self.lines = []
        with open(data_path, "r", encoding="utf-8") as reader:
            for i, line in enumerate(reader):
                skip_flag = False
                sample = json.loads(line.strip())
                src_tokens = build_inputs(query=sample["instruction"] + sample["input"], history=[])

                src_ids = tokenizer.encode(src_tokens,
                                           max_length=max_seq_length,
                                           truncation=True)

                if len(src_ids) > max_seq_length:
                    skip_flag = True

                tgt_ids = tokenizer.encode(sample["output"],
                                           max_length=max_seq_length,
                                           truncation=True,
                                           add_special_tokens=False)

                if len(tgt_ids) > max_seq_length:
                    skip_flag = True

                input_ids = src_ids + tgt_ids + [tokenizer.eos_token_id]

                # src_length = len(src_ids)
                # labels = [-100] * src_length + input_ids[src_length:]

                if skip_flag and is_skip:
                    continue

                self.lines.append({"input_ids": input_ids, "context_len": len(src_ids), 'target_len': len(tgt_ids)})

                # self.lines.append({"input_ids": input_ids, "labels": labels})

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        return self.lines[index]


class DataCollator:

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def __call__(self, features: list):
        len_ids = [len(feature["input_ids"]) for feature in features]
        # 计算出这个batch中的所有样本的 input_ids 长度，并找出最长的长度
        # 之后按照batch中最长的input_ids进行padding，不足的就用空补全
        longest = max(len_ids)
        # 初始化两个空列表 input_ids 和 labels_list，它们用于存储预处理后的输入数据和标签数据
        input_ids = []
        labels_list = []
        # 对 len_ids 和 features 进行合并，然后按照 input_ids 的长度进行逆序排序，也就是说，最长的 input_ids 会排在前面
        for length, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
            # 提取出当前样本的 input_ids 和 context_len
            ids = feature["input_ids"]
            context_len = feature["context_len"]
            # 生成标签数据
            # -100 是一个特殊的标记，它表示该位置的损失会在训练时被忽略
            # 标签数据的生成规则：先放入 context_len  个 -100，然后放入 ids 从 context_len 开始到最后的部分，最后再放入 longest - length 个 -100
            labels = (
                    [-100] * context_len + ids[context_len:] + [-100] * (longest - length)
            )  # -100标志位后面会在计算loss时会被忽略不贡献损失，我们集中优化target部分生成的loss
            # 对 input_ids 进行padding，如果 input_ids 的长度小于最长长度，那么在其后面添加足够数量的 pad_token_id
            ids = ids + [self.tokenizer.pad_token_id] * (longest - length)
            # 将处理过的 input_ids 和 labels 转为 LongTensor 类型，然后添加到相应的列表中
            input_ids.append(torch.LongTensor(ids))
            labels_list.append(torch.LongTensor(labels))

        # 将 input_ids 和 labels 的列表堆叠为一个新的tensor
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels_list)
        return {
            "input_ids": input_ids,
            "labels": labels,
        }


# def data_collator(batch: list):
#     lengths = [len(instance["input_ids"]) for instance in batch]
#     # padding by max len
#     batch_max_len = max(lengths)
#
#     input_ids_batch, labels_batch = [], []
#     for instance in batch:
#         input_ids = instance["input_ids"]
#         labels = instance["labels"]
#         padding_len = batch_max_len - len(input_ids)
#         input_ids = input_ids + [tokenizer.pad_token_id] * padding_len
#         labels = labels + [-100] * padding_len
#         input_ids_batch.append(input_ids)
#         labels_batch.append(labels)
#     return {"input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
#             "labels": torch.tensor(labels_batch, dtype=torch.long)}


# if __name__ == "__main__":
#
#     train_dataset = GLM2Dataset(data_path="D:/taohou/pyworkspace/ChatGLM2-demo/data/train_data.txt",
#                                 # tokenizer=tokenizer,
#                                 max_seq_length=512,
#                                 is_skip=True)
#
#     from torch.utils.data import DataLoader
#     train_dataloader = DataLoader(train_dataset,
#                                   collate_fn=data_collator,
#                                   batch_size=2,
#                                   num_workers=2,
#                                   shuffle=False)
#
#     for batch in train_dataloader:
#         print(batch)
#         break
