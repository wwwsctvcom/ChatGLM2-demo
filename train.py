import argparse
from pathlib import Path
from loguru import logger
from utils.tools import *
from utils.trainer import Trainer
from utils.dataset import GLM2Dataset, DataCollator
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType


def set_args():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument("--model_name_or_path", default="", type=str,
                        help="pretrained model name or path!", required=False)

    # dataset
    parser.add_argument("--train_path", default="data/train_data.txt", type=str, help="train dataset !")
    parser.add_argument("--test_path", default="data/test_data.txt", type=str, help="test dataset !")
    parser.add_argument("--max_seq_length", type=int, default=512, help="max length for input data !")
    parser.add_argument("--is_skip", action='store_true', help="skip the too long data !")

    # train
    parser.add_argument("--train_type", type=str, default="lora", choices=["lora", "all"],
                        help="train type for lora or all parameters training !")
    parser.add_argument("--lora_module_name", type=str, default="query_key_value",
                        help="lora module name for training !")
    parser.add_argument("--output_dir", type=str, default=None, help="")
    parser.add_argument("--epochs", type=int, default=1, help="")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size for train and test !")
    parser.add_argument("--num_workers", type=int, default=2, help="num workers for dataloader !")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate for training!")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="")
    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--local_rank", type=int, default=-1, help="")
    parser.add_argument("--gradient_checkpointing", action='store_true', help="")
    # todo, deepspeed
    # parser.add_argument("--ds_file", type=str, default="ds_zero2.json", help="")
    # parser = deepspeed.add_config_arguments(parser)
    # lora
    parser.add_argument("--lora_dim", type=int, default=8, help="")
    parser.add_argument("--lora_alpha", type=int, default=32, help="")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="")
    return parser.parse_args()


if __name__ == "__main__":
    args = set_args()

    # seed
    seed_everything(args.seed)

    # distributed
    if args.local_rank == -1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        logger.info("distributing training !")
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # deepspeed.init_distributed()
    logger.info(f"using device {device} !")

    # load tokenizer and config
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True, device_map='auto')

    # load ChatGLM2-6B model not in 8bit and tokenizer
    model = AutoModel.from_pretrained(args.model_name_or_path,
                                      load_in_8bit=False,
                                      trust_remote_code=True,
                                      device_map='auto')

    # gradient_checkpointing for saving cuda resource
    if args.gradient_checkpointing:
        model.supports_gradient_checkpointing = True
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # close model cache to ignore some warning, but need to reopen when inference.
    model.config.use_cache = False

    lora_module_name = "query_key_value".split(",")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False,
        r=8,
        target_modules=lora_module_name,
        bias="none",
        lora_alpha=32, lora_dropout=0.1,
    )

    model = get_peft_model(model, peft_config)
    model.is_parallelizable = True
    model.model_parallel = True
    model.print_trainable_parameters()

    # train dataset
    data_collator = DataCollator(tokenizer)
    train_dataset = GLM2Dataset(data_path=args.train_path,
                                tokenizer=tokenizer,
                                max_seq_length=args.max_seq_length,
                                is_skip=args.is_skip)
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=data_collator,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True)

    # test dataset
    test_dataset = GLM2Dataset(data_path=args.test_path,
                               tokenizer=tokenizer,
                               max_seq_length=args.max_seq_length,
                               is_skip=args.is_skip)
    test_dataloader = DataLoader(test_dataset,
                                 collate_fn=data_collator,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 shuffle=False)

    # start train
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                           T_max=args.epochs,
                                                           eta_min=args.learning_rate)
    trainer = Trainer(model=model,
                      tokenizer=tokenizer,
                      optimizer=optimizer,
                      scheduler=scheduler)
    trainer.train(epochs=args.epochs,
                  train_data_loader=train_dataloader,
                  test_data_loader=test_dataloader)

    # save lora parameter
    if args.lora_module_name:
        trainer.save_lora_ckpt("lora_param")

    # save model
    trainer.save_model((Path(args.output_dir) / get_cur_time()).__str__())
