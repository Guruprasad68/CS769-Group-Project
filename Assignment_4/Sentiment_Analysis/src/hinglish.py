from fastcore.utils import store_attr
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AdamW
import pandas as pd
from transformers import get_linear_schedule_with_warmup
from hinglishutils import (
    add_padding,
    check_for_gpu,
    create_attention_masks,
    evaulate_and_save_prediction_results,
    load_lm_model,
    load_masks_and_inputs,
    load_sentences_and_labels,
    make_dataloaders,
    modify_transformer_config,
    save_model,
    set_seed,
    tokenize_the_sentences,
    train_model,
    make_dev_dataloader
)
from datetime import datetime
import wandb


class HinglishTrainer:
    def __init__(
        self,
        model_name: str,
        batch_size: int = 32,
        attention_probs_dropout_prob: float = 0.4,
        learning_rate: float = 5e-7,
        adam_epsilon: float = 1e-8,
        hidden_dropout_prob: float = 0.3,
        epochs: int = 3,
        max_len: int = 150,
        lm_model_dir: str = None,
        wname=None,
        drivepath="../drive/My\ Drive/HinglishNLP/repro",
        train_json: str ="self_train_data/concat_train.json",
        dev_json: str ="self_train_data/valid.json",
        test_json: str ="self_train_data/test.json",
        test_labels_csv:str ="self_train_data/test_labels.csv",
    ):
        store_attr()
        self.timestamp = str(datetime.now().strftime("%d.%m.%y"))
        if not self.wname:
            self.wname = self.model_name
        wandb.init(
            project="hinglish",
            config={
                "model_name": self.model_name,
                "batch_size": self.batch_size,
                "attention_probs_dropout_prob": self.attention_probs_dropout_prob,
                "learning_rate": self.learning_rate,
                "adam_epsilon": self.adam_epsilon,
                "hidden_dropout_prob": self.hidden_dropout_prob,
                "epochs": self.epochs,
            },
            name=f"{self.wname} {self.timestamp}",
        )
        print({"Model Info": f"Setup self.model training for {model_name}"})
        self.device = check_for_gpu(self.model_name)
        if not lm_model_dir:
            if self.model_name == "bert":
                self.lm_model_dir = "bert"
            elif self.model_name == "distilbert":
                self.lm_model_dir = "distilBert"
            elif self.model_name == "roberta":
                self.lm_model_dir = "roberta"

    def setup(self):
        print("Loading Sentences and Labels")
        sentences, labels, self.le = load_sentences_and_labels(
                        train_json=self.train_json, 
                        text_col="text", 
                        label_col="sentiment"
                        )

        self.tokenizer, input_ids = tokenize_the_sentences(
            sentences, self.model_name, self.lm_model_dir
        )
        input_ids = add_padding(input_ids, max_len = self.max_len)
        
        attention_masks = create_attention_masks(input_ids)
        (
            train_inputs,
            train_masks,
            train_labels,
            validation_inputs,
            validation_masks,
            validation_labels,
        ) = load_masks_and_inputs(input_ids, labels, attention_masks)
        
        print("Setting Config")
        self.config = modify_transformer_config(
            "bert",
            self.batch_size,
            self.attention_probs_dropout_prob,
            self.learning_rate,
            self.adam_epsilon,
            self.hidden_dropout_prob,
            self.lm_model_dir,
        )

        print("Preparing Dataloaders")

        # dev dataloader is prep from dev.json
        self.dev_dataloader = make_dev_dataloader(
            dev_json = self.dev_json,
            tokenizer = self.tokenizer,
            max_len = self.max_len
        )
        
        # train and valid dataloaders are splitted from train.json
        self.train_dataloader, self.validation_dataloader = make_dataloaders(
            train_inputs,
            train_masks,
            train_labels,
            self.batch_size,
            validation_inputs,
            validation_masks,
            validation_labels,
        )
        print("Train Length:", len(self.train_dataloader))
        print("Valid Length:",len(self.validation_dataloader))
        print("Dev Length:", len(self.dev_dataloader))

        print("Loading Model")
        self.model = load_lm_model(self.config, self.model_name, self.lm_model_dir)
        print('model loaded')

    def train(self):
        self.setup()
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            eps=self.adam_epsilon,
        )
        total_steps = len(self.train_dataloader) * self.epochs
        print("num_training_steps = ", total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=100,
            num_training_steps=total_steps,
        )
        set_seed()
        loss_values = []
        train_model(
            self.epochs,
            self.model,
            self.train_dataloader,
            self.device,
            optimizer,
            scheduler,
            loss_values,
            self.model_name,
            self.validation_dataloader,
            self.dev_dataloader
        )

    def evaluate(self):
        
        #Save predictions for validation set
        output = evaulate_and_save_prediction_results(
            self.tokenizer,
            self.max_len,
            self.model,
            self.device,
            self.le,
            final_name=self.dev_json,
            model_name=self.model_name,
        )
        v=pd.read_json(self.dev_json)
        print("Validation Accuracy: ",accuracy_score(output["sentiment"],v["sentiment"]))
        

        #Save predictions for test set
        full_output = evaulate_and_save_prediction_results(
            self.tokenizer,
            self.max_len,
            self.model,
            self.device,
            self.le,
            final_name=self.test_json,
            model_name=self.model_name,
        )
        l = pd.read_csv(self.test_labels_csv)
        prf = precision_recall_fscore_support(
            full_output["sentiment"], l["sentiment"], average="macro"
        )
        test_acc = accuracy_score(full_output["sentiment"], l["sentiment"])
        print("Test Accuracy: ",test_acc)
        print(f"Test Precision: {prf[0]}, Recall: {prf[1]} ,F1: {prf[2]}")
        wandb.log({"Precision": prf[0], "Recall": prf[1], "F1": prf[2]})
        wandb.log(
            {"Accuracy": str(test_acc)}
        )
        print("saving final model")
        save_model(full_output, self.model, self.tokenizer, self.model_name)
