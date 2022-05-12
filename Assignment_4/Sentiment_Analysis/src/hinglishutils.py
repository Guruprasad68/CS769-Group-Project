import re

import gdown
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import torch
import tarfile
from keras.preprocessing.sequence import pad_sequences
from sklearn import preprocessing
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset
)
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer
)
import os
import time
import random
import wandb
import datetime
import time


def print_confusion_matrix(confusion_matrix, class_names, figsize=(10, 7), fontsize=14):
    """Prints a confusion matrix, as returned by 
    sklearn.metrics.confusion_matrix, as a heatmap.

    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call 
        to sklearn.metrics.confusion_matrix.
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the 
        given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal 
        size of the outputed figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix,
        index=class_names,
        columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(
        heatmap.yaxis.get_ticklabels(), rotation=0, ha="right", fontsize=fontsize
    )
    heatmap.xaxis.set_ticklabels(
        heatmap.xaxis.get_ticklabels(), rotation=45, ha="right", fontsize=fontsize
    )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


def get_files_from_gdrive(url: str, fname: str) -> None:
    """Converts google share link to something that can be
    downloaded using gdown

    Args:
        url (str): google drive url
        fname (str): output filename
    """
    file_id = url.split("/")[5]
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, fname, quiet=False)
    if fname[-3:] == "tar":
        tf = tarfile.open(fname)
        tf.extractall()


def clean(df, col):
    """Cleaning Twiitter data

    Arguments:
        df {[pandas dataframe]} -- Dataset that needs to be cleaned
        col {[string]} -- column in which text is present

    Returns:
        [pandas dataframe] -- Datframe with a "clean_text" column
    """
    df["clean_text"] = df[col]
    df["clean_text"] = (
        (df["clean_text"])
        .apply(lambda text: re.sub(r"RT\s@\w+:", "Retweet", text))  # Removes RTS
        .apply(lambda text: re.sub(r"@", "mention ", text))  # Replaces @ with mention
        .apply(lambda text: re.sub(r"#", "hashtag ", text))  # Replaces # with hastag
        .apply(lambda text: re.sub(r"http\S+", "", text))  # Removes URL
    )
    return df


def flat_accuracy(preds, labels):
    """Accuracy calulations for tensors"""
    print("logits = ", preds)
    print("labels =", labels)
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    print("logits_flat = ", pred_flat)
    print("labels_flat =", labels_flat)
    print("length_labels_flat = ",len(labels_flat))
    print("length_logits_flat = ",len(pred_flat))
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def flat_prf(preds, labels):
    print("logits = ", preds)
    print("labels =",labels)
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    print("length_labels_flat = ",len(labels_flat))
    print("length_logits_flat = ",len(pred_flat))
    return precision_recall_fscore_support(
        labels_flat, pred_flat, labels=[0, 1, 2], average="macro"
    )


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """

    elapsed_rounded = int(round((elapsed)))

    return str(datetime.timedelta(seconds=elapsed_rounded))


def modify_transformer_config(
    model,
    batch_size=8,
    attention_probs_dropout_prob=0.4,
    learning_rate=5e-7,
    adam_epsilon=1e-8,
    hidden_dropout_prob=0.3,
    lm_model_dir=None,
):
    if model == "bert":
        config = BertConfig.from_json_file(f"{lm_model_dir}/config.json")
    elif model == "distilbert":
        config = DistilBertConfig.from_json_file(f"{lm_model_dir}/config.json")
    elif model == "roberta":
        config = RobertaConfig.from_json_file(f"{lm_model_dir}/config.json")
    config.attention_probs_dropout_prob = attention_probs_dropout_prob
    config.do_sample = True
    config.num_beams = 500
    config.hidden_dropout_prob = hidden_dropout_prob
    config.repetition_penalty = 5
    config.num_labels = 3
    return config


def check_for_gpu(name):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_sentences_and_labels(
    train_json="train.json", text_col="text", label_col="sentiment"
):
    train_df = pd.read_json(train_json)
    sentences = train_df[text_col]
    labels = train_df[label_col]
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    labels = le.transform(labels)
    return sentences, labels, le


def evaulate_and_save_prediction_results(
    tokenizer,
    max_len,
    model,
    device,
    le,
    final_name,
    model_name,
    text_col="text",
    label_col="sentiment",
):
    save_name = os.path.basename(final_name)[:-5]
    print(f"Evaluating {final_name} using {model_name} and Saving the Predictions")
    final_test_df = pd.read_json(final_name)
    sentences = final_test_df[text_col]

    prediction_inputs, prediction_masks = prep_input(sentences, tokenizer, max_len)

    batch_size = 32

    prediction_data = TensorDataset(prediction_inputs, prediction_masks)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(
        prediction_data, sampler=prediction_sampler, batch_size=batch_size
    )
    model.eval()

    predictions = get_preds_from_model(prediction_dataloader, device, model)

    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    proba = [item for sublist in predictions for item in sublist]
    preds = np.argmax(proba, axis=1).flatten()

    output = le.inverse_transform(flat_predictions.tolist())
    output_df = pd.DataFrame(
        {
            "uid": list(final_test_df["uid"]),
            "sentiment": output,
            text_col: list(final_test_df[text_col]),
        }
    )
    
    output_df.to_csv(f"{model_name}-{save_name}-output-df.csv")
    #wandb.save(f"{name}-{save_name}-output-df.csv")
    proba = [item for sublist in predictions for item in sublist]
    preds = np.argmax(proba, axis=1).flatten()
    full_output = output_df
    full_output["proba_negative"] = pd.DataFrame(proba)[0]
    full_output["proba_neutral"] = pd.DataFrame(proba)[1]
    full_output["proba_positive"] = pd.DataFrame(proba)[2]
    full_output.to_csv(f"{model_name}-{save_name}-full-output.csv")
    #wandb.save(f"{name}-{save_name}-full-output.csv")
    return full_output


def get_preds_from_model(prediction_dataloader, device, model):
    predictions = []

    for batch in prediction_dataloader:

        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_input_mask = batch

        with torch.no_grad():

            outputs = model(b_input_ids, attention_mask=b_input_mask)

        logits = outputs[0]

        logits = logits.detach().cpu().numpy()

        predictions.append(logits)
    return predictions


def prep_input(sentences, tokenizer, max_len):
    input_ids = []

    for sent in sentences:
        if sent:
            encoded_sent = tokenizer.encode(
                sent,
                add_special_tokens=True,
            )

            input_ids.append(encoded_sent)
        if not sent:
            print(f"NAN sent detected {sent}")

    input_ids = pad_sequences(
        input_ids, maxlen=max_len, dtype="long", 
        truncating="post", padding="post", value=0,
    )

    attention_masks = []

    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    prediction_inputs = torch.tensor(input_ids)
    prediction_masks = torch.tensor(attention_masks)
    return prediction_inputs, prediction_masks


def set_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


def make_dev_dataloader(
    dev_json,
    tokenizer,
    max_len = 150
):
    batch_size = 32

    dev_sentences, dev_labels, _ = load_sentences_and_labels(
                        dev_json, 
                        text_col="text", 
                        label_col="sentiment"
                        )

    dev_labels = torch.tensor(dev_labels)
    dev_inputs, dev_masks = prep_input(dev_sentences, tokenizer, max_len)

    dev_data = TensorDataset(dev_inputs, dev_masks, dev_labels )
    dev_sampler = SequentialSampler(dev_data)
    dev_dataloader = DataLoader(
        dev_data, 
        sampler=dev_sampler, 
        batch_size=batch_size
    )
    return dev_dataloader

def make_dataloaders(
    train_inputs,
    train_masks,
    train_labels,
    batch_size,
    validation_inputs,
    validation_masks,
    validation_labels,
):

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, 
        sampler=train_sampler, 
        batch_size=batch_size, 
    )

    validation_data = TensorDataset(
        validation_inputs, validation_masks, validation_labels
    )
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(
        validation_data, 
        sampler=validation_sampler, 
        batch_size=batch_size,
    )
    return train_dataloader, validation_dataloader


def load_masks_and_inputs(input_ids, labels, attention_masks):

    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(
        input_ids, labels, random_state=2018, test_size=0.2
    )

    train_masks, validation_masks, _, _ = train_test_split(
        attention_masks, labels, random_state=2018, test_size=0.2
    )

    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)

    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)

    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)
    return (
        train_inputs,
        train_masks,
        train_labels,
        validation_inputs,
        validation_masks,
        validation_labels,
    )


def create_attention_masks(input_ids):

    attention_masks = []

    for sent in input_ids:

        att_mask = [int(token_id > 0) for token_id in sent]

        attention_masks.append(att_mask)
    return attention_masks


def add_padding(input_ids, max_len):

    input_ids = pad_sequences(
        input_ids,
        maxlen=max_len,
        dtype="long",
        truncating="post",
        padding="post",
    )
    return input_ids


def tokenize_the_sentences(sentences, model_name, lm_model_dir):

    if model_name == "bert":
        print("Loading BERT tokenizer...\n")
        tokenizer = BertTokenizer.from_pretrained(lm_model_dir)
    elif model_name == "distilbert":
        print("Loading DistilBERT tokenizer...\n")
        tokenizer = DistilBertTokenizer.from_pretrained(lm_model_dir)
    elif model_name == "roberta":
        print("Loading Roberta tokenizer...\n")
        tokenizer = RobertaTokenizer.from_pretrained(lm_model_dir)
    
    #tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    input_ids = []
    for sent in sentences:

        encoded_sent = tokenizer.encode(
            sent,
            add_special_tokens=True,
        )

        input_ids.append(encoded_sent)

    return tokenizer, input_ids

def save_checkpoint(model,model_name):
    d = time.strftime('%d_%m')
    output_dir = f"./{model_name}/checkpoints_{d}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)

def save_model(full_output, model, tokenizer, model_name):
    
    d = time.strftime('%d_%m')
    full_output.to_csv(f"{model_name}_preds_{d}.csv")
    #wandb.save(f"{model_name}_preds.csv")

    output_dir = f"./{model_name}/{d}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    #wandb.save(f"{output_dir}/*")


def load_lm_model(config, model_name, lm_model_dir):
    if model_name == "bert":
        model = BertForSequenceClassification.from_pretrained(
            lm_model_dir, config=config
        )
    elif model_name == "distilbert":
        model = DistilBertForSequenceClassification.from_pretrained(
            lm_model_dir, config=config
        )
    if model_name == "roberta":
        model = RobertaForSequenceClassification.from_pretrained(
            lm_model_dir, config=config
        )
    model.cuda()
    params = list(model.named_parameters())
    return model


def train_model(
    epochs,
    model,
    train_dataloader,
    device,
    optimizer,
    scheduler,
    loss_values,
    model_name,
    validation_dataloader,
    dev_dataloader,
):
    best_f1 = -1
    for epoch_i in range(0, epochs):
        print("Starting epoch :",epoch_i+1)
        start = time.time()

        total_loss = 0

        model.train()

        for step, batch in enumerate(train_dataloader):
            n_steps = len(train_dataloader)
            if (step%100 == 0):
              print(f"====== Epoch {epoch_i+1} : Step {step} of {n_steps} ======")
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()
            if model_name == "bert":
                outputs = model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                )
            else:
                outputs = model(
                    b_input_ids,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                )

            loss = outputs[0]

            total_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            scheduler.step()

            if step % 200 == 0 and not step == 0:
  
                valid_f1 = run_valid(
                  model, 
                  model_name, 
                  validation_dataloader, 
                  device,
                  log_name="Valid")
                
                dev_f1 = run_valid(
                  model, 
                  model_name, 
                  dev_dataloader, 
                  device,
                  log_name="Dev(Test)")

                avg_train_loss = total_loss / (step+1)
                loss_values.append(avg_train_loss)
                wandb.log({"Training loss": avg_train_loss})
        
                if dev_f1 > best_f1:
                  best_f1 = dev_f1
                  print(f"Best dev_f1 = {best_f1:.3f}")
                  print("Saving Model")
                  save_checkpoint(model,model_name)
        
        train_f1 = run_valid(
                  model, 
                  model_name, 
                  train_dataloader, 
                  device,
                  log_name="Train")

        end = time.time()
        epoch_min = (end-start)/60
        print(f"Epoch {epoch_i+1} completed! Epoch Time : {epoch_min:.3f} min")

    print("Training complete!\n")


def run_valid(
  model, model_name, 
  validation_dataloader, 
  device,log_name):

    start = time.time()
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    eval_p = 0
    eval_r = 0
    eval_f1 = 0

    (
        eval_accuracy,
        nb_eval_steps,
        eval_p,
        eval_r,
        eval_f1,
    ) = evaluate_data_for_one_epochs(
        eval_accuracy,
        eval_p,
        eval_r,
        eval_f1,
        nb_eval_steps,
        model,
        model_name,
        validation_dataloader,
        device,
    )
    
    print(f"{log_name} Accuracy = {eval_accuracy:.3f} || f1_score = {eval_f1:.3f}")
    
    wandb.log({f"{log_name} Accuracy": eval_accuracy})
    wandb.log(
        {
            f"{log_name} Precision": (eval_p),
            f"{log_name} Recall": (eval_r),
            f"{log_name} F1": (eval_f1),
        }
    )
    
    end = time.time()
    eval_time = (end-start)/60
    print(f"Evaluation Time : {eval_time:.3f} min")
    return eval_f1


def evaluate_data_for_one_epochs(
    eval_accuracy,
    eval_p,
    eval_r,
    eval_f1,
    nb_eval_steps,
    model,
    model_name,
    validation_dataloader,
    device,
):

    logits_list = []
    labels_list = []
    for batch in validation_dataloader:

        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            if model_name == "bert":
                outputs = model(
                    b_input_ids, token_type_ids=None, attention_mask=b_input_mask
                )
            else:
                outputs = model(b_input_ids, attention_mask=b_input_mask)
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to("cpu").numpy()
        logits_list.append(logits)
        labels_list.append(label_ids)
        # tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        # temp_eval_prf = flat_prf(logits, label_ids)

        # eval_accuracy += tmp_eval_accuracy
        # eval_p += temp_eval_prf[0]
        # eval_r += temp_eval_prf[1]
        # eval_f1 += temp_eval_prf[2]
        nb_eval_steps += 1
    
    flat_logits = [item for sublist in logits_list for item in sublist]
    flat_preds = np.argmax(flat_logits, axis=1).flatten()

    flat_labels = [item for sublist in labels_list for item in sublist]
    flat_labels = np.array(flat_labels)

    eval_accuracy = accuracy_score(flat_labels, flat_preds)
    prf = precision_recall_fscore_support(
                                          flat_preds, 
                                          flat_labels, 
                                          labels=[0,1,2], 
                                          average="macro")
    eval_p = prf[0]
    eval_r = prf[1]
    eval_f1 = prf[2]

    return eval_accuracy, nb_eval_steps, eval_p, eval_r, eval_f1
