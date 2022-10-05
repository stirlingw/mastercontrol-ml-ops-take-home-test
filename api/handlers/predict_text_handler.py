import torch
from torch.nn import Linear
from torch.utils.data import DataLoader
import torchmetrics
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from typing import Dict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_CHECKPOINT = "sentence-transformers/all-MiniLM-L6-v2"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)


class TextClassifierModel(torch.nn.Module):
    def __init__(self, label_vocab, model_checkpoint):
        super().__init__()
        self.label_vocab = label_vocab
        self.id2label = {id: label for id, label in enumerate(self.label_vocab)}
        self.label2id = {label: id for id, label in enumerate(self.label_vocab)}
        self.base_model = AutoModel.from_pretrained(model_checkpoint)
        self.classifier = Linear(self.base_model.config.hidden_size, len(self.label_vocab))

    def forward(self, X, attention_mask):
        encoded_layers = self.base_model(
            input_ids=X, attention_mask=attention_mask
        ).last_hidden_state
        embeddings = encoded_layers[:, 0]
        logits = self.classifier(embeddings)
        return logits


class Dataset(torch.utils.data.Dataset):
    def __init__(self, text_ser, text_label_ser=None, label_vocab=None, ds_split_name=None):
        if text_label_ser is None and not label_vocab:
            raise ValueError(f"both text_label_ser and label_vocab-list are None; one of them must be non-None")
        self.text_ser = text_ser.reset_index(drop=True)
        self.text_label_ser = text_label_ser.reset_index(drop=True) if text_label_ser is not None else None
        self.ds_split_name = ds_split_name
        self.label_vocab = label_vocab or text_label_ser.unique().tolist()
        self.id2label = {id: label for id, label in enumerate(self.label_vocab)}
        self.label2id = {label: id for id, label in enumerate(self.label_vocab)}

    def __len__(self):
      return len(self.text_ser)

    def __getitem__(self, idx):
        text = self.text_ser[idx]
        label = self.label2id[self.text_label_ser[idx]] if self.text_label_ser is not None else None
        return text, label


def dl_collate_batch(batch):
    text_batch, label_batch = zip(*batch)
    text_batch, label_batch = list(text_batch), list(label_batch)

    X, attention_mask, y = None, None, None
    if text_batch is not None:
        encodings = TOKENIZER(text_batch, truncation=True, padding=True)
        X = torch.tensor(encodings["input_ids"])
        attention_mask = torch.tensor(encodings["attention_mask"])
        y = torch.tensor(label_batch)

    return X, attention_mask, y


def predict_text_handler(predict_text_request: Dict):
    batch_size = 32
    text_col = "comment"
    label_col = "Focal_Obj"
    file_name = predict_text_request["file_name"]
    tagged_text_datapath = f"./data/{file_name}"
    text_df = pd.read_csv(tagged_text_datapath)
    label_vocab = text_df[label_col].unique().tolist()
    test_ds = Dataset(text_df[text_col], text_label_ser=text_df[label_col],
                      label_vocab=label_vocab, ds_split_name="test")
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=dl_collate_batch)
    model_outpath = "./models/ml/tht_model_save_dict.pth"
    model_dict = torch.load(model_outpath)
    text_model = TextClassifierModel(model_dict["label_vocab"], model_dict["base_model_checkpoint"])
    text_model.to("cpu")
    text_model.load_state_dict(model_dict["state_dict"])
    text_model.to(DEVICE)
    test_acc = torchmetrics.Accuracy()
    text_model.eval()

    # print(f"Testing data: (size: {len(test_dl.dataset)}):\n")
    all_logits = []
    with torch.no_grad():
        for batch_data in test_dl:
            X, attention_mask, y = batch_data
            X = X.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            y = y.to(DEVICE)

            logits = text_model(X, attention_mask)
            all_logits.append(logits)

            test_acc(torch.argmax(logits.to("cpu"), 1), y.to("cpu"))
    all_logits = torch.cat(all_logits, dim=0)

    return {
        "vocab": text_model.label_vocab,
        "predictions": torch.argmax(all_logits.to("cpu"), 1).numpy().tolist(),
        "confidences": torch.max(torch.softmax(all_logits.to("cpu"), 1), 1).values.numpy().tolist(),
    }
