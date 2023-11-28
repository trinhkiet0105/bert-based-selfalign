import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, RobertaTokenizer,AutoTokenizer
from typing import Dict, List, Tuple, Union
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from torch.utils.data import Sampler

import os
import pickle

class GPReviewDataset(Dataset):

    def __init__(self, question1, question2, targets, tokenizer, max_len):
        self.question1 = question1
        self.question2 = question2
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.question1)

    def __getitem__(self, item):

        question1 = str(self.question1[item])
        question2 = str(self.question2[item])
        target = self.targets[item]

        question1_encoding = self.tokenizer.encode_plus(
            question1,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )

        question2_encoding = self.tokenizer.encode_plus(
            question2,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )

        return {
            'question1': question1,
            'question2': question2,
            'question1_ids': question1_encoding['input_ids'].flatten(),
            'question1_attention_mask': question1_encoding['attention_mask'].flatten(),
            'question2_ids': question2_encoding['input_ids'].flatten(),
            'question2_attention_mask': question2_encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }


class TripletDataset(Dataset):

    def __init__(self, anchor, positive, negative, tokenizer, max_len):
        self.anchor = anchor
        self.positive = positive
        self.negative = negative
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.anchor)

    def __getitem__(self, item):

        anchor = str(self.anchor[item])
        positive = str(self.positive[item])
        negative = str(self.negative[item])

        anchor_encoding = self.tokenizer.encode_plus(
            anchor,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )

        positive_encoding = self.tokenizer.encode_plus(
            positive,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )

        negative_encoding = self.tokenizer.encode_plus(
            negative,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )

        return {
            'anchor': anchor,
            'positive': positive,
            'negative': negative,
            'anchor_ids': anchor_encoding['input_ids'].flatten(),
            'anchor_attention_mask': anchor_encoding['attention_mask'].flatten(),
            'positive_ids': positive_encoding['input_ids'].flatten(),
            'positive_attention_mask': positive_encoding['attention_mask'].flatten(),
            'negative_ids': negative_encoding['input_ids'].flatten(),
            'negative_attention_mask': negative_encoding['attention_mask'].flatten(),
        }


def create_data_loader(df, tokenizer, max_len, batch_size, mode='train'):
    ds = GPReviewDataset(
        question1=df.question_x.to_numpy(),
        question2=df.question_y.to_numpy(),
        targets=df.is_duplicate.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    if mode == 'train':
        return DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=4,
            shuffle=True,
        )
    else:
        return DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=4,
            shuffle=False,
        )


def create_triplet_data_loader(df, tokenizer, max_len, batch_size, mode='train'):
    ds = TripletDataset(
        anchor=df.anchor.to_numpy(),
        positive=df.positive.to_numpy(),
        negative=df.negative.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    if mode == 'train':
        return DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=4,
            shuffle=True,
        )
    else:
        return DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=4,
            shuffle=False,
        )

def get_data_df(train_dir,test_dir,config):
    df_test = pd.read_csv(test_dir,index_col=False)
    df_test['is_duplicate'] = [1] * len(df_test)
    # df_test2 = pd.read_csv('../test_final.csv')
    # df_test = df_test[['question1','question2','is_duplicate']]
    # df_test = pd.concat([df_test,df_test2])


    # Train data
    # df_train = pd.read_csv('../test_final.csv') #
    # df_train2 = pd.read_csv('../3254_train.csv') #
    # df_train3 = pd.read_csv('../600_train.csv') #
    # df_train3 = df_train3.drop('Unnamed: 0',axis=1)
    # df_train4 = pd.read_csv('../9303_train.csv') #
    # df_train4.sample(frac=1)
    # df_train4 = df_train4[:3000]
    # df_train = pd.concat([df_train,df_train2,df_train3,df_train4])

    #Train data triplet

    df_train = pd.read_csv(train_dir)
    
    if config.use_aug_data == True:
      df_train2 = pd.read_csv('./data/triplet_data.csv')
      df_train2 = df_train2.drop('Unnamed: 0',axis=1)
      df_train = pd.concat([df_train,df_train2])

    print(df_train.shape, df_test.shape) # question1, question2, is_duplicate
    return df_train, df_test


class MELDDataset(Dataset):
    def __init__(self, filepath, tokenizer):
        self.data = pd.read_csv(filepath)
        self.tokenizer = tokenizer
        self.labels = self.data['Emotion'].astype('category').cat.codes
        self.label_classes = self.data['Emotion'].astype('category').cat.categories
        self.num_labels = len(self.label_classes)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data['Utterance'].iloc[idx]
        label = self.labels.iloc[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            truncation=True,
            padding=False,
            return_tensors='pt',
            return_token_type_ids=False,
            return_attention_mask=True,
            add_special_tokens=True
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
def collate_fn(batch):
    # Get the maximum length of input_ids in the current batch
    max_length = max([len(item['input_ids']) for item in batch])
    
    # Pad the sequences to the max_length
    padded_input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=0)
    padded_attention_mask = pad_sequence([item['attention_mask'] for item in batch], batch_first=True, padding_value=0)
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'input_ids': padded_input_ids,
        'attention_mask': padded_attention_mask,
        'labels': labels
    }

    
def get_MELD_dataloader(filepath, tokenizer, train: bool, batch_size=8):
    dataset = MELDDataset(filepath, tokenizer)
    if train:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return dataloader


class IEMOCAPAudioDataset(Dataset):
    def __init__(self, filepath):
        with open(filepath, "rb") as train_file:
            self.data_list = pickle.load(train_file)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _, _, waveform, label = self.data_list[index].values()

        if isinstance(waveform, torch.Tensor):
            waveform = waveform.clone().detach()
        else:  # assuming it's a numpy array
            waveform = torch.tensor(waveform)

        label = torch.tensor(label, dtype=torch.long).squeeze()  # assuming label is an integer

        return waveform, label
    
def IEMOCAPAudioDataset_collate_fn(batch):
    waveforms, labels = zip(*batch)

    # Ensure waveforms are 1D
    waveforms = [wf.squeeze() for wf in waveforms if wf.ndim > 1]

    waveforms_padded = pad_sequence(waveforms, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)
    return waveforms_padded, labels

    
def get_IEMOCAPAudio_dataloader(filepath, train: bool, batch_size=8):
    dataset = IEMOCAPAudioDataset(filepath)
    if train:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=IEMOCAPAudioDataset_collate_fn)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=IEMOCAPAudioDataset_collate_fn)
    return dataloader