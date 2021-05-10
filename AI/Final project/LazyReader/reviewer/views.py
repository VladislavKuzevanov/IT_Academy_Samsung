from django.shortcuts import render
from django.views import View
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from pytorch_transformers import BertTokenizer, BertConfig
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import numpy as np

import requests
import json
import re
import os

import pickle

try:
    import transformers
    from transformers import BartTokenizer, BartForConditionalGeneration
except ImportError:
    print("Error")

# Create your views here.


class Index(View):

    def get(self, request):

        return render(request, 'reviewer/index.html', {})

    def post(self, request):

        url = str(request.POST.get("rt_url"))
        print("URL: ", url)

        from LazyReader.Parser import get_reviews_on_page, get_reviews

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        def modified_texts(texts):
            texts = ["[CLS] " + str(text) + " [SEP]" for text in texts]

            tokenized_texts = [tokenizer.tokenize(sent[:512]) for sent in texts]
            input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

            input_ids = pad_sequences(
                input_ids,
                maxlen=100,
                dtype="long",
                truncating="post",
                padding="post"
            )

            attention_masks = [[float(i > 0) for i in seq] for seq in input_ids]

            classification_inputs = torch.tensor(input_ids).to(torch.int64)
            classification_masks = torch.tensor(attention_masks).to(torch.int64)

            classification_data = TensorDataset(
                classification_inputs,
                classification_masks
            )

            classification_dataloader = DataLoader(
                classification_data,
                sampler=SequentialSampler(classification_data),
                batch_size=32
            )

            return classification_dataloader

        reviews = get_reviews(url)

        classification_dataloader = modified_texts(reviews)

        with open('bert_model.pkl', 'rb') as f:
            model_ = pickle.load(f)

        model_.eval()
        test_preds = []

        for batch in classification_dataloader:
            batch = tuple(t for t in batch)

            b_input_ids, b_input_mask = batch

            with torch.no_grad():
                logits = model_(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

            logits = logits[0].detach().cpu().numpy()

            batch_preds = np.argmax(logits, axis=1)
            test_preds.extend(batch_preds)

        positive = sum(1 for elem in test_preds if elem == 1)
        print("num of positive: ", positive)
        negative = sum(1 for elem in test_preds if elem == 0)
        print("num of negative: ", negative)

        general_text = []
        if (positive >= negative):
            for i in range(len(test_preds)):
                if (test_preds[i] == 1):
                    general_text.append(reviews[i])
        else:
            for i in range(test_preds):
                if (test_preds[i] == 0):
                    general_text.append(reviews[i])

        general_text = "".join(general_text).replace('\n', '')

        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

        article_input_ids = tokenizer.batch_encode_plus([general_text], return_tensors='pt', max_length=512)[
            'input_ids']
        article_input_ids = article_input_ids
        summary_ids = model.generate(article_input_ids,
                                     num_beams=4,
                                     length_penalty=2.0,
                                     max_length=512,
                                     min_length=128,
                                     no_repeat_ngram_size=5)

        summary_txt = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)
        context = {'summary': summary_txt}

        return render(request, 'reviewer/index.html', context)
