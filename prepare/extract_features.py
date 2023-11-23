from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import numpy as np
import string
import spacy
nlp = spacy.load('en_core_web_sm')

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

### stop words
stop_words = set(stopwords.words('english'))


def count_repeat_words(text):
    # first remove stop words and punctuation
    text = text.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    filtered_sentence = [w for w in tokens if not w in stop_words]

    # now count words
    frequency = Counter(filtered_sentence)
    # take only words that appear more than once
    repeated = {k: v for k, v in frequency.items() if v > 1}
    return sum(repeated.values())


def words_count(text):
    return len(text.split())


def get_perplexity_with_model(prompts, model_name="gpt2", name=""):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_length = model.config.n_positions if model_name == "gpt2" else model.config.max_position_embeddings
    stride = 512

    ppl_all = []
    for i in range(len(prompts)):
        inputs = tokenizer(prompts[i], return_tensors="pt")
        seq_len = inputs.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = inputs.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over input tokens.
                # Multiply it with trg_len to get the summation instead of average.
                # We will take average over all the tokens to get the true average
                # in the last step of this example.
                neg_log_likelihood = outputs.loss * trg_len

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
        ppl_all.append(ppl.item())

    return ppl_all


def get_concreteness_score(text):
    # read concreteness ratings
    conc_df_path = "resources/Concreteness_ratings_Brysbaert_et_al_BRM.xlsx"
    conc_df = pd.read_excel(conc_df_path)
    conc_dict = dict(zip(conc_df['Word'], conc_df['Conc.M']))

    doc = nlp(text)
    tokens = [token for token in doc]
    tokens = [token for token in tokens if token.pos_ != 'PUNCT']
    text_as_list = [token.lemma_ if token.lemma_ != '-PRON-' else token.text for token in tokens]
    scores = [conc_dict[word.lower()] for word in text_as_list if word.lower() in conc_dict]
    not_in_dict = [word for word in text_as_list if word.lower() not in conc_dict]

    return pd.Series([np.mean(scores) if scores else np.nan, len(not_in_dict)])


def count_max_depth(text):
    max_depth = 0
    cur_depth = 0
    for char in text:
        if char == "(":
            cur_depth += 1
            max_depth = max(max_depth, cur_depth)
        elif char == ")":
            cur_depth -= 1
    return max_depth


def get_constituency_depth_and_num_sentences(prompts):
    nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
    num_sentences = [0 for _ in range(len(prompts))]
    max_depth = [0 for _ in range(len(prompts))]
    for i, promt in enumerate(prompts):
        doc = nlp(promt[:512])
        sents = list(doc.sents)
        num_sentences[i] = len(sents)
        for sent in sents:
            max_depth_per_sent = count_max_depth(sent._.parse_string)
            max_depth[i] = max(max_depth[i], max_depth_per_sent)
    print(max_depth)
    print(num_sentences)
    return max_depth, num_sentences


def get_magic_words_ratio(prompts, magic_words_set):
    magic_words_ratio = []
    for prompt in prompts:
        if pd.isna(prompt):
            print("error!", prompt)
        prompt_clean = prompt.translate(str.maketrans('', '', string.punctuation))
        prompt_clean = prompt_clean.split()
        if len(prompt_clean) == 0:
            magic_words_ratio.append(0)
            continue
        magic_words = 0
        for word in prompt_clean:
            if word in magic_words_set:
                magic_words += 1
        magic_words_ratio.append(magic_words / len(prompt_clean))
    return magic_words_ratio
