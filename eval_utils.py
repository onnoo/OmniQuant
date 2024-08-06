import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'  # suppress lm_eval warning

import torch
import torch.nn as nn
import lm_eval
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

from eval_model import CacheHFLM


def evaluate(model,
             tokenizer,
             tasks,
             limit=None,
             max_length=None,
             prefix_ids=None,
             past_key_values=None):
    """
    tasks=['piqa', 'lambada_openai', 'wikitext', 'hellaswag', 'winogrande']
    past_key_values overrides prefix_ids.
    """

    device = next(model.parameters()).device

    lm = CacheHFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=1,
        max_batch_size=1,
        max_length=max_length,
    )

    if past_key_values:
        lm.set_past_key_values(past_key_values)
    elif prefix_ids:
        prefix_ids = torch.LongTensor(prefix_ids).unsqueeze(0).to(device)
        lm.set_prefix_ids(prefix_ids)
    
    # lm_eval.tasks.initialize_tasks('./tasks/')
    task_dir = os.path.dirname(os.path.abspath(__file__)) + '/tasks/'
    lm_eval.tasks.include_path(task_dir)

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        limit=limit,
        log_samples=False,
        write_out=False,
    )

    return results


@torch.no_grad()
def benchmark(model, dataset, past_key_values=None):
    model.eval()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    if past_key_values is not None:
        past_key_value_len = past_key_values[0][0].size(2)
        dataset = dataset[:, past_key_value_len:]
    
    # Warmup (slow)
    input_ids = dataset[0].unsqueeze(0).cuda()
    outputs = model(input_ids, past_key_values=past_key_values)
    
    times = []
    max_memory = 0
    
    for input_ids in tqdm(dataset, ncols=80, ascii=True, desc='benchmark'):
        input_ids = input_ids.unsqueeze(0).cuda()
        torch.cuda.synchronize()
        start.record()
        outputs = model(input_ids, past_key_values=past_key_values)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

        max_memory = max(max_memory, torch.cuda.memory_allocated() / 1024 / 1024)

    results = {
        'times': times,
        'max_memory': max_memory
    }
    return results


def evaluate_c4(model,
                tokenizer,
                seqlen=2000,
                prefix_ids=None,
                past_key_values=None):
    """
    past_key_values overrides prefix_ids.
    """

    device = next(model.parameters()).device

    lm = CacheHFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=1,
        max_batch_size=1,
        max_length=seqlen,
    )

    if past_key_values:
        lm.set_past_key_values(past_key_values)
    elif prefix_ids:
        prefix_ids = torch.LongTensor(prefix_ids).unsqueeze(0).to(device)
        lm.set_prefix_ids(prefix_ids)

    testenc = get_c4_testset(seqlen=seqlen, tokenizer=tokenizer)

    nsamples = testenc.numel() // seqlen
    lm.model.eval()

    nlls = []
    for i in tqdm(range(nsamples)):
        batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to(device)
        # outputs = lm.model.model(batch)
        # hidden_states = outputs[0]
        # logits = lm.model.lm_head(hidden_states)
        logits = lm._model_call(batch)
        shift_logits = logits[:, :-1, :]
        shift_labels = testenc[:, (i * lm.seqlen) : ((i + 1) * lm.seqlen)][
            :, 1:
        ].to(lm.model.lm_head.weight.device)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        neg_log_likelihood = loss.float() * lm.seqlen
        nlls.append(neg_log_likelihood)
    
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * lm.seqlen))
    print(f'C4 PPL : {ppl.item()}')

    return { 'results': { 'c4.ppl': ppl.item() } }


def get_c4_testset(seqlen, tokenizer):
    import random

    print("get_c4")
    # traindata = load_dataset(
    #     'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    # )
    valdata = load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )

    # random.seed(seed)
    # trainloader = []
    # for _ in range(nsamples):
    #     while True:
    #         i = random.randint(0, len(traindata) - 1)
    #         trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
    #         if trainenc.input_ids.shape[1] >= seqlen:
    #             break
    #     i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
    #     j = i + seqlen
    #     inp = trainenc.input_ids[:, i:j]
    #     tar = inp.clone()
    #     tar[:, :-1] = -100
    #     trainloader.append((inp, tar))

    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)

    return valenc
