from tqdm import trange, tqdm
import logging
import torch
from eval import eval
from datetime import datetime
import os

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

last_checkpoint_f1_score = 0

def save_model(model, model_name, model_dir, f1_score_result):
    global last_checkpoint_f1_score
    if f1_score_result > last_checkpoint_f1_score:
        weight_dir = os.path.join(model_dir, f"{model_name.replace('/', '-')}.pt") 
        torch.save(model.state_dict(), weight_dir)
        last_checkpoint_f1_score = f1_score_result

def train(train_iter, eval_iter, model, model_name, model_dir, tag_labels, optimizer, scheduler, num_epochs, device):
    logger.info("starting to train")
    max_grad_norm = 1.0  # should be a flag

    for epoch in trange(num_epochs, desc="Epoch"):
        model = model.train()
        tr_loss = 0
        nb_tr_steps = 0
        for step, batch in enumerate(tqdm(train_iter)):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_labels, b_input_mask, b_token_type_ids, b_label_masks = batch
            loss, logits, labels = model(b_input_ids, token_type_ids=b_token_type_ids,
                                         attention_mask=b_input_mask, labels=b_labels,
                                         label_masks=b_label_masks)
            loss.backward()
            tr_loss += loss.item()
            nb_tr_steps += 1
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
        print("Train loss: {}".format(tr_loss / nb_tr_steps))
        logger.info("Train loss: {}".format(tr_loss / nb_tr_steps))
        f1_score_result = eval(eval_iter, model, tag_labels, device)
        save_model(model, model_name, model_dir, f1_score_result)
