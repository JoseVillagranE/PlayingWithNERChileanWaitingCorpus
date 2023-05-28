from tqdm import tqdm
import torch
import torch.nn.functional as F
import logging
from seqeval.metrics import accuracy_score, f1_score, classification_report

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def eval(iter_data, model, tag_labels, device):
    logger.info("starting to evaluate")
    model = model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps = 0
    predictions, true_labels = [], []
    for batch in tqdm(iter_data):
        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_labels, b_input_mask, b_token_type_ids, b_label_masks = batch

        with torch.no_grad():
            tmp_eval_loss, logits, reduced_labels = model(b_input_ids,
                                                          token_type_ids=b_token_type_ids,
                                                          attention_mask=b_input_mask,
                                                          labels=b_labels,
                                                          label_masks=b_label_masks)

        logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
        logits = logits.detach().cpu().numpy()
        reduced_labels = reduced_labels.to('cpu').numpy()

        labels_to_append = []
        predictions_to_append = []

        for prediction, r_label in zip(logits, reduced_labels):
            preds = []
            labels = []
            for pred, lab in zip(prediction, r_label):
                if lab.item() == -1:  # masked label; -1 means do not collect this label
                    continue
                preds.append(pred)
                labels.append(lab)
            predictions_to_append.append(preds)
            labels_to_append.append(labels)

        predictions.extend(predictions_to_append)
        true_labels.append(labels_to_append)

        eval_loss += tmp_eval_loss.mean().item()

        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    logger.info("Validation loss: {}".format(eval_loss))
    pred_tags = [tag_labels[p_i] for p in predictions for p_i in p]
    valid_tags = [tag_labels[l_ii] for l in true_labels for l_i in l for l_ii in l_i]

    f1_score_result = f1_score([valid_tags], [pred_tags])

    logger.info("Seq eval accuracy: {}".format(accuracy_score(valid_tags, pred_tags)))
    logger.info("F1-Score: {}".format(f1_score_result))
    logger.info("Classification report: -- ")
    logger.info(classification_report([valid_tags], [pred_tags]))

    return f1_score_result