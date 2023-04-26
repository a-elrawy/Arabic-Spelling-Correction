from data_loader import unprep

def get_metrics(predicted_batch, target_batch):
    # Compute character-level accuracy
    predicted_batch = [unprep(text) for text in predicted_batch ]
    target_batch = [unprep(text) for text in target_batch ]

    total_chars = sum(len(target) for target in target_batch)
    correct_chars = sum(1 for predicted, target in zip(predicted_batch, target_batch) for i in range(min(len(predicted), len(target))) if predicted[i] == target[i])
    char_accuracy = correct_chars / total_chars

    # Compute precision, recall, and F1-score for the entire batch
    predicted_words_batch = [predicted.split() for predicted in predicted_batch]
    target_words_batch = [target.split() for target in target_batch]

    tp = 0
    fp = 0
    fn = 0

    for predicted_words, target_words in zip(predicted_words_batch, target_words_batch):
        # Compute true positives
        tp += sum(1 for word in predicted_words if word in target_words)

        # Compute false positives
        fp += sum(1 for word in predicted_words if word not in target_words)

        # Compute false negatives
        fn += sum(1 for word in target_words if word not in predicted_words)

    precision = tp / (tp + fp)
    recall = 0 if tp + fn == 0 else tp / (tp + fn)
    f1_score = 0 if precision + recall == 0 else \
        2 * precision * recall / (precision + recall)

    return char_accuracy, precision, recall, f1_score
