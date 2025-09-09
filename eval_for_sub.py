import json
import re
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, \
    confusion_matrix


def extract_label(text):
    """Extract the label from the output text."""
    # Match "Label: 0" or "Label: 1"
    match = re.search(r'Label:\s*(\d)', text)
    if match:
        return int(match.group(1))

    # If the standard format is not matched, try other possible formats
    if '1' in text:
        return 1
    elif '0' in text:
        return 0
    else:
        return 0  # Default to non-hate speech


def load_predictions(file_path):
    """Load the prediction results file."""
    predictions = []
    labels = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())

            # Extract the predicted label and the true label
            pred_label = extract_label(data['predict'])
            true_label = extract_label(data['label'])

            predictions.append(pred_label)
            labels.append(true_label)

    return predictions, labels


def evaluate_model(predictions, labels):
    """Calculate model evaluation metrics."""

    # Basic metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='binary')
    recall = recall_score(labels, predictions, average='binary')
    f1 = f1_score(labels, predictions, average='binary')

    # Macro-average and micro-average (for binary classification, micro-average equals accuracy)
    precision_macro = precision_score(labels, predictions, average='macro')
    recall_macro = recall_score(labels, predictions, average='macro')
    f1_macro = f1_score(labels, predictions, average='macro')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro
    }


def main():
    # Change this to the path of your prediction results file
    file_path = 'D:/Pycharm/jargon_detcection/train/eval/eval_2025-09-07-algo4_test/generated_predictions.jsonl'

    try:
        print("Loading prediction results...")
        predictions, labels = load_predictions(file_path)

        print(f"Successfully loaded {len(predictions)} samples")
        print(f"True label distribution: 0={labels.count(0)}, 1={labels.count(1)}")
        print(f"Predicted label distribution: 0={predictions.count(0)}, 1={predictions.count(1)}")

        # Calculate evaluation metrics
        results = evaluate_model(predictions, labels)

        # Print the results
        print("\n" + "=" * 50)
        print("Hate Speech Detection Model Evaluation Results")
        print("=" * 50)
        print(f"Accuracy:              {results['accuracy']:.4f}")
        print(f"Precision:             {results['precision']:.4f}")
        print(f"Recall:                {results['recall']:.4f}")
        print(f"F1-Score:              {results['f1']:.4f}")
        print("-" * 30)
        print("Macro-Averaged Metrics:")
        print(f"Precision (Macro):     {results['precision_macro']:.4f}")
        print(f"Recall (Macro):        {results['recall_macro']:.4f}")
        print(f"F1-Score (Macro):      {results['f1_macro']:.4f}")

        # Detailed classification report
        print("\n" + "=" * 50)
        print("Detailed Classification Report:")
        print("=" * 50)
        print(classification_report(labels, predictions,
                                    target_names=['Not Hate Speech', 'Hate Speech'],
                                    digits=4))

        # Confusion matrix
        print("Confusion Matrix:")
        cm = confusion_matrix(labels, predictions)
        print(f"True\\Pred    0    1")
        print(f"0         {cm[0, 0]:4d} {cm[0, 1]:4d}")
        print(f"1         {cm[1, 0]:4d} {cm[1, 1]:4d}")

        # Calculate some additional metrics
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

        print(f"\nSpecificity: {specificity:.4f}")
        print(f"Sensitivity (Recall): {sensitivity:.4f}")

        # Save the results
        output_file_path = 'eval/eval_2025-09-07-algo4_test/lora4_test_results.txt'
        with open(output_file_path, 'w', encoding='utf-8') as f:

            f.write("Hate Speech Detection Model Evaluation Results\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total Samples: {len(predictions)}\n")
            f.write(f"Accuracy: {results['accuracy']:.4f}\n")
            f.write(f"Precision: {results['precision']:.4f}\n")
            f.write(f"Recall: {results['recall']:.4f}\n")
            f.write(f"F1-Score: {results['f1']:.4f}\n")
            f.write("\nDetailed Classification Report:\n")
            f.write(classification_report(labels, predictions,
                                          target_names=['Not Hate Speech', 'Hate Speech']))

        print(f"\nResults have been saved to {output_file_path}")

    except FileNotFoundError:
        print(f"Error: File not found '{file_path}'")
        print("Please make sure the file path is correct and the file exists")
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")


if __name__ == "__main__":
    main()
