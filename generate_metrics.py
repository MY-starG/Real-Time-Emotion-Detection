import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.metrics import roc_curve, auc, log_loss, accuracy_score, f1_score
import matplotlib.pyplot as plt

def preprocess_data(df):
    X = []
    y = []
    
    for index, row in df.iterrows():
        # Convert pixel data back to arrays and normalize
        pixels = np.array(row['pixels'].split(), dtype='float32') / 255.0  # Normalize pixel values
        X.append(pixels)
        y.append(row['emotion'])
    
    X = np.array(X)
    y = np.array(y)

    # Reshape for CNN input (48x48 images with 1 color channel)
    X = X.reshape(X.shape[0], 48, 48, 1)

    return X, y

def main():
    # Load the dataset
    df = pd.read_csv(r'C:\Users\mbasi\OneDrive\Desktop\basith\Basith 5 th sem\prorjct 2\my_emotion_dataset.csv')

    # Preprocess the training data
    X, y = preprocess_data(df)
    
    # Encode the labels
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Convert labels to categorical
    from keras.utils import to_categorical
    num_classes = len(label_encoder.classes_)
    y_one_hot = to_categorical(y_encoded, num_classes)

    # Split data into training and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

    # Load the trained model
    model = load_model('my_emotion_model.h5')

    # Evaluate the model
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    accuracy = accuracy_score(y_true, y_pred_classes)
    f1 = f1_score(y_true, y_pred_classes, average='weighted')
    logloss = log_loss(y_true, y_pred)

    # Calculate ROC Curve and AUC for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"F1 Score: {f1:.2f}")
    print(f"Log Loss: {logloss:.2f}")

    # Plot ROC Curve
    plt.figure()
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.savefig('static/roc_curve_my_emotion_model.png')
    plt.close()

    # Plot Log Loss
    epochs = list(range(1, 31))  # Assuming 30 epochs
    log_loss_values = [logloss] * 30  # Simulated log loss values for demonstration
    plt.figure()
    plt.plot(epochs, log_loss_values, label='Log Loss', marker='o')
    plt.title('Log Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Log Loss')
    plt.legend()
    plt.savefig('static/log_loss_my_emotion_model.png')
    plt.close()

if __name__ == "__main__":
    main()