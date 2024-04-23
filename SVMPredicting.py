import joblib
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from Bio import SeqIO


# FASTA file path

#Load the test file and read the sequences
fasta_file = "/Users/ananyaanand/UCLA/Fall 2023/Comp Bio Hack/Files/test.fasta"


sequences_actual = []
labels_actual = []
for record in SeqIO.parse(fasta_file, "fasta"):
    sequences_actual.append(str(record.seq))  # Extract sequence
    labels_actual.append(record.id)  # Extract label (assuming label is the sequence ID)


k = 8  # Adjust the k-mer length as needed
vectorizer = joblib.load("rbf_model_47239K_200K.joblib")
X_ACTUAL = vectorizer.fit_transform(sequences_actual)

svm_model = joblib.load("rbf_model_47239K_200K.joblib")

print("Model loaded")

# Predict labels for the test set using the trained model
Y_ACTUAL = svm_model.predict(X_ACTUAL)

count = 0
for i in Y_ACTUAL:
    if i == "a":
        print(labels_actual[count])
    count = count + 1

