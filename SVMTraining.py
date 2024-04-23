import joblib
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from Bio import SeqIO

sequences = []
labels = []
MAX_RECORDS_ACCESSIBLE = 47239
MAX_RECORDS_NOT_ACCESSIBLE = 200000
count = 0

# Extract records from accessible.fasta file
fasta_file = "/Users/ananyaanand/UCLA/Fall 2023/Comp Bio Hack/Files/accessible.fasta"
for record in SeqIO.parse(fasta_file, "fasta"):
    sequences.append(str(record.seq))  # Extract sequence
    labels.append(record.id[0])  # Extract label (assuming label is the sequence ID)
    count = count + 1
    if count > MAX_RECORDS_ACCESSIBLE:
        break

# Extract records from notaccessible.fasta file
count = 0
fasta_file = "/Users/ananyaanand/UCLA/Fall 2023/Comp Bio Hack/Files/notaccessible.fasta"
for record in SeqIO.parse(fasta_file, "fasta"):
    sequences.append(str(record.seq))  # Extract sequence
    labels.append(record.id[0])  # Extract label
    count = count + 1
    if count > MAX_RECORDS_NOT_ACCESSIBLE:
        break

# Print sequences and their corresponding labels
# for i in range(len(sequences)):
#     print(f"Sequence: {sequences[i]}, Label: {labels[i]}")


# Convert sequences into k-mer counts
k = 8  
vectorizer = CountVectorizer(analyzer='char', ngram_range=(k, k))
X = vectorizer.fit_transform(sequences)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=0)

print("Training data : ",X_train.shape)
print("Training data : ",X_test.shape)

# Initialize SVM classifier
svm_model = svm.SVC(kernel='rbf')

# Train the SVM model
svm_model.fit(X_train, y_train)

# Predict labels for the test set that contains both accessible and not accessible sequenecs
y_pred = svm_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Save the trained model and vectorizer to disk

joblib.dump(svm_model, "rbf_model_47239K_200K.joblib")
joblib.dump(vectorizer, "rbf_vectorizer_47239K_200K.joblib")

# PREDICTION COMPLETE
