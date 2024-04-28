import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

data_url = 'https://drive.google.com/file/d/1CEql-OEexf9p02M5vCC1RDLXibHYE9Xz/view'
data_path = 'C:\\Users\\umasi\\OneDrive\\Desktop\\coding\\heart_disease_data.csv'

try:
    data = pd.read_csv(data_path, sep=',', encoding='utf-8')
except pd.errors.ParserError as e:
    print(f"ParserError: {e}")

data = pd.read_csv(data_url, sep=',')

X = data.drop('target', axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

plt.figure(figsize=(8, 6))
plt.bar(['Accuracy'], [accuracy], color='blue')
plt.xlabel('Metrics')
plt.ylabel('Value')
plt.title('Model Performance Metrics')
plt.ylim(0, 1)
plt.tight_layout()

plt.text(0, accuracy + 0.02, f'{accuracy:.2f}', ha='center', va='bottom')

plt.show()
