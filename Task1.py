
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Iris.csv')

if 'Id' in df.columns:
    df = df.drop('Id', axis=1)

le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])  # setosa -> 0, versicolor -> 1, virginica -> 2

X = df.drop('Species', axis=1)  
y = df['Species']              

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("✅ Model Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Detailed Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('🔍 Confusion Matrix - Iris Classification')
plt.tight_layout()
plt.show()
