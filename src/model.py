from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

class GeneClassifier:
    def __init__(self, model_type="rf"):
        if model_type == "rf":
            self.model = RandomForestClassifier()
        elif model_type == "svm":
            self.model = SVC(kernel='linear')
        else:
            raise ValueError("Choose 'rf' or 'svm'")

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)