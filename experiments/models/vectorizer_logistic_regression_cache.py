from joblib import Memory
from pathlib import Path

class Executable:
    """Helper object to the vectorizer function to be cached.
    
    This object will be instantiated and executed inside the cached function,
    allowing this function to dynamically change itself without losing its
    cache.
    """
    pass

executable = Executable()

def vectorizer(X):
    """Vectorizer function to be cached.
    
    This function will be cached by `joblib.Memory`. Because it calls for an
    object's method, it can be dynamically changed by the object without losing
    its cache."""
    return executable.execute(X)

class VectorizerLogisticRegressionCache:
    """Cache object for `TfidfLogisticRegression` and
    `DistilBertLogisticRegression`.
    
    It caches the vectorization of the model, allowing a speedup in the
    explanation extraction, e.g., Lime's.
    """
    def __init__(self, path=None):
        """Inits class.
        
        Args:
            path (str): Path to the cache directory. If None, the cache
                directory will be created in the current directory.
        """
        if path is not None:
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
        else:
            path = Path().absolute() / "cache"
        self.memory = Memory(location=path, verbose=0)
        self.vectorizer = self.memory.cache(vectorizer)
        self.executable = executable

    def __call__(self, vectorizer, X):
        """Execute a vectorizer function if not cached.

        Every call to this object must be done with the same "formula,"
        although the exact vectorizer function can change. If the return value
        is already cached, it is returned. An example of use of this cache
        object is when one has several `TfidfLogisticRegression` models trained
        over the same dataset: every call to `model.predict_proba` would be
        computed with the same vectorization, so it can be cached to speed up.
        
        Args:
            vectorizer (function): Vectorizer function.
            X (list of str): List of texts to be vectorized.
        """
        self.executable.execute = vectorizer
        return self.vectorizer(X)

    def clear(self):
        """Clear cache."""
        self.memory.clear(warn=False)
