import os
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem import SnowballStemmer

# Descargar recursos de NLTK necesarios
nltk.download('punkt')
nltk.download('stopwords')

# Inicializa el stemmer
stemmer = SnowballStemmer('spanish')

# Función para preprocesar el texto
def preprocess_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [token.lower() for token in tokens]
    tokens = [token for token in tokens if token not in string.punctuation and token not in stopwords.words('spanish')]
    tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(tokens)

# Función para extraer conceptos más significativos usando TF-IDF
def extract_significant_concepts(text, top_n=15):  # Reducido a top_n=10 para mayor relevancia
    try:
        # Cargar stopwords en español desde NLTK
        spanish_stopwords = stopwords.words('spanish')
        
        vectorizer = TfidfVectorizer(
            stop_words=spanish_stopwords,  # Pasar la lista de stopwords en español
            ngram_range=(1, 1),            # Unigramas y bigramas
            min_df=1,                      # Aparecen en al menos 2 documentos
            max_features=top_n             # Limitar a los top_n conceptos más significativos
        )
        
        tfidf_matrix = vectorizer.fit_transform([text])
        scores = tfidf_matrix.toarray().flatten()
        words = vectorizer.get_feature_names_out()

        # Filtrar palabras de longitud mínima (por ejemplo, 4 caracteres o más)
        ranked_words = [(word, score) for word, score in zip(words, scores) if len(word) > 3]
        
        # Ordenar palabras por puntuación de TF-IDF
        ranked_words = sorted(ranked_words, key=lambda x: x[1], reverse=True)
        
        # Devolver las top_n palabras más significativas
        return [word for word, score in ranked_words[:top_n]]
    except ValueError as e:
        print("Error al procesar el texto:", e)
        return []
