from collections import Counter
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
from nltk.tokenize import sent_tokenize, word_tokenize
from kneed import KneeLocator
from bertsummarizer.tools import paragraph_formater
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk import download

download('stopwords')
download('punkt')


class BertSummarizer():

    def __init__(self, max_clusters: int = 100, model_name='neuralmind/bert-base-portuguese-cased'):
        self.embedder = SentenceTransformer(model_name)       
        self.max_clusters = max_clusters

    def extract_central_paragraphs(self, text: str, clusters: int = 2):
        paragraphs = paragraph_formater.format(text.split('\n\n'))
        paragraph_embeddings = self.embedder.encode([self._normalize(p) for p in paragraphs], convert_to_tensor=True)
        model = self.__train_model(paragraph_embeddings)
        self.model = model
        sentences_groups = model.predict(paragraph_embeddings)
        
        paragraph_2_group = {}
        for i in range(len(paragraphs)):
            paragraph_2_group[paragraphs[i]] = sentences_groups[i]

        groups = Counter(sentences_groups)    
        common =  set([x[0] for x in groups.most_common(clusters)])       

        summary = []
        for x in range(len(paragraphs)):
            if paragraph_2_group[paragraphs[x]] in common:
                summary.append(paragraphs[x])      
        self.summary = ' '.join(summary).replace('\n\n', '. ').replace('\n', '. ')        
        return summary            

    def extract_central_sentences(self, text: str, clusters: int = 2):
        sentences = paragraph_formater.format(text.split('\n\n'))
        sentence_tokens = []
        for p in sentences:
            sentence_tokens.extend(sent_tokenize(p, language='portuguese'))

        sentence_embeddings = self.embedder.encode([self._normalize(p) for p in sentence_tokens], convert_to_tensor=True)

        model = self.__train_model(sentence_embeddings)
        self.model = model
        sentences_centroids = model.predict(sentence_embeddings)
        
        sentence_2_group = {}
        for i in range(len(sentence_tokens)):
            sentence_2_group[sentence_tokens[i]] = sentences_centroids[i]

        groups = Counter(sentences_centroids)           
        common =  set([x[0] for x in groups.most_common(clusters)])

        summary = []
        for p in sentences:
            pf = ''
            for s in sent_tokenize(p, language='portuguese'):
                group = sentence_2_group[s]
                if group in common:
                    pf += s + ' '
            if len(pf) > 0:
                summary.append(pf.strip())        
        self.summary = ' '.join(summary).replace('\n\n', '. ').replace('\n', '. ')        
        return summary                 

    def extract_key_paragraphs(self, text: str, extractions_per_cluster: int = 1, min_cluster_freq: int = 3):
        paragraphs = paragraph_formater.format(text.split('\n\n'))
        paragraph_embeddings = self.embedder.encode([self._normalize(x) for x in paragraphs], convert_to_tensor=True)
        model = self.__train_model(paragraph_embeddings)
        self.model = model
        sentences_centroids = model.predict(paragraph_embeddings)
        paragraph_2_cluster = {}
        for i in range(len(paragraphs)):
            paragraph_2_cluster[paragraphs[i]] = sentences_centroids[i]
        cluster_freq = Counter(sentences_centroids)        

        centroids = torch.tensor(model.cluster_centers_)        

        closest_n = self._get_closest_n(centroids, paragraph_embeddings, extractions_per_cluster)
        summary = []
        for i in closest_n:
            if cluster_freq[paragraph_2_cluster[paragraphs[i]]] >= min_cluster_freq:
                summary.append(paragraphs[i])
        self.summary = ' '.join(summary).replace('\n\n', '. ').replace('\n', '. ')        

        return summary     

    def extract_key_sentences(self, text: str, extractions_per_cluster: int = 1, min_cluster_freq: int = 3):
        self.text = text
        paragraphs = paragraph_formater.format(text.split('\n\n'))  
        paragraphs = sent_tokenize(' '.join(paragraphs), language='portuguese')
        paragraph_embeddings = self.embedder.encode([self._normalize(x) for x in paragraphs], convert_to_tensor=True)

        model = self.__train_model(paragraph_embeddings)
        self.model = model
        sentences_centroids = model.predict(paragraph_embeddings)
        paragraph_2_cluster = {}
        for i in range(len(paragraphs)):
            paragraph_2_cluster[paragraphs[i]] = sentences_centroids[i]
        cluster_freq = Counter(sentences_centroids)           

        centroids = torch.tensor(model.cluster_centers_)

        closest_n = self._get_closest_n(centroids, paragraph_embeddings, extractions_per_cluster)
        summary = []
        for i in closest_n:
            if cluster_freq[paragraph_2_cluster[paragraphs[i]]] >= min_cluster_freq:
                summary.append(paragraphs[i])

        self.summary = ' '.join(summary).replace('\n\n', '. ').replace('\n', '. ')
        
        return summary            

    def sumary_wordcloud(self):
        stop_words = set(stopwords.words('portuguese'))
        words = []
        for w in word_tokenize(self.summary.lower(), language='portuguese'):
            if w not in stop_words and w.isnumeric() == False:
                words.append(w)
        wordcloud = WordCloud(background_color = 'white').generate(' '.join(words))
        return wordcloud

    def _get_closest_n(self, centroids, embeddings, n=5):
        closest_n = []
        for i in range(len(centroids)):
            distances = util.pytorch_cos_sim(centroids[i].double(), embeddings.double())[0]
            values, indices = distances.topk(k=n)
            closest_n.extend(indices)
        # tensor to int
        return sorted(set([x.item() for x in closest_n]))

    def __train_model(self, x):
        if self.max_clusters > len(x):
            self.max_clusters = len(x) - 1
        inertia = []
        for n in range(1, self.max_clusters+1):
            kmeans = KMeans(n_clusters=n, random_state=42, verbose=False)
            kmeans.fit(x)
            inertia.append(kmeans.inertia_)
        x_axis = range(1, len(inertia)+1)
        kl = KneeLocator(x_axis, inertia, curve='convex', direction='decreasing')
        knee_point = kl.knee
        kmeans = KMeans(n_clusters=knee_point, random_state=42, verbose=False)
        kmeans.fit(x)
        return kmeans     

    def _normalize(self, text: str):
        words = word_tokenize(text, language='portuguese')
        words = [w for w in words if w.isnumeric() == False] 
        return ' '.join(words)             

