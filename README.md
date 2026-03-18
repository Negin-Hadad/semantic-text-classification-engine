
# Advanced Data Clustering & Semantic NLP Optimizer
This project presents a dual-phase approach to computational intelligence: an iterative **Key Identification-based Clustering** algorithm and an **improved Bag of Words (BoW)** model using Word2Vec-driven synonym mapping to enhance text classification accuracy.

### Key Features
#### 1. Key Identification Clustering
**Iterative Process**: Implements a novel clustering method that iteratively identifies "key" elements within a dataset based on a specific objective function.  
**High Scalability**: Designed to process large-scale datasets efficiently, potentially handling millions of data points in seconds.  
**Custom Distance Metrics**: Utilizes a pairwise distance matrix to represent relationships between data samples.  

#### 2. Semantic NLP Enhancement (Improved BoW)
**Synonym Mapping**: Uses a pre-trained **Word2Vec** model (`GoogleNews-vectors-negative300`) to extract the top five synonyms for words within the dataset.  
**Semantic Consistency**: Replaces various synonyms in user comments with a single "head" word from a generated dictionary to standardize the vocabulary.  
**Performance Gain**: Achieved a **3% accuracy improvement** (reaching ~83.2%) over the baseline Bag of Words model using an **SVM classifier**.  

### Project Structure
**`clustering.py`**: Implementation of the iterative clustering algorithm and objective functions.  
**`word2vec.py`**: Logic for extracting synonym sets and creating the semantic dictionary.  
**`improved.py`**: The enhanced classification pipeline applying the synonym mapping to the dataset.  
**`bag_of_words.py`**: Baseline NLP model for performance comparison.  
**`similar_words.txt`**: The generated dictionary of related semantic terms.  

### Getting Started
#### Prerequisites
* Python 3.x 
* Libraries: `scikit-learn`, `nltk`, `gensim`, `pandas`, `numpy` 

#### Usage
1. **Clustering**: Run `clustering.py` to process the input array and view the generated cluster labels.

2. **Synonym Generation**: Run `word2vec.py` (requires Google News vectors) to generate the `similar_words.txt` dictionary.

3. **Classification**: Run `improved.py` to train the SVM on the semantically enhanced comments and view the final accuracy.
