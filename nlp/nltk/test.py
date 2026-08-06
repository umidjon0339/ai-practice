"""
===========================================================
NLTK CORPORA COMPLETE EXAMPLE
===========================================================

This script demonstrates:

1. Brown Corpus
2. Gutenberg Corpus
3. Reuters Corpus
4. Movie Reviews Corpus
5. WordNet

Topics Covered
--------------
✓ Download corpora
✓ Load corpora
✓ Explore categories
✓ Count words
✓ Count sentences
✓ Vocabulary size
✓ Frequency Distribution
✓ Read documents
✓ Sentiment dataset
✓ Synonyms
✓ Definitions
✓ Hypernyms
✓ Hyponyms

===========================================================
"""

# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

import nltk
from nltk.corpus import brown
from nltk.corpus import gutenberg
from nltk.corpus import reuters
from nltk.corpus import movie_reviews
from nltk.corpus import wordnet
from nltk import FreqDist

# ==========================================================
# DOWNLOAD REQUIRED DATASETS
# (Run once. Afterwards they are stored locally.)
# ==========================================================

nltk.download("brown")
nltk.download("gutenberg")
nltk.download("reuters")
nltk.download("movie_reviews")
nltk.download("wordnet")
nltk.download("omw-1.4")

print("=" * 70)
print("NLTK CORPORA DEMONSTRATION")
print("=" * 70)

###########################################################################
# 1. BROWN CORPUS
###########################################################################

print("\n")
print("=" * 70)
print("1. BROWN CORPUS")
print("=" * 70)

# Display all available categories
print("\nCategories:")
print(brown.categories())

# Total number of documents
print("\nTotal documents:")
print(len(brown.fileids()))

# Total words
print("\nTotal words:")
print(len(brown.words()))

# Total sentences
print("\nTotal sentences:")
print(len(brown.sents()))

# Show first 30 words
print("\nFirst 30 words:")
print(brown.words()[:30])

# Load only NEWS category
news_words = brown.words(categories="news")

print("\nWords in NEWS category:")
print(len(news_words))

# Frequency distribution
fd = FreqDist(word.lower() for word in news_words if word.isalpha())

print("\nTop 20 most common words:")
print(fd.most_common(20))

###########################################################################
# 2. GUTENBERG CORPUS
###########################################################################

print("\n")
print("=" * 70)
print("2. GUTENBERG CORPUS")
print("=" * 70)

# Available books
print("\nBooks:")
print(gutenberg.fileids())

# Load Emma
book = gutenberg.words("austen-emma.txt")

print("\nTotal words:")
print(len(book))

# Unique vocabulary
print("\nUnique vocabulary:")
print(len(set(book)))

# First 50 words
print("\nFirst 50 words:")
print(book[:50])

# Sentences
sentences = gutenberg.sents("austen-emma.txt")

print("\nNumber of sentences:")
print(len(sentences))

# Average sentence length
average = len(book) / len(sentences)

print("\nAverage sentence length:")
print(round(average, 2))

###########################################################################
# 3. REUTERS CORPUS
###########################################################################

print("\n")
print("=" * 70)
print("3. REUTERS CORPUS")
print("=" * 70)

# Number of news articles
print("\nTotal articles:")
print(len(reuters.fileids()))

# Available categories
print("\nFirst 20 categories:")
print(reuters.categories()[:20])

# Select one article
article_id = reuters.fileids()[0]

print("\nArticle ID:")
print(article_id)

# Categories of article
print("\nArticle categories:")
print(reuters.categories(article_id))

# Read article
article_words = reuters.words(article_id)

print("\nFirst 60 words:")
print(article_words[:60])

###########################################################################
# 4. MOVIE REVIEWS
###########################################################################

print("\n")
print("=" * 70)
print("4. MOVIE REVIEWS")
print("=" * 70)

# Positive and negative labels
print("\nCategories:")
print(movie_reviews.categories())

# Total reviews
print("\nTotal reviews:")
print(len(movie_reviews.fileids()))

# Positive reviews
print("\nPositive reviews:")
print(len(movie_reviews.fileids("pos")))

# Negative reviews
print("\nNegative reviews:")
print(len(movie_reviews.fileids("neg")))

# Read one positive review
review_id = movie_reviews.fileids("pos")[0]

print("\nReview ID:")
print(review_id)

review = movie_reviews.raw(review_id)

print("\nFirst 600 characters:")
print(review[:600])

###########################################################################
# 5. WORDNET
###########################################################################

print("\n")
print("=" * 70)
print("5. WORDNET")
print("=" * 70)

# Search the word "computer"
synsets = wordnet.synsets("computer")

print("\nAvailable Synsets:")
print(synsets)

# Select first meaning
syn = synsets[0]

print("\nName:")
print(syn.name())

print("\nDefinition:")
print(syn.definition())

print("\nExample sentences:")
print(syn.examples())

print("\nSynonyms:")
print(syn.lemma_names())

print("\nHypernyms (more general concepts):")
print(syn.hypernyms())

print("\nHyponyms (more specific concepts):")
print(syn.hyponyms()[:10])

###########################################################################
# 6. SIMPLE TEXT ANALYSIS
###########################################################################

print("\n")
print("=" * 70)
print("6. SIMPLE TEXT ANALYSIS")
print("=" * 70)

# Lowercase words
words = [
    word.lower()
    for word in brown.words(categories="news")
    if word.isalpha()
]

print("\nTotal cleaned words:")
print(len(words))

print("\nVocabulary size:")
print(len(set(words)))

fd = FreqDist(words)

print("\nTop 30 frequent words:")
print(fd.most_common(30))

###########################################################################
# 7. LONGEST WORDS
###########################################################################

print("\n")
print("=" * 70)
print("7. LONGEST WORDS")
print("=" * 70)

unique_words = set(words)

longest = sorted(unique_words, key=len, reverse=True)

print("\nTop 20 longest words:")
print(longest[:20])

###########################################################################
# 8. WORD SEARCH
###########################################################################

print("\n")
print("=" * 70)
print("8. SEARCH WORD")
print("=" * 70)

search_word = "government"

count = words.count(search_word)

print(f"\n'{search_word}' appears {count} times.")

###########################################################################
# 9. WORD LENGTH STATISTICS
###########################################################################

print("\n")
print("=" * 70)
print("9. WORD LENGTH")
print("=" * 70)

lengths = [len(word) for word in words]

print("Shortest word length :", min(lengths))
print("Longest word length  :", max(lengths))
print("Average word length  :", round(sum(lengths) / len(lengths), 2))

###########################################################################
# 10. SUMMARY
###########################################################################

print("\n")
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print("""
Brown Corpus
------------
General English
Good for language analysis.

Gutenberg Corpus
----------------
Classic books.
Good for literary analysis.

Reuters Corpus
--------------
News articles.
Good for text classification.

Movie Reviews
--------------
Positive & Negative reviews.
Good for sentiment analysis.

WordNet
--------
Dictionary + semantic relations.
Good for synonyms, meanings,
hypernyms, hyponyms, NLP semantics.
""")

print("=" * 70)
print("END OF DEMONSTRATION")
print("=" * 70)


# Sample text
text = "play is good"

# Tokenize the text into words
tokens = nltk.word_tokenize(text)

# Perform POS tagging
pos_tags = nltk.pos_tag(tokens)

# Print the results
print(pos_tags)