import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest

# Load the language model
nlp = spacy.load('en_core_web_sm')

# Input text
introduction_text = (
    'London is the capital of the UK, '
    'with a population of nearly 9 million people. '
    'London is one of the most diverse cities in the world. '
    'London has over 100 museums, galleries and exhibitions. '
    'London has 40 universities and higher education institutions. '
    'London has over 15,500 restaurants, serving Italian, Indian, Thai and Chinese cuisines. '
    'London is also one of the world\'s capitals of finance, fashion, arts and entertainment.'
)

doc = nlp(introduction_text)
print("# Sentences =====================================================")
sentences = list(doc.sents)
print(f"Total sentences: {len(sentences)}")
for i, sent in enumerate(sentences, 1):
    print(f"{i}. {sent}")

print("\n# Token Analysis ===============================================")
print(f"Total tokens: {len(doc)}")
print("First 10 tokens with indices:")
for token in doc[:10]:
    print(f"  '{token.text}' (idx: {token.idx}, POS: {token.pos_})")

non_stop_tokens = [token.text for token in doc if not token.is_stop]
print(f"\nNon-stop tokens: {len(non_stop_tokens)}")
print(f"Sample: {non_stop_tokens[:10]}")

print("\n# Lemmatization Examples =======================================")
for token in doc[:15]:  # Show first 15 examples
    if token.text != token.lemma_:
        print(f"  '{token.text}' → '{token.lemma_}'")

print("\n# Word Frequency ===============================================")
words = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]
word_freq = Counter(words)
print("Top 5 most common words:")
for word, freq in word_freq.most_common(5):
    print(f"  '{word}': {freq}")


unique_words = [word for word, freq in word_freq.items() if freq == 1]
print(f"\nUnique words (appear only once): {len(unique_words)}")
print(f"Sample unique words: {unique_words[:10]}")


print("\n# Summary Sentences ============================================")
pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']
stopwords = list(STOP_WORDS)
keywords = [token.text for token in doc if token.pos_ in pos_tag and token.text.lower() not in stopwords and token.text not in punctuation]
freq_word = Counter(keywords)

sent_strength = {}
for sent in doc.sents:
    for word in sent:
        if word.text in freq_word:
            sent_strength[sent] = sent_strength.get(sent, 0) + freq_word[word.text]

summarized_sentences = nlargest(3, sent_strength, key=sent_strength.get)
for sent in summarized_sentences:
    print(f"- {sent.text}")
