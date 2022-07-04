from nltk.corpus import treebank
import numpy as np
from sklearn.model_selection import train_test_split
import nltk

# nltk.download('treebank')
# nltk.download('universal_tagset')

taggeddata = list(treebank.tagged_sents(tagset='universal'))


# print(taggeddata[:2])
# traindata, testdata = train_test_split(
#     taggeddata, train_size=0.8, test_size=0.2, random_state=101)

# traintagged = [tup for sent in traindata for tup in sent]
# testtagged = [tup for sent in testdata for tup in sent]

# tags = {tag for word, tag in traintagged}
# vocab = {word for word, tag in traintagged}


# def emission(word, tag, train_bag=traintagged):
#     taglist = [pair for pair in train_bag if pair[1] == tag]
#     counttag = len(taglist)
#     wgiventaglist = [pair[0] for pair in taglist if pair[0] == word]
#     countwgiventag = len(wgiventaglist)

#     return (countwgiventag, counttag)
