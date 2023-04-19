# imports
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import brown
nltk.download('brown')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')

### whole brown corpus ###
corpus = [word for word in brown.words() if any(c.isalpha() for c in word)]

# sorted list for whole corpus
fdist = nltk.FreqDist([w.lower() for w in corpus])
sorted_whole = sorted(fdist, key=fdist.get, reverse=True)
# for word in sorted_whole[:10]:
#     print(word, fdist[word])

### sorted lists for two brown categories ###
lore = brown.words(categories='lore')
lore = [word for word in lore if any(c.isalpha() for c in word)]
adventure = brown.words(categories='adventure')
adventure = [word for word in adventure if any(c.isalpha() for c in word)]

loredist = nltk.FreqDist([w.lower() for w in lore])
advdist = nltk.FreqDist([w.lower() for w in adventure])

sorted_lore = sorted(loredist, key=loredist.get, reverse=True)
sorted_adv = sorted(advdist, key=advdist.get, reverse=True)

# for word in sorted_lore[:10]:
#     print(word, loredist[word])

# for word in sorted_adv[:10]:
#     print(word, advdist[word])

plt.plot(list(fdist.keys()), list(fdist.values()), label='corpus', color='black')  # plot the frequency curve for the corpus
plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.legend()
plt.show()

plt.loglog(list(fdist.keys()), list(fdist.values()), label='corpus', color='black') 
plt.xlabel('Log rank')
plt.ylabel('Log frequency')
plt.legend()
plt.show()