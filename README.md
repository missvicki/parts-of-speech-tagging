# parts-of-speech-tagging

Use the first 10k tagged sentences from the Brown corpus to generate the
components of a part-of-speech hidden markov model: the transition
matrix, observation matrix, and initial state distribution. Use the universal
tagset: nltk.corpus.brown.tagged_sents(tagset=’universal’)[:10000]

Also hang on to the mappings between states/observations and indices. Include an OOV observation and smoothing everywhere.


Implement a function viterbi() that takes arguments:
1. obs - the observations [list of ints]
2. pi - the initial state probabilities [list of floats]
3. A - the state transition probability matrix [2D numpy array]
4. B - the observation probability matrix [2D numpy array]
and returns:
1. states - the inferred state sequence [list of ints]

Do everything in log space to avoid underflow.
