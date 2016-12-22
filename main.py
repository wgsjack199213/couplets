# coding:utf-8

import jieba
import pinyin
import re
import numpy as np

def count_words(word, corpus):
    # corpus is a string
    pos = re.findall(r"(?=" + word + ")", corpus)
    return len(pos)

def test_tokenizer():
    #filename = "../对联-唐诗-5言或7言.txt"
    filename = "../chunlian.txt"

    with open(filename) as fin:
        lines = fin.readlines()

    #jieba.enable_parallel()

    counter = 0
    #for k in range(len(lines)):
    for k in range(1000):
        line = lines[k].decode('utf-8')
        [fs, ss] = line.strip().split()
        bps = []    # break positions
        tokens_pair = []
        
        for x in [fs, ss]:
            tokens = list(jieba.cut(x))
            tokens_pair.append(tokens)
            lengths = []
            for tok in tokens:
                lengths.append(len(tok))
            bps.append(np.cumsum(lengths[:-1]))
        if set(bps[0]).issubset(set(bps[1])) or set(bps[1]).issubset(set(bps[0])):
            continue

        if not list(bps[0]) == list(bps[1]):
            '''
            print bps[0], bps[1]
            for sent in tokens_pair:
                for x in sent:
                    print x,
                print '/'
            print ''A
            '''
            counter += 1


    print counter, len(lines)

    training_set = ' '.join(lines)


    
def get_corpus_lines():
    #filename = "../对联-唐诗-5言或7言.txt"
    #filename = "../chunlian.txt"
    filename = "poems_etl.txt"

    with open(filename) as fin:
        lines = fin.readlines()

    return lines

def print_dict(M):
    for x in M:
        print x + ':',
        if type(M[x]) == list:
            for y in M[x]:
                print y,
        else:
            print M[x],
        print ''


def tokenize(lines):
    antithesis_table = {}   # phrase: a list of candidates (with repeatitions)
    word_freq = {}  # word: number of occurrences

    #for k in range(10):
    for k in range(len(lines)):
        line = lines[k].decode('utf-8')
        [fs, ss] = line.strip().split()
        fs_ss = [fs, ss]
        bps = []    # break positions
        #print line

        for x in [fs, ss]:
            tokens = list(jieba.cut(x))
            lengths = []
            for tok in tokens:
                lengths.append(len(tok))
            bps.append(np.cumsum(lengths))
        break_points = [0] + list(set(bps[0]) | set(bps[1]))

        # Extract words
        pair = [[], []]
        for i in range(len(break_points) - 1):
            for bit in range(2):
                x = fs_ss[bit]
                word = x[break_points[i] : break_points[i + 1]]
                pair[bit].append(word)

                # Split the phrase XXX
                if len(word) > 1:
                    for char in word:
                        pair[bit].append(char)
                #print pair
            
        for bit in range(2):
            for j in range(len(pair[0])):
                antithesis_table.setdefault(pair[bit][j], [])
                antithesis_table[pair[bit][j]].append(pair[1 - bit][j])
                word_freq.setdefault(pair[bit][j], 0)
                word_freq[pair[bit][j]] += 1


    #print_dict(word_freq)
    #print_dict(antithesis_table)
    return word_freq, antithesis_table


def ptm_and_iptm(w_fs, w_ss, word_freq, antithesis_table):
    '''
    Compute the phrase translation model (PTM),
    and the iverted phrase translation model (IPTM).
    '''
    #print 'w_fs, w_ss', w_fs, w_ss
    #print 'f_given_s', antithesis_table[w_ss].count(w_fs), len(antithesis_table[w_ss])
    #print 's_given_f', antithesis_table[w_fs].count(w_ss), len(antithesis_table[w_fs])
    p_f_given_s = antithesis_table[w_ss].count(w_fs) * 1.0 / len(antithesis_table[w_ss])
    p_s_given_f = antithesis_table[w_fs].count(w_ss) * 1.0 / len(antithesis_table[w_fs])

    return p_f_given_s, p_s_given_f

def count_colocation(head, tails, corpus):
    k = 0
    count = [0 for i in range(len(tails))]
    while k < len(corpus):
        if corpus[k:k+len(head)] == head:
            for t in range(len(tails)):
                if corpus[k+len(head):k+(len(head) + len(tails[t]))] == tails[t]:
                    count[t] += 1
            k += len(head)
        else:
            k += 1
    return count


def get_full_connect(N_phrase, k, i, candidates, word_freq, antithesis_table, corpus):
    '''
    A character-based trigram language model with Katz back-off
    P(B|A) = Count(AB) / Count(A)
    '''
    count_A = count_words(candidates[k][i], corpus)
    count_AB = []
    
    # The original implementation search the corpus serveral times
    #for w in candidates[k + 1]:
    #    count_AB.append(count_words(candidates[k][i] + w, corpus))
    count_AB = count_colocation(candidates[k][i], candidates[k + 1], corpus)

    #print 'count_AB', count_AB
    #print 'count_AB_another', count_AB_another

    #p_Laplace = (np.array(count_AB) + 1) * 1.0 / (count_A + len(word_freq.keys()))
    lamuda = 0.5
    p_Lid = (np.array(count_AB) + lamuda) / (count_A + len(word_freq.keys()) * lamuda)  # Jeffreys-Perks law

    p_B_given_A = p_Lid
    #print p_B_given_A
    return p_B_given_A


def compute_optimal_trace(N_phrase, candidates, scores, lambda_ptm, lambda_iptm, lambda_bigram, broken_tokens):
    prev_table = [[0 for k in range(len(candidates[i]))] for i in range(N_phrase) ]
    for k in range(N_phrase):
        # Check repeatition
        '''
        same = -1
        for j in range(k):
            if broken_tokens[j] == broken_tokens[k]:
                print 'Same word!', j, 'and', k
            same = j
            break
        '''

        for i in range(len(candidates[k])):
            ptm = np.log(scores[k][i]['ptm'])
            iptm = np.log(scores[k][i]['iptm'])

            if k == 0:
                p = ptm * lambda_ptm + iptm * lambda_iptm
                scores[k][i]['total'] = p
            else:
                max_p, max_w = -99999999999999999.0, 0
                for prev_i in range(len(candidates[k - 1])):
                    bigram = np.log(scores[k-1][prev_i]['full_con'][i])
                    #print 'ptm, iptm, bigram:', ptm, iptm, bigram
                    p = scores[k-1][prev_i]['total'] + ptm * lambda_ptm + iptm * lambda_iptm + bigram * lambda_bigram

                    # Check repeatition
                    '''
                    if same >= 0:
                        current = prev_i
                        for temp in range(k - 1, same - 1, -1):
                            current = prev_table[temp][current]
                        if not candidates[same][current] == candidates[k][i]:
                            p = -9999999999999999.0
                    '''
                    if p > max_p:
                        max_p, max_w = p, prev_i
                prev_table[k][i], scores[k][i]['total'] = max_w, max_p

    result = []
    #print prev_table
    final_scores = []
    for x in scores[N_phrase - 1]:
        final_scores.append(x['total'])
    #print final_scores
    s = np.argmax(final_scores)
    result.append(s)
    for k in range(N_phrase - 1, 0, -1):
        s = prev_table[k][s]
        result.append(s)
    result.reverse()
    sent = ' '.join([candidates[k][result[k]] for k in range(N_phrase)])
    print sent
    return sent



def process_input(word_freq, antithesis_table, corpus):
    #FS = u'风吹湖水动'
    #FS = u'风吹湖水哥斯拉'
    print 'Please input a sentence.'
    FS = raw_input().decode('utf-8')

    if FS.strip() == '':
        return 0, 'Empty input. Please try again.'

    '''
    Step1: Check if all phrases appeared in the first sentences are seen by us.
    '''
    tokens = list(jieba.cut(FS))
    broken_tokens = []
    for tok in tokens:
        if tok in word_freq:
            broken_tokens.append(tok)
        else:
            for char in tok:
                if char in word_freq:
                    broken_tokens.append(char)
                else:
                    return (-1, "Sorry, I have never seen '" + char + "' before.\n" + 
                            "Please try another sentence.")
    N_phrase = len(broken_tokens)

    '''
    Step2: Determine the levels and oblique tones of the last characters.
    '''
    tone_FS = pinyin.get(FS[-1], format='numerical')[-1]
    #print FS[-1], pinyin.get(FS[-1], format='numerical')
    if tone_FS in ['3', '4']:
        tone_SS = ['1', '2']
    else:
        tone_SS = ['3', '4']

    '''
    Step3: Find all candidate counter parts for each phrase in the FS
    '''
    candidates = []
    scores = []
    full_connect = []   # Bigram probabilities between layers
    prev_table = []   # For back-trace
    for k in range(N_phrase):
        cand = list(set(antithesis_table[broken_tokens[k]]))
        if k == N_phrase - 1:
            cand = filter(lambda x: pinyin.get(x[-1], format='numerical')[-1] in tone_SS, cand)
        if cand == []:
            return (-1, 'No proper levels and oblique tones.')

        # Filter out some unusual words XXX
        cand_temp = filter(lambda y: word_freq[y] > 1, cand)
        if len(cand_temp) > 0:
            cand = cand_temp
        #word_rank = [(word_freq[cand[i]], cand[i]) for i in range(len(cand))]
        word_rank = [(antithesis_table[broken_tokens[k]].count(cand[i]), cand[i]) for i in range(len(cand))]
        word_rank = sorted(word_rank, reverse=True)[:40]
        cand = [x[1] for x in word_rank]

        # Filter chars that have exist in the FS
        cand = filter(lambda x: x not in broken_tokens, cand)

        candidates.append(cand)

        scores.append([{} for k in range(len(cand))])
        full_connect.append([{} for k in range(len(cand))])

    for x in range(N_phrase):
        print broken_tokens[x] + ':', 
        for y in candidates[x]:
            #print y + ':' + str(word_freq[y]),
            print y + ':' + str(antithesis_table[y].count(broken_tokens[x])),
        print ""

    '''
    Step4: Compute the P(S|F) for each candidate S in a DP way.
    We use three models here:
        (1) A phrase translation model (PTM), 
        (2) A inverted phrase translation model (IPTM),
        (3) A character-based trigram language model with Katz back-off 
    '''
    print "Now computing the P(S|F) for each candidate S in a DP way."
    for k in range(N_phrase):
        print "Now we come to the No. " + str(k) + "phrase."

        for i in range(len(candidates[k])):
            ptm, iptm = ptm_and_iptm(broken_tokens[k], candidates[k][i], word_freq, antithesis_table)
            scores[k][i]['ptm'] = ptm
            scores[k][i]['iptm'] = iptm
            #print candidates[k][i], ptm, iptm
            # Markov bigram model
            if k < N_phrase - 1:
                #pass
                scores[k][i]['full_con'] = get_full_connect(N_phrase, k, i, candidates, word_freq,\
                                                        antithesis_table, corpus)
    
    lambda_ptm = 0.001
    lambda_iptm = 1.0
    lambda_bigram = 1.0


    '''
    Step5: Aggregate the P(S|F) for each model, and compute the optimal trace.
    '''
    print "Now back-tracing to get several output sentences."
    # Dynamic programming by traverse the layers


    results = set([])
    for lambda_bigram in [0.001, 0.01, 0.1, 1.0, 10, 100, 1000]:
        for lambda_ptm in [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]:
            print lambda_bigram, lambda_ptm
            sent = compute_optimal_trace(N_phrase, candidates, scores, lambda_ptm, lambda_iptm, lambda_bigram, broken_tokens)
            results.add(sent)
    print '================='
    for l in results:
        print l


    return (0, ' '.join(broken_tokens))


if __name__ == '__main__':
    #test_tokenizer()
    '''
    # Training 
    '''
    # Read in training set
    corpus = get_corpus_lines()
    # Tokenization
    word_freq, antithesis_table = tokenize(corpus)

    while True:
        return_code, message = process_input(word_freq, antithesis_table, ' '.join(corpus).decode('utf-8'))
        #if return_code < 0:
        print message
        #break

    





