import argparse
import collections
import string

UNK = 'UNK'
PUNC = string.punctuation

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_in', help='File of vectors to map')
    parser.add_argument('--vocab_size', type=int, default=10000, 
                        help='Vocab size to use constructing word map')
    parser.add_argument('--lowercase', default=True, action='store_true',
                        help='True if lowercasing')
    parser.add_argument('--keep_punc', default=False, action='store_true',
                        help='True if keeping punctuation, otherwise stripped')
    parser.add_argument('--wmap', help='Location of wordmap to apply')
    parser.add_argument('--out_dir', default='/tmp', 
                        help='Location to save idx/wmap')
    parser.add_argument('--reverse', default=False, action='store_true', 
                        help='True if reversing wordmap (id to word)')
    return parser.parse_args()

def construct_wmap(f_in, wmap, vocab_size, lowercase, keep_punc):
    # Construct a wordmap from the text file with a given vocab size and 
    # optional lowercasing.
    c = collections.Counter()
    with open(f_in, 'r') as f:
        for line in f:
            if lowercase:
                line = line.lower()
            c.update(line.split())
    if not keep_punc:
        strip_punc(c)
    index = len(wmap)
    for pair in c.most_common(vocab_size - 3):
        wmap[pair[0]] = index
        index += 1

def strip_punc(counter):
    # Remove any punctuation from the wordmap counter
    for punc in PUNC:
        del counter[punc]

def save_wmap(out_dir, wmap):
    # Save wmap
    with open (out_dir + '/wmap', 'w') as f:
        for key in sorted(wmap, key=lambda x: wmap[x]):
            f.write('{} {}\n'.format(key, wmap[key]))

def read_wmap(f_in, wmap):
    # Read wmap file line-by-line and pass to dictionary
    with open(f_in, 'r') as f:
        for line in f:
            tok, index = line.split()
            wmap[tok] = index

def apply_wmap(src, wmap, out_dir, lowercase, keep_punc):
    # Apply wmap to input file line-by-line and save output
    with open(src, 'r') as f_in, open('{}/out.idx'.format(out_dir), 'w') as f_out:
        for line in f_in:
            if lowercase:
                line = line.lower()
            out = []
            for tok in line.split():
                if tok in wmap:
                    out.append(str(wmap[tok]))
                else:
                    if not keep_punc and tok in PUNC:
                        continue
                    else:
                        out.append(str(wmap[UNK]))
            f_out.write(' '.join(out) + '\n')
            print(' '.join(out) + '\n')
    
def reverse_wmap(file_in, out_dir, wmap):
    reverse_wmap = {idx: word for word, idx in wmap.items()}
    with open(file_in, 'r') as f_in, open(out_dir + '/words', 'w') as f_out:
        for line in f_in:
            out = []
            for idx in line.split():
                out.append(reverse_wmap[idx.strip()])
            f_out.write(' '.join([str(tok) for tok in out]) + '\n')

if __name__ == '__main__':
    args = get_args()
    wmap = {'<epsilon>': 0, '<s>': 1, '</s>': 2, UNK: 3}
    if args.wmap:
        read_wmap(args.wmap, wmap)
    if args.reverse:
        reverse_wmap(args.file_in, args.out_dir, wmap)
    else:
        if not args.wmap:
            construct_wmap(args.file_in, wmap, args.vocab_size, args.lowercase, args.keep_punc)
            save_wmap(args.out_dir, wmap)
        apply_wmap(args.file_in, wmap, args.out_dir, args.lowercase, args.keep_punc)
               
