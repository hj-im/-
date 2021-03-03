import json
import os
import argparse
import sys
sys.path.append('coco-caption')

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
# Define a context manager to suppress stdout and stderr.


class suppress_stdout_stderr:
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


class COCOScorer(object):
    def __init__(self):
        print('init COCO-EVAL scorer')

    def score(self, GT, RES, IDs):
        self.eval = {}
        self.imgToEval = {}
        gts = {}
        res = {}
        for ID in IDs:
            #            print ID
            gts[ID] = GT[ID]
            res[ID] = RES[ID]
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            #(Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        eval = {}
        for scorer, method in scorers:
            print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, IDs, m)
                    print("%s: %0.3f" % (m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, IDs, method)
                print("%s: %0.3f" % (method, score))

        # for metric, score in self.eval.items():
        #    print '%s: %.3f'%(metric, score)
        return self.eval

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if imgId not in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score


def score(ref, sample):
    # ref and sample are both dict
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        print('computing %s score with COCO-EVAL...' % (scorer.method()))
        score, scores = scorer.compute_score(ref, sample)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores

def make_json_metric(gts_path,read_path, gt_json_exist = False):

    print('json_making_start')
    if gt_json_exist == False:
        with open(gts_path,'r') as gt:
            gt_k = json.load(gt)
        # print(lengt_k[0])
        gts_dic_k = gt_k['sentences']
        gts_dic = {} 
        count = 0

        id_lst = []
        lst_gt = []
        for i in range(len(gts_dic_k)): # len : 2876*20
            #print(gts_dic_k[i]['video_id'][-4:])
            if i%10000==0:
              print(i)
            if gts_dic_k[i]['video_id'][-4:] not in gts_dic.keys():
                gts_dic[gts_dic_k[i]['video_id'][-4:]] = []

            gts_dic_cap = {}
            gts_dic_cap['image_id'] = gts_dic_k[i]['video_id'][-4:]
            gts_dic_cap['cap_id'] = len(gts_dic[gts_dic_k[i]['video_id'][-4:]])
            gts_dic_cap['caption'] = gts_dic_k[i]['caption']
            gts_dic_cap['tokenized'] = gts_dic_k[i]['caption']

            gts_dic[gts_dic_k[i]['video_id'][-4:]].append(gts_dic_cap)
        
        with open('/content/sample_data/gt.json','w') as gt_js:
            json.dump(gts_dic,gt_js)    

        
        print('Making gts json Success')

    elif gt_json_exist == True:
        with open('/content/sample_data/gt.json','r') as gt_js:
            gts_dic = json.load(gt_js)
    if gts_dic == None:
       print("Building gts_dic is failed!")
       exit()
    print("Gts_dic construction : Success")

    with open(read_path,'r') as cap:
        cap_k = json.load(cap)
    cap_dic = {}
    return_id = []
    for i in range(len(cap_k)):
        cap_dic_cap = {}
        lst = []
        cap_dic_cap['image_id'] = cap_k[i]['file_path'][-8:-4]
        cap_dic_cap['caption'] = cap_k[i]['caption']
        cap_dic[cap_k[i]['file_path'][-8:-4]] = []
        cap_dic[cap_k[i]['file_path'][-8:-4]].append(cap_dic_cap)
        return_id.append(str(cap_k[i]['file_path'][-8:-4]))

    print("Cap_dic construction : Success")    

    print(cap_dic['7790'])
    return gts_dic, cap_dic, return_id

def cocoscorer(gts_path,read_path):
    gts, caps, IDs = make_json_metric(gts_path,read_path, True)
    scorer = COCOScorer()
    #IDes = ['184321']
    scorer.score(gts, caps, IDs)

def get_args():
    parser = argparse.ArgumentParser('Use cocosocorer metric')
    parser.add_argument('--read_path',type=str,default='/content/sample_data/ckpt-19.json')
    parser.add_argument('--gts_path',type=str,default='/content/sample_data/gt.json')
  
    args = parser.parse_args()
    return args
   

if __name__ == '__main__':
    print('1')
    # args = get_args()
    # cocoscorer(args.read_path,args.gts_path)
    cocoscorer('/content/sample_data/gt.json','/content/sample_data/ckpt-5.json')
