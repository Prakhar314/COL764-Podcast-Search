import gc
import json
import os

import numpy as np
import pandas as pd
from pyserini.search import SimpleSearcher
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import transformers
from transformers import BertTokenizer, BertConfig, TFBertModel
transformers.logging.set_verbosity_error()

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class BM25(object):
    def __init__(self):
        self.searcher = SimpleSearcher('dataset/indexes/sample_collection_jsonl')
        self.k = 20
    
    def search(self,q):
        hits = self.searcher.search(q,k=self.k)
        docids = []
        scores = []
        for hit in hits:
            docids.append(hit.docid)
            scores.append(hit.score)
        results = pd.DataFrame({'docid':docids,'score':scores})
        results = results.sort_values('score', ascending = False).reset_index(drop=True)
        return results

class Pointwise(object):
    
    def create_inputs(self,num_nodes,name):
        layers = []
        for layer_name in ['input_ids','token_type_ids','attention_mask']:
            layers.append(tf.keras.layers.Input(shape=(num_nodes,),dtype=tf.int32,name=layer_name))
        return layers

    def create_model_pointwise(self,output_bias=None):
        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)
        config = BertConfig(hidden_dropout_prob=0.1)
        bert = TFBertModel.from_pretrained('./scripts/bert-model',config=config)
        for layer in bert.layers[:]:
            if isinstance(layer, transformers.models.bert.modeling_tf_bert.TFBertMainLayer):
                layer.embeddings.trainable=False
                layer.pooler.trainable=False
                for idx, layer in enumerate(layer.encoder.layer):
                    # print(layer)
                    # freeze first 10
                    if idx in range(8):
                        layer.trainable = False
            else:
                layer.trainable = False
                
        input_layer = self.create_inputs(512,'pair')
        bert_out = bert(input_layer).last_hidden_state
        cls = tf.keras.layers.Lambda(lambda x:x[:,0,:])(bert_out)
        # print(avg_q.shape)
        output = tf.keras.layers.Dense(1, activation="sigmoid",bias_initializer=output_bias)(cls)
        model = tf.keras.models.Model(inputs=input_layer, outputs=[output])
        # opt,schedule = transformers.create_optimizer(num_train_steps=num_train_steps,init_lr=3e-5,adam_beta1=0.9,adam_beta2=0.999,weight_decay_rate=0.01,num_warmup_steps=10000)
        opt = tf.keras.optimizers.Adam()
        model.compile(optimizer=opt,
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.BinaryAccuracy(),
                        tf.keras.metrics.AUC(curve="ROC")])
        # model.summary()
        return model

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('./scripts/bert-tokenizer')
        self.model = self.create_model_pointwise()
        self.model.load_weights("./scripts/final_model_pointwise.h5")
        self.q_trunc = 64
        self.p_trunc = 512-64+1

    def rerank(self,q,bm25_ranks):
        bm25_ranks['query'] = q
        q = self.tokenizer(bm25_ranks['query'].values.tolist(), return_tensors="tf",padding="max_length",max_length=self.q_trunc,truncation=True)
        p = self.tokenizer(bm25_ranks['content'].values.tolist(), return_tensors="tf",padding="max_length",max_length=self.p_trunc,truncation=True)
        X_test = []
        for id in ['input_ids','token_type_ids','attention_mask']:
            X_test.append(tf.concat([q[id],p[id][:,1:]],1))
        q = None
        p = None
        gc.collect()
        with tf.device('/device:GPU:0'):
            scores = self.model.predict(X_test,batch_size=2,verbose=1)
        bm25_ranks['score'] = scores
        bm25_ranks = bm25_ranks.drop(["query"],axis=1)
        bm25_ranks = bm25_ranks.sort_values('score', ascending = False).reset_index(drop=True)
        return bm25_ranks


class Pairwise(object):
    
    
    def create_inputs(self,num_nodes,name):
        layers = []
        for layer_name in ['input_ids','token_type_ids','attention_mask']:
            layers.append(tf.keras.layers.Input(shape=(num_nodes,),dtype=tf.int32,name=layer_name+f'_{name}'))
        return layers

    def create_model_pairwise(self,output_bias=None):
        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)
        config = BertConfig(hidden_dropout_prob=0.1)
        bert = TFBertModel.from_pretrained('./scripts/bert-model',config=config)

        for layer in bert.layers[:]:
            if isinstance(layer, transformers.models.bert.modeling_tf_bert.TFBertMainLayer):
                layer.embeddings.trainable=False
                layer.pooler.trainable=False
                for idx, layer in enumerate(layer.encoder.layer):
                    # print(layer)
                    # freeze first 10
                    if idx in range(10):
                        layer.trainable = False
            else:
                layer.trainable = False
                
        input_1 = self.create_inputs(512,'pair_1')
        input_2 = self.create_inputs(512,'pair_2')
        bert_out_1 = bert(input_1).last_hidden_state
        bert_out_2 = bert(input_2).last_hidden_state
        cls_1 = tf.keras.layers.Lambda(lambda x:x[:,0,:])(bert_out_1)
        cls_2 = tf.keras.layers.Lambda(lambda x:x[:,0,:])(bert_out_2)
        concated = tf.keras.layers.Concatenate()([cls_1,cls_2])
        # print(avg_q.shape)
        output = tf.keras.layers.Dense(1, activation="sigmoid",bias_initializer=output_bias)(concated)
        model = tf.keras.models.Model(inputs=[input_1,input_2], outputs=[output])
        # opt,schedule = transformers.create_optimizer(num_train_steps=num_train_steps,init_lr=3e-5,adam_beta1=0.9,adam_beta2=0.999,weight_decay_rate=0.01,num_warmup_steps=num_train_steps//10)
        opt = tf.keras.optimizers.Adam(learning_rate=3e-5)
        model.compile(optimizer=opt,
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.BinaryAccuracy(),
                        tf.keras.metrics.AUC(curve="ROC")])
        # model.summary()
        return model

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('./scripts/bert-tokenizer')
        self.model = self.create_model_pairwise()
        self.model.load_weights("./scripts/pairwise_best_model_freeze10.h5")
        self.q_trunc = 64
        self.p_trunc = 512-64+1

    def getRanks(self,results):
        pivot = results.pivot('docid_x','docid_y','score')
        pivot_t = pivot.T
        np.fill_diagonal(pivot.values,1)
        results_ranked = pd.DataFrame((pivot>pivot_t).sum(axis=1) + pivot.mean(axis=1))
        results_ranked = results_ranked.reset_index()
        dropped = results.drop(['docid_y','score','content_y'],axis=1).drop_duplicates()
        results_ranked = results_ranked.merge(dropped,how='left',left_on='docid_x',right_on='docid_x')
        results_ranked.columns = ['docid','score','content']
        return results_ranked
        
    def rerank(self,q,bm25_ranks):
        bm25_ranks['query'] = q
        bm25_ranks = bm25_ranks.merge(bm25_ranks[['query','content','docid']],how='left',left_on='query',right_on='query')
        q = self.tokenizer(bm25_ranks['query'].values.tolist(), return_tensors="tf",padding="max_length",max_length=self.q_trunc,truncation=True)
        p_1 = self.tokenizer(bm25_ranks['content_x'].values.tolist(), return_tensors="tf",padding="max_length",max_length=self.p_trunc,truncation=True)
        p_2 = self.tokenizer(bm25_ranks['content_y'].values.tolist(), return_tensors="tf",padding="max_length",max_length=self.p_trunc,truncation=True)
        X_test = []
        for id in ['input_ids','token_type_ids','attention_mask']:
            X_test.append(tf.concat([q[id],p_1[id][:,1:]],1))
        for id in ['input_ids','token_type_ids','attention_mask']:
            X_test.append(tf.concat([q[id],p_2[id][:,1:]],1))
        q = None
        p_1 = None
        p_2 = None
        gc.collect()
        with tf.device('/device:GPU:0'):
            scores = self.model.predict(X_test,batch_size=2,verbose=1)
        bm25_ranks['score'] = scores
        bm25_ranks = bm25_ranks.drop(["query"],axis=1)
        bm25_ranks = self.getRanks(bm25_ranks).reset_index(drop=True)
        bm25_ranks = bm25_ranks.sort_values('score', ascending = False).reset_index(drop=True)
        return bm25_ranks

def printResults(df,title,original=None):
    print(bcolors.HEADER+"-"*80+title+"-"*80+bcolors.ENDC)
    for i in range(len(df)-1,-1,-1):
        if original is not None:
            print(f'{bcolors.BOLD}{original[original.docid==df.loc[i,"docid"]].index.values[0]+1}->{bcolors.ENDC}',end="")
        print(f'{bcolors.BOLD}{i+1}{bcolors.ENDC}\t{bcolors.OKBLUE}{df.loc[i,"docid"]}{bcolors.ENDC}\t{bcolors.OKCYAN}{df.loc[i,"score"]:.4f}{bcolors.ENDC}')
        print(df.loc[i,'content'])
    print(bcolors.HEADER+"-"*80+title+"-"*80+bcolors.ENDC)

def getContentForDoc(docid):
    ep = docid.split('_')[0].split(':')[-1]
    with open('dataset/output/'+ep+'.json','r') as f:
        cont = json.load(f)
        for i in cont:
            if i['id'] == docid:
                return i['contents']

if __name__=="__main__":
    QUIT = 'q'
    NEW_q = 'n'
    POINT = '2'
    PAIR = '3'
    BM = '1'
    base = BM25()
    pairwise = Pairwise()
    pointwise = Pointwise()
    while True:
        os.system('clear')
        query = input("Enter a query:")
        original_bm = None
        original_pair = None
        original_point = None
        lastCommand = BM
        while True:
            os.system('clear')
            if lastCommand==BM:
                if original_bm is None:
                    original_bm = base.search(query)
                    original_bm['content'] = original_bm['docid'].map(getContentForDoc)
                printResults(original_bm,"BM25")
            elif lastCommand==POINT:
                if original_point is None:
                    original_point = pointwise.rerank(query,original_bm.copy())
                printResults(original_point,"Pointwise",original=original_bm)
            elif lastCommand==PAIR:
                if original_pair is None:
                    original_pair = pairwise.rerank(query,original_bm.copy())
                printResults(original_pair,"Pairwise",original=original_bm)
            else:
                print("unknown command")
            print("Press 1 for BM25")
            print("Press 2 for POINT")
            print("Press 3 for PAIR")
            print("Press n for new query")
            print("Press q to quit")
            lastCommand = input("Command:")
            if lastCommand=='n' or lastCommand=='q':
                break
        if lastCommand=='q':
            break
