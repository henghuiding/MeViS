###########################################################################
# Created by: NTU
# Email: heshuting555@gmail.com
# Copyright (c) 2023
###########################################################################

import pickle


ckpt = pickle.load(open('model_final_86143f.pkl', 'rb'))

ckpt['model']['sem_seg_head.predictor.query_embed.weight'] = ckpt['model']['sem_seg_head.predictor.query_embed.weight'][:20]
ckpt['model']['sem_seg_head.predictor.static_query.weight'] = ckpt['model']['sem_seg_head.predictor.static_query.weight'][:20]
with open("model_final_86143f.pkl", 'wb') as file:
       pickle.dump(ckpt, file)