from typing_extensions import final
import numpy as np
import os
import timm
import time


print(time.strftime('%Y%m%d-%H%M%S'))



#print(timm.list_models('*efficientnet*'))

'''
preds = np.zeros((100, 29), dtype=np.float32)
auc_score = [1]*29
map_score = [0]*29

msg = '----- Multilabel scores -----\n'
msg += 'auc_score: {}\n'.format(auc_score[0])
msg += 'mAP: {}\n'.format(map_score[0])
msg += 'task score: {}\n'.format(auc_score[0]+map_score[0])
msg += '----- Binary scores -----\n'
msg += 'auc: {}\n'.format(auc_score[0])
msg += 'mAP: {}\n'.format(map_score[0])
msg += '----- Final Score -----\n'
msg += str(auc_score[0]+map_score[0])


np.savetxt(os.path.join(os.getcwd(), 'preds.csv'), 
    preds,
    delimiter =", ", 
    fmt ='% s')

np.savetxt(os.path.join(os.getcwd(), 'scores.csv'),
    np.column_stack((np.array(auc_score), np.array(map_score))),
    header='auc, map',
    delimiter=', ',
    fmt='% s')

f = open("final_scores.txt", "w")
f.write(msg)
f.close()

print(msg)
'''