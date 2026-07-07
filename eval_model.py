import json, numpy as np, os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_NUM_INTRAOP_THREADS'] = '2'
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

name = sys.argv[1]
CL = ['PPE_Compliant', 'HardHat_Only', 'Vest_Only', 'NO_PPE']
Xte = np.load('Xte.npy'); yte = np.load('yte.npy')
m = tf.keras.models.load_model(f'best_{name}.keras')
prob = m.predict(Xte, verbose=0, batch_size=16)
pred = prob.argmax(1)
rep = classification_report(yte, pred, target_names=CL, output_dict=True, zero_division=0)
auc = roc_auc_score(tf.keras.utils.to_categorical(yte, 4), prob, multi_class='ovr', average='macro')
out = {'report': rep, 'cm': confusion_matrix(yte, pred).tolist(),
       'auc': float(auc), 'params': int(m.count_params()),
       'prob': prob.tolist(), 'yte': yte.tolist()}
json.dump(out, open(f'res_{name}.json', 'w'))
print(name, 'ACC %.4f' % rep['accuracy'], 'F1 %.4f' % rep['macro avg']['f1-score'], 'AUC %.4f' % auc)
