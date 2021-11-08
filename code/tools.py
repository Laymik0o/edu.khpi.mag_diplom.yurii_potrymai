import numpy as np
import wfdb
from scipy.signal import resample
from tensorflow.keras.models import model_from_json

RESULTS_PATH = './../results/results.txt'


# preprocessing
def pp(data):
    x = np.max(data)
    if x > 20:
        b = np.argwhere(data > 20)
        for k in b[:, 0]:
            if k > 0 and data[k]-data[k-1] > 20:
                data[k] = data[k-1]
    return data


def load_data(database):
    real_r = [
        'N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j',
        'n', 'E', '/', 'f', 'Q', '?'
    ]
    with open(f'./../data/{database}/RECORDS', 'r') as f:
        lines = f.readlines()
    db = []
    db_r = []
    name = []
    print(f'Reading data from {database}')
    for line in lines:
        fname = line.strip()
        name.append(fname)
        path = f'./../data/{database}/{fname}'
        sample = wfdb.rdsamp(path)
        db.append(sample)
        ann = wfdb.rdann(path, 'atr')
        beats = ann.sample[np.isin(ann.symbol, real_r)]
        db_r.append(beats)
    fs = sample[1]['fs']
    print(f'{len(db)} records are found')
    return db, db_r, name, fs


def resamp(db, db_r, fs):
    data = []
    ref = []
    for i in range(len(db)):
        samp = db[0][0][:, 0]
        ln = len(samp)//fs
        remain = len(samp) % fs
        new = resample(db[i][0][:ln*fs, 0], ln*500)
        if remain > 1:
            rem = resample(db[i][0][ln*fs:, 0], int(remain/fs*500))
            new = np.concatenate((new, rem))
        mean = np.mean(new)
        data.append(new-mean)
        new_r = (db_r[i]/fs*500).astype(int)
        ref.append(new_r)
    print(f'Date were resampled at 500 Hz from {fs} Hz')
    return data, ref


def load_model(modelname):
    print(f'Start to load model {modelname}')
    if modelname == 'crnn':
        model = model_from_json(open('./models/CRNN.json').read())
        model.load_weights('./models/CRNN.h5')
    else:
        model = model_from_json(open('./models/CNN.json').read())
        model.load_weights('./models/CNN.h5')
    print('Model is loaded')
    return model


def score(data, ref, model, name, fs):
    with open(RESULTS_PATH, 'a') as f:
        print('%10s %10s %10s %10s' % ('Record', 'TP', 'FP', 'FN'), file=f)
    pf = performance(data, ref, model, False, name, fs)
    ppr = pf[0]/(pf[0]+pf[1])
    se = pf[0]/(pf[0]+pf[2])
    er = (pf[1]+pf[2])/(pf[0]+pf[1]+pf[2])
    f1 = 2*ppr*se/(se+ppr)
    with open(RESULTS_PATH, 'a') as f:
        print('%10s %9s %9s %9s %9s' % (' ', 'Se', 'Ppr', 'Er', 'F1'),
              file=f)
        print('%10s %9.4f %9.4f %9.4f %9.4f\n' % ('Total', se, ppr, er, f1),
              file=f)


def decision(result, thresh=2):
    pos = np.argwhere(result > 0.5).flatten()
    rpos = []
    pre = 0
    last = len(pos)
    for j in np.where(np.diff(pos) > 2)[0]:
        if j-pre > thresh:
            rpos.append((pos[pre]+pos[j])*4)
        pre = j+1
    rpos.append((pos[pre]+pos[last-1])*4)
    qrs = np.array(rpos)
    qrs_diff = np.diff(qrs)
    check = True
    while check:
        qrs_diff = np.diff(qrs)
        if len(qrs_diff) > 1:
            for r in range(len(qrs_diff)):
                if qrs_diff[r] < 100:
                    if result[int(qrs[r]/8)] > result[int(qrs[r+1]/8)]:
                        qrs = np.delete(qrs, r+1)
                        check = True
                        break
                    else:
                        qrs = np.delete(qrs, r)
                        check = True
                        break
                check = False
        else:
            check = False
    return qrs


def recheck(result, qrs, thresh=1):
    qrs_diff = np.diff(qrs)
    miss = np.where(qrs_diff > 600)[0]
    for i in miss:
        add_qrs = decision(result[qrs[i]//8-1:qrs[i+1]//8+2], thresh=thresh)
        if len(add_qrs) > 2:
            maxposb = add_qrs[1]
            for add in add_qrs[1:-1]:
                if result[qrs[i]//8-1+add//8] > result[qrs[i]//8-1+maxposb//8]:
                    maxposb = add
            qrs = list(qrs)
            qrs.append(maxposb+qrs[i]-8)
    qrs = np.sort(np.array(qrs))
    return qrs


def QRS_decision(result):
    qrs = decision(result, thresh=2)
    qrs_diff = np.diff(qrs)
    if np.max(qrs_diff) > 600:
        qrs = recheck(result, qrs, thresh=1)
        qrs_diff = np.diff(qrs)
        if np.max(qrs_diff) > 600:
            qrs = recheck(result, qrs, thresh=0)
            qrs_diff = np.diff(qrs)
            if np.max(qrs_diff) > 600:
                qrs = recheck(result, qrs, thresh=-1)
    return qrs


def performance(data, refs, model, write_to_file=False, name=None, fs=500):
    R_ans = []
    tp_all = 0
    fp_all = 0
    fn_all = 0
    for i in range(len(data)):
        pred = model.predict(data[i].reshape(1, -1, 1))[0][:, 0]
        r_ans = QRS_decision(pred)
        # remove predictions from flutter segments of record 207 in MITDB
        if name and name[i] == '207':
            r_ans = r_ans[~((r_ans > 14894//360*500-250) *
                            (r_ans < 21608//360*500+250))]
            r_ans = r_ans[~((r_ans > 87273//360*500-250) *
                            (r_ans < 100956//360*500+250))]
            r_ans = r_ans[~((r_ans > 554826//360*500-250) *
                            (r_ans < 589660//360*500+250))]
        
        R_ans.append(r_ans)
        if write_to_file:
            if name:
                filename = name[i]
            else:
                filename = str(i)
            np.savetxt(f'./../results/{filename}_QRS.txt', r_ans*fs//500,
                       fmt='%d', delimiter=' ')

        if refs:
            tp = 0
            fp = 0
            fn = 0
            for j in refs[i]:
                loc = np.where(np.abs(R_ans[i]-j) < 500*0.15)[0]
                if len(loc) > 0:
                    tp += 1
                    fp += len(loc)-1
                else:
                    fn += 1
            for k in R_ans[i]:
                loc = np.where(np.abs(refs[i]-k) < 500*0.15)[0]
                if len(loc) == 0:
                    fp += 1
            tp_all += tp
            fp_all += fp
            fn_all += fn
            if name is None:
                record = str(i)
            else:
                record = name[i]
            with open(RESULTS_PATH, 'a') as f:
                print('%10s %10d %10d %10d' % (record, tp, fp, fn), file=f)
    return tp_all, fp_all, fn_all
