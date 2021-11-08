import os
import tools

RESULTS_PATH = './../results/results.txt'

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # modelnames = ['cnn', 'crnn']
    databases = ['mitdb', 'nstdb', 'qtdb']

    # Now test only on cnn
    modelname = 'cnn'
    model = tools.load_model(modelname)

    for database in databases:
        db, db_r, name, fs = tools.load_data(database)
        print(f'Resampling {database} data')
        data, ref = tools.resamp(db, db_r, fs)
        with open(RESULTS_PATH, 'a') as f:
            print(f'Predict QRS with {modelname} model on {database}', file=f)
        tools.score(data, ref, model, name, fs)
