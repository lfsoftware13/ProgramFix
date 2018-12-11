import os

from config import root

if __name__ == '__main__':
    start_epoch = 24
    end_epoch = 37
    config_name = 'encoder_sample_config11'
    load_model_name = 'encoder_sample_config11.pkl'
    save_name = 'test.pkl'
    root_path = root
    gpu = 0
    stop_type = 'normal'
    log_file = '{}_{}_batch_run.log'.format(config_name, stop_type)

    for i in range(start_epoch, end_epoch):
        real_load_model_name = load_model_name + str(i)
        special_log_file = log_file + str(i)
        log_path = os.path.join(root, 'log', special_log_file)
        comm = 'PYTHONPATH={} python train.py --debug=False --config_name={} --parallel=False --gpu={} --load_previous=True --output_log=log/test.log --load_model_name={} --save_model_name={} > {}'.format(
            root_path, config_name, gpu, real_load_model_name, save_name, log_path
        )
        print('run command: {}'.format(comm))
        res = os.system(comm)
