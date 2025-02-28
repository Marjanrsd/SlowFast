params = dict()

params['num_classes'] = 101

params['dataset'] = r'D:\SlowFastNN\UCF-101\archive'

params['epoch_num'] = 40
params['batch_size'] = 16
params['step'] = 10
params['num_workers'] = 4
params['learning_rate'] = 1e-2
params['momentum'] = 0.9
params['weight_decay'] = 1e-5
params['display'] = 10
#params['pretrained'] = None
# for testing only
params['pretrained'] = r'C:\Users\marjan\UCF101\2025-02-12-10-12-06\clip_len_64frame_sample_rate_1_checkpoint_39.pth.tar'
params['gpu'] = [0]
params['log'] = 'log'
#params['save_path'] = 'UCF101'
params['save_path'] = 'for_debugging_UCF101'
params['clip_len'] = 64
params['frame_sample_rate'] = 1
