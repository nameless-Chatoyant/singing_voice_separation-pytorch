from easydict import EasyDict as edict

cfg = edict()



cfg.encoder = edict()
cfg.encoder.leakiness = 0.2
cfg.encoder.ch_out = [16, 32, 64, 128, 256, 512]
cfg.encoder.ch_in = [1] + cfg.encoder.ch_out[:-1]
cfg.encoder.stride = 2
cfg.encoder.kernel_size = (5,5)


cfg.decoder = edict()
cfg.decoder.ch_out = [256, 128, 64, 32, 16]
cfg.decoder.ch_in = [1] + cfg.decoder.ch_out[:-1]
cfg.decoder.stride = 2
cfg.decoder.kernel_size = (5,5)