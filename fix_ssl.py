import torch
from omegaconf import DictConfig, OmegaConf, open_dict
cp_path = 'pretrained/w2v_large_lv_fsh_swbd_cv.pt'
cp = torch.load(cp_path)
cfg = DictConfig(cp['cfg'])


dd = OmegaConf.to_container(cfg, resolve=True)
for k,v in dd.items():
    if not isinstance(v, dict):
        continue
    for key, _ in v.items():
        if key == 'eval_wer':
            print(k)
            break


with open_dict(cfg):
    cfg.task.pop('eval_wer')
    cfg.task.pop('eval_wer_config')
    cfg.task.pop('eval_wer_tokenizer')
    cfg.task.pop('eval_wer_post_process')
    cfg.task.pop('autoregressive')
cp['cfg'] = cfg
torch.save(cp, 'pretrained/w2v_large_lv_fsh_swbd_cv_fixed.pt')