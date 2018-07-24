import os
from predict import main


if __name__ == '__main__':
    top_dir = '/g/kreshuk/pape/Work/data/networks/mad/isbi'
    keys = ['isbi_hed_ms_v1', 'isbi_unet_ms_v1', 'isbi_unet_lr_v1']
    project_dirs = [os.path.join(top_dir, key) for key in keys]
    for pdir, key in zip(project_dirs, keys):
        print("Start prediction for:", key)
        main(pdir, '', 'template_config/inf_config.yml', key)
