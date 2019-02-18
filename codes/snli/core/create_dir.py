#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import os

class Dir_Utilities(object):

    def __init__(self, data_in_path, model_nm):
        self.data_out = 'data_out/'
        self.data_in_path = data_in_path
        self.export_pb_path=os.path.join(self.data_out, 'export_pb/')
        self.log_path=os.path.join(self.data_out, 'logs/')
        self.backup_path=os.path.join(self.data_out, 'backups/')
        self.model_nm = model_nm

        if self.model_nm == 'onehot':
            self.nn_models_dssm_path = os.path.join(self.data_out, 'nn_models_onehot/')
        else:
            self.nn_models_dssm_path = os.path.join(self.data_out, 'nn_models_embed/')

    def create_domain_dir(self, dir_path):
        """ domain에 따른 폴더 생성 """
        if os.path.isdir(dir_path):
            print("{} --- Folder already exists \n".format(dir_path))
        else:
            os.makedirs(self.data_out, exist_ok=True)
            print("{} --- Folder create complete \n".format(dir_path))

    def folder_init(self):
        print("---- start test -----")

        self.create_domain_dir(self.data_out)
        self.create_domain_dir(self.export_pb_path)
        self.create_domain_dir(self.log_path)
        self.create_domain_dir(self.backup_path)
        self.create_domain_dir(self.nn_models_dssm_path)

        print("---- end test -----")