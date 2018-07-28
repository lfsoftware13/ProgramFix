# root path of the project
import os

from common import util

root = r'/home/lf/Project/GrammaLanguageModel'
# tmp path
temp_code_write_path = r'tmp'
# scrapyOJ db path
scrapyOJ_path = r'/home/lf/new_disk/data_store/codeforces/scrapyOJ.db'
# cache path
cache_path = r'/home/lf/Project/GrammaLanguageModel/data/cache_data'
save_model_root = os.path.join(root, 'trained_model')
util.make_dir(save_model_root)
summarization_source_code_to_method_name_path = r'/home/lf/Project/GrammaLanguageModel/data/summarization_method_name/json'
DEEPFIX_DB = '/home/lf/new_disk/data_store/deepfix/deepfix.db'
SLK_SAMPLE_DBPATH = os.path.join(root, 'data', 'slk_sample_data.db')
FAKE_DEEPFIX_ERROR_DATA_DBPATH = os.path.join(root, 'data', 'fake_deepfix_error_data.db')
