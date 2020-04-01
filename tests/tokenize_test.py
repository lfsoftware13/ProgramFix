from io import BytesIO
from tokenize import tokenize
import pandas as pd
import json

from common.pycparser_util import tokenize_by_clex_fn
from read_data.read_experiment_data import read_fake_common_deepfix_error_dataset_with_same_ids


def calculate_record_code_length(data_df):
    tokenize_fn = tokenize_by_clex_fn()
    data_df['tokenized_code'] = data_df['similar_code'].map(tokenize_fn)
    data_df['tokenized_code'] = data_df['tokenized_code'].map(list)
    data_df['code_length'] = data_df['tokenized_code'].map(len)
    return data_df


def statistics_length(df: pd.DataFrame):
    d = {}
    for i, row in df.iterrows():
        d[row['id']] = row['code_length']
    return d


def record_all_data_main():
    train_df, valid_df, test_df = read_fake_common_deepfix_error_dataset_with_same_ids()
    train_df = calculate_record_code_length(train_df)
    valid_df = calculate_record_code_length(valid_df)
    test_df = calculate_record_code_length(test_df)
    train_length_dict = statistics_length(train_df)
    valid_length_dict = statistics_length(valid_df)
    test_length_dict = statistics_length(test_df)
    length_dict = {**train_length_dict, **valid_length_dict, **test_length_dict}
    s = json.dumps(length_dict)
    with open('./text_file/cparser_code_length.txt', mode='w') as f:
        f.write(s)


def main():
    s = r'''


int main() {
	double a,b,delta,fx,xn;
	int n,i;
	scanf("%lf%lf%d",&a,&b,&n);
	delta=(b-a)/n;
	xn=a;
	fx=0.0f;
	while(xn<=b){
	    if(xn<=-1)
	    fx+=(1*delta);
	    else if(xn>-1 && xn<1)
	    fx+=(xn*xn*delta);
	    else
	    fx+=(xn*xn*xn*delta);
	    xn+=delta;
	}
	printf("%.4lf",fx);
	return 0;
}
    '''
    tokenize_fn = tokenize_by_clex_fn()
    g = tokenize_fn(s)
    i = 0
    for tok in g:
        print(i, tok)
        i += 1


if __name__ == '__main__':
    # main()
    record_all_data_main()



