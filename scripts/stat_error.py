from scripts.scripts_util import read_experiment_result_df


def calculate_compile_result(df):
    total_records = 6975.0
    correct_df = df[df['compile_res'].map(lambda x: x > 0)]
    correct_count = len(correct_df)
    correct = float(correct_count) / total_records
    print('correct_count: {}/{}, correct: {}'.format(correct_count, total_records, correct))


def calculate_part_correct(df):
    def part_correct(one):
        if one['error_count'] <= 0:
            return False
        return one['error_count'] < one['original_error_count']

    total_records = 6975.0
    part_correct_df = df[df.apply(part_correct, axis=1, raw=True)]
    part_correct_count = len(part_correct_df)
    part_correct = float(part_correct_count) / total_records
    print('part_correct_count: {}/{}, part_correct: {}'.format(part_correct_count, total_records, part_correct))


def calculate_error_solver(df):
    # original_total_error = sum(df['original_error_count'].tolist())
    original_total_error = 16766.0
    effect_df = df[df['error_count'].map(lambda x: x >= 0)]
    model_total_error = sum(effect_df['error_count'].tolist())
    resolved = 1 - (float(model_total_error)/float(original_total_error))
    print('model_total_error: {}, original_total_error: {}, resolved: {}'.format(model_total_error,
                                                                                 original_total_error, resolved))


def stat_main(db_path, table_name, compile_result=True, part_correct=True, error_solver=True, max_sample_step=None):
    df = read_experiment_result_df(db_path, table_name)
    if max_sample_step is not None:
        df = df[df['sample_step'].map(lambda x: x <= max_sample_step)]
    if compile_result:
        calculate_compile_result(df)
    if part_correct:
        calculate_part_correct(df)
    if error_solver:
        calculate_error_solver(df)


if __name__ == '__main__':
    from config import DATA_RECORDS_DEEPFIX_DBPATH
    table_name = 'pretrain_encoder_sample_no_pretrain_config1_33'
    stat_main(DATA_RECORDS_DEEPFIX_DBPATH, table_name)

