import os
import torch

from torch import nn, optim
from tqdm import tqdm

from common import torch_util, problem_util, util
from common.logger import init_a_file_logger, info
from common.problem_util import to_cuda
from common.util import data_loader
import torch.functional as F

from database.database_util import create_table, insert_items

IGNORE_TOKEN = -1


def get_model(model_fn, model_params, path, load_previous=False, parallel=False, gpu_index=None):
    m = model_fn(
        **model_params
    )
    # to_cuda(m)
    if parallel:
        m = nn.DataParallel(m.cuda(), device_ids=[0, 1])
    elif gpu_index is not None:
        m = nn.DataParallel(m.cuda(), device_ids=[gpu_index])
    else:
        m = nn.DataParallel(m.cuda(), device_ids=[0])
    if load_previous:
        torch_util.load_model(m, path)
        print("load previous model")
    else:
        print("create new model")
    if gpu_index is None:
        m = m.module.cpu()
    return m


def train(model, dataset, batch_size, loss_function, optimizer, clip_norm, epoch_ratio, parse_input_batch_data_fn,
          parse_target_batch_data_fn, create_output_ids_fn, evaluate_obj_list):
    total_loss = to_cuda(torch.Tensor([0]))
    steps = 0
    for o in evaluate_object_list:
        o.clear_result()
    model.train()

    with tqdm(total=(len(dataset)*epoch_ratio)) as pbar:
        for batch_data in data_loader(dataset, batch_size=batch_size, is_shuffle=True, drop_last=True, epoch_ratio=epoch_ratio):
            model.zero_grad()

            model_input = parse_input_batch_data_fn(batch_data)
            model_output = model.forward(*model_input)

            model_target = parse_target_batch_data_fn(batch_data)
            loss = loss_function(*model_output, *model_target)

            loss.backward()
            optimizer.step()

            output_ids = create_output_ids_fn(model_output, model_input, False)
            for evaluator in evaluate_obj_list:
                evaluator.add_result(output_ids, model_output, model_target, model_input, batch_data=batch_data)

            total_loss += loss

            step_output = 'in train step {}  loss: {}'.format(steps, loss.data.item())
            # print(step_output)
            info(step_output)

            steps += 1
            pbar.update(batch_size)

    return evaluate_obj_list, (total_loss / steps).item()


def evaluate(model, dataset, batch_size, loss_function, parse_input_batch_data_fn, parse_target_batch_data_fn,
             do_sample=False, print_output=False, create_output_ids_fn=None, evaluate_obj_list=[],
             expand_output_and_target_fn=None):
    total_loss = to_cuda(torch.Tensor([0]))
    total_batch = to_cuda(torch.Tensor([0]))
    steps = 0
    for o in evaluate_object_list:
        o.clear_result()
    model.eval()

    with tqdm(total=len(dataset)) as pbar:
        with torch.no_grad():
            for batch_data in data_loader(dataset, batch_size=batch_size, drop_last=True):
                model.zero_grad()

                # model_input = parse_input_batch_data(batch_data)
                model_input = parse_input_batch_data_fn(batch_data, do_sample=do_sample)
                # model_output = model.forward(*model_input, test=do_sample)
                if do_sample:
                    model_output = model.forward(*model_input, test=True)

                    model_target = parse_target_batch_data_fn(batch_data)

                    model_output, model_target = expand_output_and_target_fn(model_output, model_target)
                else:
                    model_output = model.forward(*model_input)
                    model_target = parse_target_batch_data_fn(batch_data)

                loss = loss_function(*model_output, *model_target)

                output_ids = create_output_ids_fn(model_output, model_input, do_sample)
                total_loss += loss
                total_batch += batch_size

                step_output = 'in evaluate step {}  loss: {}, '.format(steps, loss.data.item())
                for evaluator in evaluate_obj_list:
                    res = evaluator.add_result(output_ids, model_output, model_target, model_input, batch_data=batch_data)
                    step_output += res
                # print(step_output)
                info(step_output)

                if print_output and steps % 10 == 0:
                    pass
                    # output_ids = output_ids.tolist()
                    # target_ids = batch_data['ac_tokens']
                    # is_copy = (is_copy > 0.5).tolist()
                    # target_is_copy = target_is_copy.tolist()
                    # value_output = torch.squeeze(torch.topk(F.softmax(value_output, dim=-1), k=1, dim=-1)[1], dim=-1)
                    # value_output = value_output.tolist()
                    # target_ac_tokens = target_ac_tokens.tolist()
                    # pointer_output = torch.squeeze(torch.topk(F.softmax(pointer_output, dim=-1), k=1, dim=-1)[1], dim=-1)
                    # pointer_output = pointer_output.tolist()
                    # target_pointer_output = target_pointer_output.tolist()
                    # target_length = torch.sum(output_mask, dim=-1)
                    # target_length = target_length.tolist()
                    # for out, tar, cop, tar_cop, val, tar_val, poi, tar_poi, tar_len in zip(output_ids, target_ids, is_copy,
                    #                                                               target_is_copy, value_output,
                    #                                                               target_ac_tokens,
                    #                                                               pointer_output,
                    #                                                               target_pointer_output, target_length):
                    # # for out, tar,  in zip(output_ids, target_ids):
                    #     out_code, end_pos = convert_one_token_ids_to_code(out, id_to_word_fn=vocab.id_to_word, start=start_id,
                    #                                          end=end_id, unk=unk_id)
                    #     tar_code, tar_end_pos = convert_one_token_ids_to_code(tar[1:], id_to_word_fn=vocab.id_to_word, start=start_id,
                    #                                          end=end_id, unk=unk_id)
                    #     info('-------------- step {} ------------------------'.format(steps))
                    #     info('output: {}'.format(out_code))
                    #     info('target: {}'.format(tar_code))
                    #     cop = [str(c) for c in cop]
                    #     tar_cop = [str(int(c)) for c in tar_cop]
                    #     poi = [str(c) for c in poi]
                    #     tar_poi = [str(c) for c in tar_poi]
                    #     info('copy output: {}'.format(' '.join(cop[:tar_len])))
                    #     info('copy target: {}'.format(' '.join(tar_cop[:tar_len])))
                    #     info('pointer output: {}'.format(' '.join(poi[:tar_len])))
                    #     info('pointer target: {}'.format(' '.join(tar_poi[:tar_len])))
                    #
                    #     value_list = []
                    #     target_list = []
                    #     for c, v, t in zip(tar_cop, val, tar_val):
                    #         if c == '1':
                    #             value_list += ['<COPY>']
                    #             target_list += ['<COPY>']
                    #         else:
                    #             value_list += [vocab.id_to_word(int(v))]
                    #             target_list += [vocab.id_to_word(int(t))]
                    #     info('value output: {}'.format(' '.join(value_list[:tar_len])))
                    #     info('value target: {}'.format(' '.join(target_list[:tar_len])))

                steps += 1
                pbar.update(batch_size)

    return evaluate_obj_list, (total_loss / steps).item()


def sample_and_save(model, dataset, batch_size, loss_function, parse_input_batch_data_fn, parse_target_batch_data_fn,
             do_sample=False, print_output=False, create_output_ids_fn=None, evaluate_obj_list=[],
             expand_output_and_target_fn=None, add_data_record_fn=None, db_path='', table_name=''):
    # total_loss = to_cuda(torch.Tensor([0]))
    total_batch = to_cuda(torch.Tensor([0]))
    saved_count = 0
    steps = 1
    for o in evaluate_object_list:
        o.clear_result()
    model.eval()

    total_saved_list = []

    with tqdm(total=len(dataset)) as pbar:
        with torch.no_grad():
            for batch_data in data_loader(dataset, batch_size=batch_size, drop_last=True):
                model.zero_grad()

                # model_input = parse_input_batch_data(batch_data)
                model_input = parse_input_batch_data_fn(batch_data, do_sample=do_sample)
                # model_output = model.forward(*model_input, test=do_sample)
                if do_sample:
                    model_output = model.forward(*model_input, test=True)

                    model_target = parse_target_batch_data_fn(batch_data)

                    model_output, model_target = expand_output_and_target_fn(model_output, model_target)
                else:
                    model_output = model.forward(*model_input)
                    model_target = parse_target_batch_data_fn(batch_data)

                # loss = loss_function(*model_output, *model_target)

                output_ids = create_output_ids_fn(model_output, model_input)
                # total_loss += loss
                total_batch += batch_size

                # step_output = 'in evaluate step {}  loss: {}, '.format(steps, loss.data.item())
                step_output = 'in evaluate step {} '.format(steps)
                for evaluator in evaluate_obj_list:
                    res = evaluator.add_result(output_ids, model_output, model_target, model_input, batch_data=batch_data)
                    step_output += res
                # print(step_output)
                info(step_output)

                saved_list = add_data_record_fn(output_ids, model_output, batch_data)
                total_saved_list += saved_list

                if steps % 100 == 0:
                    create_table(db_path, table_name)
                    insert_items(db_path, table_name, total_saved_list)
                    saved_count += len(total_saved_list)
                    print('saved {} record in total {}. '.format(saved_count, total_batch.item()))
                    total_saved_list = []

                if print_output and steps % 100 == 0:
                    pass
                    # output_ids = output_ids.tolist()
                    # target_ids = batch_data['ac_tokens']
                    # is_copy = (is_copy > 0.5).tolist()
                    # target_is_copy = target_is_copy.tolist()
                    # value_output = torch.squeeze(torch.topk(F.softmax(value_output, dim=-1), k=1, dim=-1)[1], dim=-1)
                    # value_output = value_output.tolist()
                    # target_ac_tokens = target_ac_tokens.tolist()
                    # pointer_output = torch.squeeze(torch.topk(F.softmax(pointer_output, dim=-1), k=1, dim=-1)[1], dim=-1)
                    # pointer_output = pointer_output.tolist()
                    # target_pointer_output = target_pointer_output.tolist()
                    # target_length = torch.sum(output_mask, dim=-1)
                    # target_length = target_length.tolist()
                    # for out, tar, cop, tar_cop, val, tar_val, poi, tar_poi, tar_len in zip(output_ids, target_ids, is_copy,
                    #                                                               target_is_copy, value_output,
                    #                                                               target_ac_tokens,
                    #                                                               pointer_output,
                    #                                                               target_pointer_output, target_length):
                    # # for out, tar,  in zip(output_ids, target_ids):
                    #     out_code, end_pos = convert_one_token_ids_to_code(out, id_to_word_fn=vocab.id_to_word, start=start_id,
                    #                                          end=end_id, unk=unk_id)
                    #     tar_code, tar_end_pos = convert_one_token_ids_to_code(tar[1:], id_to_word_fn=vocab.id_to_word, start=start_id,
                    #                                          end=end_id, unk=unk_id)
                    #     info('-------------- step {} ------------------------'.format(steps))
                    #     info('output: {}'.format(out_code))
                    #     info('target: {}'.format(tar_code))
                    #     cop = [str(c) for c in cop]
                    #     tar_cop = [str(int(c)) for c in tar_cop]
                    #     poi = [str(c) for c in poi]
                    #     tar_poi = [str(c) for c in tar_poi]
                    #     info('copy output: {}'.format(' '.join(cop[:tar_len])))
                    #     info('copy target: {}'.format(' '.join(tar_cop[:tar_len])))
                    #     info('pointer output: {}'.format(' '.join(poi[:tar_len])))
                    #     info('pointer target: {}'.format(' '.join(tar_poi[:tar_len])))
                    #
                    #     value_list = []
                    #     target_list = []
                    #     for c, v, t in zip(tar_cop, val, tar_val):
                    #         if c == '1':
                    #             value_list += ['<COPY>']
                    #             target_list += ['<COPY>']
                    #         else:
                    #             value_list += [vocab.id_to_word(int(v))]
                    #             target_list += [vocab.id_to_word(int(t))]
                    #     info('value output: {}'.format(' '.join(value_list[:tar_len])))
                    #     info('value target: {}'.format(' '.join(target_list[:tar_len])))

                steps += 1
                pbar.update(batch_size)

    create_table(db_path, table_name)
    insert_items(db_path, table_name, total_saved_list)
    saved_count += len(total_saved_list)
    print('saved {} record in total {}. '.format(saved_count, total_batch.item()))

    return evaluate_obj_list


def train_and_evaluate(model, batch_size, train_dataset, valid_dataset, test_dataset, ac_copy_dataset,
                       learning_rate, epoches, saved_name, train_loss_fn, optimizer, optimizer_dict,
                       parse_input_batch_data_fn, parse_target_batch_data_fn,
                       create_output_ids_fn, evaluate_obj_list,
                       load_previous=False, is_debug=False, epoch_ratio=1.0, clip_norm=1,
                       do_sample_evaluate=False, print_output=False, expand_output_and_target_fn=None,
                       addition_train=False, addition_train_remain_frac=1.0, addition_epoch_ratio=0.4,
                       start_epoch=0, ac_copy_train=False, ac_copy_radio=1.0,
                       do_sample_and_save=False, db_path=None, table_basename=None,
                       do_multi_step_evaluate=False, max_sample_times=1, compile_file_path=None, multi_step_no_target=False,
                       add_data_record_fn=None):
    valid_loss = 0
    test_loss = 0
    valid_accuracy = 0
    test_accuracy = 0
    valid_correct = 0
    test_correct = 0
    sample_valid_loss = 0
    sample_test_loss = 0
    sample_valid_accuracy = 0
    sample_test_accuracy = 0
    sample_valid_correct = 0
    sample_test_correct = 0

    save_path = os.path.join(config.save_model_root, saved_name)

    addition_dataset = None

    optimizer = optimizer(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, **optimizer_dict)

    if load_previous:
        # valid_loss, valid_accuracy, valid_correct = evaluate(model=model, dataset=valid_dataset, batch_size=batch_size,
        #                                       loss_function=loss_fn, vocab=vocabulary, add_value_mask=add_value_mask)
        # test_evaluator, test_loss = evaluate(model=model, dataset=test_dataset, batch_size=batch_size,
        #                                      loss_function=train_loss_fn, do_sample=False,
        #                                      parse_input_batch_data_fn=parse_input_batch_data_fn,
        #                                      parse_target_batch_data_fn=parse_target_batch_data_fn,
        #                                      create_output_ids_fn=create_output_ids_fn,
        #                                      evaluate_obj_list=evaluate_obj_list,
        #                                      expand_output_and_target_fn=expand_output_and_target_fn)
        # print('previous test loss: {}, evaluator : '.format(test_loss))
        # for evaluator in test_evaluator:
        #     print(evaluator)

        if do_sample_and_save:
            sample_and_save_evalutor = sample_and_save(model=model, dataset=test_dataset, batch_size=batch_size,
                                                       loss_function=train_loss_fn, do_sample=True,
                                                       print_output=print_output,
                                                       parse_input_batch_data_fn=parse_input_batch_data_fn,
                                                       parse_target_batch_data_fn=parse_target_batch_data_fn,
                                                       create_output_ids_fn=create_output_ids_fn,
                                                       evaluate_obj_list=evaluate_obj_list,
                                                       expand_output_and_target_fn=expand_output_and_target_fn,
                                                       add_data_record_fn=add_data_record_fn,
                                                       db_path=db_path, table_name=table_basename+'_train')
            print('sample and save evaluator : '.format(sample_test_loss))
            for evaluator in sample_and_save_evalutor:
                print(evaluator)

            sample_and_save_evalutor = sample_and_save(model=model, dataset=valid_dataset, batch_size=batch_size,
                                                       loss_function=train_loss_fn, do_sample=True,
                                                       print_output=print_output,
                                                       parse_input_batch_data_fn=parse_input_batch_data_fn,
                                                       parse_target_batch_data_fn=parse_target_batch_data_fn,
                                                       create_output_ids_fn=create_output_ids_fn,
                                                       evaluate_obj_list=evaluate_obj_list,
                                                       expand_output_and_target_fn=expand_output_and_target_fn,
                                                       add_data_record_fn=add_data_record_fn,
                                                       db_path=db_path, table_name=table_basename+'_valid')
            print('sample and save evaluator : '.format(sample_test_loss))
            for evaluator in sample_and_save_evalutor:
                print(evaluator)

            sample_and_save_evalutor = sample_and_save(model=model, dataset=train_dataset, batch_size=batch_size,
                                                       loss_function=train_loss_fn, do_sample=True,
                                                       print_output=print_output,
                                                       parse_input_batch_data_fn=parse_input_batch_data_fn,
                                                       parse_target_batch_data_fn=parse_target_batch_data_fn,
                                                       create_output_ids_fn=create_output_ids_fn,
                                                       evaluate_obj_list=evaluate_obj_list,
                                                       expand_output_and_target_fn=expand_output_and_target_fn,
                                                       add_data_record_fn=add_data_record_fn,
                                                       db_path=db_path,
                                                       table_name=table_basename+'_test')
            print('sample and save evaluator : '.format(sample_test_loss))
            for evaluator in sample_and_save_evalutor:
                print(evaluator)
            return

        if do_multi_step_evaluate:
            pass
            # sample_valid_compile = 0
            # first_sample_valid_compile = 0
            # # sample_valid_compile, sample_valid_accuracy, sample_valid_correct, first_sample_valid_compile = evaluate_multi_step(model=model, dataset=valid_dataset,
            # #          batch_size=batch_size, do_sample=True, vocab=vocabulary, print_output=print_output,
            # #          max_sample_times=max_sample_times, file_path=compile_file_path)
            # sample_test_compile, sample_test_accuracy, sample_test_correct, first_sample_test_compile = evaluate_multi_step(
            #     model=model, dataset=test_dataset, batch_size=batch_size, do_sample=True, vocab=vocabulary,
            #     print_output=print_output, max_sample_times=max_sample_times, file_path=compile_file_path,
            #     no_target=multi_step_no_target)
            # evaluate_output = 'evaluate in multi step: sample valid compile: {}, sample test compile: {}, ' \
            #                   'first_sample_valid_compile: {}, first_sample_test_compile: {}' \
            #                   'sample valid accuracy: {}, sample test accuracy: {}, ' \
            #                   'sample valid correct: {}, sample test correct: {}'.format(
            #     sample_valid_compile, sample_test_compile, first_sample_valid_compile, first_sample_test_compile,
            #     sample_valid_accuracy, sample_test_accuracy, sample_valid_correct,
            #     sample_test_correct)
            # print(evaluate_output)
            # info(evaluate_output)

        if do_sample_evaluate:
            # sample_valid_loss, sample_valid_accuracy, sample_valid_correct = evaluate(model=model, dataset=valid_dataset, batch_size=batch_size,
            #                                       loss_function=train_loss_fn, do_sample=True,
            #                                       print_output=print_output,
            #                                       parse_input_batch_data_fn=parse_input_batch_data_fn,
            #                                       parse_target_batch_data_fn=parse_target_batch_data_fn,
            #                                       create_output_ids_fn=create_output_ids_fn,
            #                                       evaluate_obj_list=evaluate_obj_list)
            sample_test_evalutor, sample_test_loss = evaluate(model=model, dataset=test_dataset, batch_size=batch_size,
                                                              loss_function=train_loss_fn, do_sample=True,
                                                              print_output=print_output,
                                                              parse_input_batch_data_fn=parse_input_batch_data_fn,
                                                              parse_target_batch_data_fn=parse_target_batch_data_fn,
                                                              create_output_ids_fn=create_output_ids_fn,
                                                              evaluate_obj_list=evaluate_obj_list,
                                                              expand_output_and_target_fn=expand_output_and_target_fn)
            print('previous sample test loss: {}, evaluator : '.format(sample_test_loss))
            for evaluator in sample_test_evalutor:
                print(evaluator)
        evaluate_output = 'evaluate: valid loss of {}, test loss of {}, ' \
                          'valid_accuracy result of {}, test_accuracy result of {}, ' \
                          'valid correct result of {}, test correct result of {}, ' \
                          'sample valid loss: {}, sample test loss: {}, ' \
                          'sample valid accuracy: {}, sample test accuracy: {}, ' \
                          'sample valid correct: {}, sample test correct: {}'.format(
            valid_loss, test_loss, valid_accuracy, test_accuracy, valid_correct, test_correct,
            sample_valid_loss, sample_test_loss, sample_valid_accuracy, sample_test_accuracy, sample_valid_correct, sample_test_correct)
        print(evaluate_output)
        info(evaluate_output)
        pass

    for epoch in range(start_epoch, start_epoch+epoches):
        print('----------------------- in epoch {} --------------------'.format(epoch))
        combine_train_set = train_dataset
        if addition_train:
            pass
            # addition_predict_dataset = train_dataset
            # if addition_dataset is not None:
            #     addition_predict_radio = addition_epoch_ratio/(addition_epoch_ratio+1)
            #     addition_predict_dataset = addition_predict_dataset.combine_dataset(addition_dataset)
            # else:
            #     addition_predict_radio = addition_epoch_ratio
            # records = sample_better_output(model=model, dataset=addition_predict_dataset, batch_size=batch_size,
            #                                vocab=vocabulary, do_sample=True, epoch_ratio=addition_predict_radio,
            #                                add_value_mask=add_value_mask)
            #
            # addition_dict = create_addition_error_data(records)
            # if addition_dataset is None:
            #     addition_dataset = CCodeErrorDataSet(pd.DataFrame(addition_dict), vocabulary, 'addition_train',
            #                                          transformer_vocab_slk=mask_transformer)
            # else:
            #     addition_dataset.remain_samples(frac=addition_train_remain_frac)
            #     addition_dataset.add_samples(pd.DataFrame(addition_dict).sample(frac=1 - addition_train_remain_frac))
            # info_output = "In epoch {}, there are {} parsed data in the {} dataset".format(epoch, len(addition_dataset), 'addition_train')
            # print(info_output)
            # combine_train_set = combine_train_set.combine_dataset(addition_dataset)

        if ac_copy_train:
            combine_train_set = combine_train_set.combine_dataset(ac_copy_dataset.remain_dataset(frac=ac_copy_radio))

        train_evaluator, train_loss = train(model=model, dataset=combine_train_set, batch_size=batch_size,
                                            loss_function=train_loss_fn, optimizer=optimizer, clip_norm=clip_norm,
                                            epoch_ratio=epoch_ratio, parse_input_batch_data_fn=parse_input_batch_data_fn,
                                            parse_target_batch_data_fn=parse_target_batch_data_fn,
                                            create_output_ids_fn=create_output_ids_fn,
                                            evaluate_obj_list=evaluate_obj_list, )
        print('epoch: {} loss: {}, train evaluator : '.format(epoch, train_loss))
        for evaluator in train_evaluator:
            print(evaluator)

        valid_evaluator, valid_loss = evaluate(model=model, dataset=valid_dataset, batch_size=batch_size,
                                               loss_function=train_loss_fn,
                                               parse_input_batch_data_fn=parse_input_batch_data_fn,
                                               parse_target_batch_data_fn=parse_target_batch_data_fn,
                                               create_output_ids_fn=create_output_ids_fn,
                                               evaluate_obj_list=evaluate_obj_list,
                                               expand_output_and_target_fn=expand_output_and_target_fn)
        print('epoch: {} loss: {}, valid evaluator : '.format(epoch, valid_loss))
        for evaluator in valid_evaluator:
            print(evaluator)
        test_evaluator, test_loss = evaluate(model=model, dataset=test_dataset, batch_size=batch_size,
                                             loss_function=train_loss_fn,
                                             parse_input_batch_data_fn=parse_input_batch_data_fn,
                                             parse_target_batch_data_fn=parse_target_batch_data_fn,
                                             create_output_ids_fn=create_output_ids_fn,
                                             evaluate_obj_list=evaluate_obj_list,
                                             expand_output_and_target_fn=expand_output_and_target_fn, )
        print('epoch: {} loss: {}, test evaluator : '.format(epoch, test_loss))
        for evaluator in test_evaluator:
            print(evaluator)

        if not is_debug:
            torch_util.save_model(model, save_path+str(epoch))


if __name__ == '__main__':
    import parameters_config
    import config
    import argparse

    def boolean_string(s):
        if s not in {'False', 'True'}:
            raise ValueError('Not a valid boolean string')
        return s == 'True'

    parser = argparse.ArgumentParser()
    parser.add_argument("--load_previous", type=boolean_string, default=False)
    parser.add_argument("--debug", type=boolean_string, default=True)
    parser.add_argument("--config_name", type=str)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--parallel", type=boolean_string)
    parser.add_argument("--just_evaluate", type=boolean_string, default=False)
    parser.add_argument("--output_log", type=str, default=None)
    args = parser.parse_args()
    load_previous = args.load_previous
    problem_util.GPU_INDEX = args.gpu
    problem_util.Parallel = args.parallel
    is_debug = args.debug
    just_evaluate = args.just_evaluate

    p_config = parameters_config.__dict__.get(args.config_name)(is_debug)
    epoches = p_config.get("epcohes", 20)
    learning_rate = p_config.get("learning_rate", 20)
    batch_size = p_config.get("batch_size", 32)
    train_loss_fn = p_config.get("train_loss", nn.CrossEntropyLoss)
    clip_norm = p_config.get("clip_norm", 10)
    optimizer = p_config.get("optimizer", optim.SGD)
    optimizer_dict = p_config.get("optimizer_dict", dict())
    epoch_ratio = p_config.get("epoch_ratio", 0.25)
    start_epoch = p_config.get('start_epoch', 0)
    ac_copy_train = p_config.get('ac_copy_train', False)
    ac_copy_radio = p_config.get('ac_copy_radio', 0.2)
    evaluate_object_list = p_config.get("evaluate_object_list")
    do_sample_evaluate = p_config.get('do_sample_evaluate', False)
    do_sample_and_save = p_config.get('do_sample_and_save', False)
    # label_preprocess_fn = p_config.get("label_preprocess", lambda x: to_cuda(torch.LongTensor(x['label'])))
    # scheduler_fn = p_config.get("scheduler_fn", lambda x: torch.optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=3, verbose=True))
    save_root_path = os.path.join(config.save_model_root, p_config.get("name"))
    util.make_dir(save_root_path)
    print("save_root_path:{}".format(save_root_path))
    init_a_file_logger(args.output_log)
    print("logger_file_path:{}".format(args.output_log))
    # need_pad = p_config.get("need_pad", False)
    save_name = p_config['save_name']
    parse_input_batch_data_fn = p_config['parse_input_batch_data_fn']
    parse_target_batch_data_fn = p_config['parse_target_batch_data_fn']
    expand_output_and_target_fn = p_config.get('expand_output_and_target_fn', None)
    add_data_record_fn = p_config.get('add_data_record_fn', None)
    db_path = p_config.get('db_path', None)
    table_basename = p_config.get('table_basename', None)
    create_output_ids_fn = p_config['create_output_ids_fn']
    model_path = os.path.join(save_root_path, p_config['load_model_name'])
    model = get_model(
        p_config['model_fn'],
        p_config['model_dict'],
        model_path,
        load_previous=load_previous,
        parallel=problem_util.Parallel,
        gpu_index=problem_util.GPU_INDEX
    )

    train_data, val_data, test_data, ac_copy_data = p_config.get("data")
    print("The size of train data: {}".format(len(train_data)))
    print("The size of val data: {}".format(len(val_data)))
    print("The size of test data: {}".format(len(test_data)))
    train_and_evaluate(model=model, batch_size=batch_size, train_dataset=train_data, valid_dataset=val_data, test_dataset=test_data,
                       ac_copy_dataset=ac_copy_data,
                       learning_rate=learning_rate, epoches=epoches, saved_name=save_name, train_loss_fn=train_loss_fn,
                       optimizer=optimizer, optimizer_dict=optimizer_dict,
                       parse_input_batch_data_fn=parse_input_batch_data_fn, parse_target_batch_data_fn=parse_target_batch_data_fn,
                       create_output_ids_fn=create_output_ids_fn, evaluate_obj_list=evaluate_object_list,
                       load_previous=load_previous, is_debug=is_debug, epoch_ratio=epoch_ratio, clip_norm=clip_norm, start_epoch=start_epoch,
                       do_sample_evaluate=do_sample_evaluate, print_output=False,
                       ac_copy_train=ac_copy_train, ac_copy_radio=ac_copy_radio,
                       addition_train=False, addition_train_remain_frac=1.0, addition_epoch_ratio=0.4,
                       do_multi_step_evaluate=False, max_sample_times=1, compile_file_path=None, multi_step_no_target=False,
                       expand_output_and_target_fn=expand_output_and_target_fn,
                       do_sample_and_save=do_sample_and_save, add_data_record_fn=add_data_record_fn, db_path=db_path,
                       table_basename=table_basename)

    # test_loss, train_test_loss = evaluate(model, test_data, batch_size, evaluate_object_list,
    #                                       train_loss_fn, "test_evaluate", label_preprocess_fn)
    # print("train_test_loss is {}".format(train_test_loss.item(),))
    # for o in  test_loss:
    #     print(o)