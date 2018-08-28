from abc import abstractmethod, ABCMeta

import torch

from common.problem_util import to_cuda
from common.torch_util import expand_tensor_sequence_to_same
from common.util import PaddedList

import torch.nn.functional as F


class Evaluator(metaclass=ABCMeta):
    @abstractmethod
    def clear_result(self):
        pass

    @abstractmethod
    def add_result(self, log_probs, target, ignore_token, batch_data):
        """

        :param log_probs: [batch, ..., vocab_size]
        :param target: [batch, ...], LongTensor, padded with target token. target.shape == log_probs.shape[:-1]
        :param ignore_token: optional, you can choose special ignore token and gpu index for one batch.
                            or use global value when ignore token and gpu_index is None
        :param gpu_index:
        :return:
        """
        pass

    @abstractmethod
    def get_result(self):
        pass


class SequenceExactMatch(Evaluator):

    def __init__(self, rank=1, ignore_token=None, gpu_index=None):
        self.rank = rank
        self.batch_count = 0
        self.match_count = 0
        self.ignore_token = ignore_token
        self.gpu_index = gpu_index

    def clear_result(self):
        self.batch_count = 0
        self.match_count = 0

    def add_result(self, log_probs, target, ignore_token=None, gpu_index=None, batch_data=None):
        """

        :param log_probs: [batch, ..., vocab_size]
        :param target: [batch, ...], LongTensor, padded with target token. target.shape == log_probs.shape[:-1]
        :param ignore_token: optional, you can choose special ignore token and gpu index for one batch.
                            or use global value when ignore token and gpu_index is None
        :param gpu_index:
        :return:
        """
        if ignore_token is None:
            ignore_token = self.ignore_token
        if gpu_index is None:
            gpu_index = self.gpu_index

        if isinstance(target, list):
            target = torch.LongTensor(target)
            if gpu_index is not None:
                target = target.cuda(gpu_index)

        if gpu_index is None:
            log_probs = log_probs.cpu()
            target = target.cpu()

        _, top1_id = torch.topk(log_probs, k=1, dim=-1)
        top1_id = torch.squeeze(top1_id, dim=-1)

        not_equal_result = torch.ne(top1_id, target)

        if ignore_token is not None:
            target_mask = torch.ne(target, ignore_token)
            not_equal_result = not_equal_result & target_mask
        batch_error_count = not_equal_result
        for i in range(len(not_equal_result.shape)-1):
            batch_error_count = torch.sum(batch_error_count, dim=-1)

        # [batch]
        batch_result = torch.eq(batch_error_count, 0)
        batch_match_count = torch.sum(batch_result).data.item()

        batch_size = log_probs.shape[0]
        self.batch_count += batch_size
        self.match_count += batch_match_count
        return batch_match_count / batch_size

    def get_result(self):
        return self.match_count / self.batch_count

    def __str__(self):
        return ' SequenceExactMatch top 1: ' + str(self.get_result())

    def __repr__(self):
        return self.__str__()


class SequenceExactMatchWithMaskSelect(Evaluator):

    def __init__(self, rank=1, ignore_token=None, gpu_index=None):
        self.rank = rank
        self.batch_count = 0
        self.match_count = 0
        self.ignore_token = ignore_token
        self.gpu_index = gpu_index

    def clear_result(self):
        self.batch_count = 0
        self.match_count = 0

    def add_result(self, log_probs, target, ignore_token=None, gpu_index=None, batch_data=None):
        """

        :param log_probs: [batch, ..., vocab_size]
        :param target: a list of tensor, LongTensor, padded with target token. target.shape == log_probs.shape[:-1]
        :param ignore_token: optional, you can choose special ignore token and gpu index for one batch.
                            or use global value when ignore token and gpu_index is None
        :param gpu_index:
        :return:
        """
        if ignore_token is None:
            ignore_token = self.ignore_token
        if gpu_index is None:
            gpu_index = self.gpu_index

        log_probs, probs_mask = log_probs

        if gpu_index is None:
            log_probs = log_probs.cpu()
            target = target.cpu()

        _, top1_id = torch.topk(log_probs, k=1, dim=-1)
        top1_id = torch.squeeze(top1_id, dim=-1)

        top1_batch_list = torch.unbind(top1_id, dim=0)
        mask_batch_list = torch.unbind(probs_mask, dim=0)

        batch_match_count = 0
        for one_top1, one_mask, one_target in zip(top1_batch_list, mask_batch_list, target):
            one_top1 = torch.masked_select(one_top1, one_mask)
            if ignore_token is not None:
                target_mask = torch.ne(one_target, ignore_token)
                one_target = torch.masked_select(one_target, target_mask)
            if one_top1.shape[0] != one_target.shape[0]:
                continue
            not_equal_result = torch.ne(one_top1, one_target)
            has_error = torch.sum(not_equal_result)
            if has_error.data.item() == 0:
                batch_match_count += 1

        batch_size = log_probs.shape[0]
        self.batch_count += batch_size
        self.match_count += batch_match_count
        return batch_match_count / batch_size

    def get_result(self):
        return self.match_count / self.batch_count

    def __str__(self):
        return ' SequenceExactMatchWithMaskSelect top 1: ' + str(self.get_result())

    def __repr__(self):
        return self.__str__()


class SequenceF1Score(Evaluator):
    """
    F1 score evaluator using in paper (A Convolutional Attention Network for Extreme Summarization of Source Code)
    """

    def __init__(self, vocab, rank=1):
        """
        Precision = TP/TP+FP
        Recall = TP/TP+FN
        F1 Score = 2*(Recall * Precision) / (Recall + Precision)
        :param rank: default 1
        :param ignore_token:
        :param gpu_index:
        """
        self.vocab = vocab
        self.rank = rank
        self.tp_count = 0
        # predict_y = TP + FP
        # actual_y = TP + FN
        self.predict_y = 0
        self.actual_y = 0

    def add_result(self, log_probs, target, ignore_token=None, gpu_index=None, batch_data=None):
        """

        :param log_probs: must be 3 dim. [batch, sequence, vocab_size]
        :param target:
        :param ignore_token:
        :param gpu_index:
        :param batch_data:
        :return:
        """
        if isinstance(target, torch.Tensor):
            target = target.cpu()
            target = target.view(target.shape[0], -1)
            target = target.tolist()

        log_probs = log_probs.cpu()
        log_probs = log_probs.view(log_probs.shape[0], -1, log_probs.shape[-1])
        _, top_ids = torch.max(log_probs, dim=-1)
        top_ids = top_ids.tolist()

        end_id = self.vocab.word_to_id(self.vocab.end_tokens[0])
        unk_id = self.vocab.word_to_id(self.vocab.unk)
        batch_tp_count = 0
        batch_predict_y = 0
        batch_actual_y = 0
        for one_predict, one_target in zip(top_ids, target):
            one_predict, _ = self.filter_token_ids(one_predict, end_id, unk_id)
            one_target, _ = self.filter_token_ids(one_target, end_id, unk_id)
            one_tp = set(one_predict) & set(one_target)
            batch_tp_count += len(one_tp)
            batch_predict_y += len(one_predict)
            batch_actual_y += len(one_target)
        self.tp_count += batch_tp_count
        self.predict_y += batch_predict_y
        self.actual_y += batch_actual_y
        precision = float(batch_tp_count ) / float(batch_predict_y)
        recall = float(batch_tp_count) / float(batch_actual_y)
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.
        return f1

    def filter_token_ids(self, token_ids, end, unk):
        def filter_special_token(token_ids, val):
            return list(filter(lambda x: x != val, token_ids))
        try:
            end_position = token_ids.index(end)
            token_ids = token_ids[:end_position+1]
        except ValueError as e:
            end_position = None
        token_ids = filter_special_token(token_ids, unk)
        return token_ids, end_position

    def clear_result(self):
        self.tp_count = 0
        self.predict_y = 0
        self.actual_y = 0

    def get_result(self):
        precision = float(self.tp_count) / float(self.predict_y)
        recall = float(self.tp_count) / float(self.actual_y)
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.
        return f1

    def __str__(self):
        return ' SequenceF1Score top 1: ' + str(self.get_result())

    def __repr__(self):
        return self.__str__()


class SequenceOutputIDToWord(Evaluator):
    def __init__(self, vocab, ignore_token=None, file_path=None):
        self.vocab = vocab
        self.ignore_token = ignore_token
        self.file_path = file_path
        if file_path is not None:
            with open(file_path, 'w') as f:
                pass

    def add_result(self, log_probs, target, ignore_token=None, gpu_index=None, batch_data=None):
        """

        :param log_probs: [batch, seq, vocab_size]
        :param target: [batch, seq]
        :param ignore_token:
        :param gpu_index:
        :param batch_data:
        :return:
        """
        if self.file_path is None:
            return
        if isinstance(target, torch.Tensor):
            target = target.cpu()
            target = target.tolist()

        log_probs = log_probs.cpu()
        _, top_ids = torch.max(log_probs, dim=-1)
        top_ids = top_ids.tolist()

        input_text = batch_data["text"]

        for one_input, one_top_id, one_target in zip(input_text, top_ids, target):
            predict_token = self.convert_one_token_ids_to_code(one_top_id, self.vocab.id_to_word)
            target_token = self.convert_one_token_ids_to_code(one_target, self.vocab.id_to_word)
            self.save_to_file(one_input, predict_token, target_token)

    def save_to_file(self, input_token=None, predict_token=None, target_token=None):
        if self.file_path is not None:
            with open(self.file_path, 'a') as f:
                f.write('---------------------------------------- one record ----------------------------------------\n')
                if input_token is not None:
                    f.write('input: \n')
                    f.write(str(input_token) + '\n')
                if predict_token is not None:
                    f.write('predict: \n')
                    f.write(predict_token + '\n')
                if target_token is not None:
                    f.write('target: \n')
                    f.write(target_token + '\n')

    def filter_token_ids(self, token_ids, start, end, unk):

        def filter_special_token(token_ids, val):
            return list(filter(lambda x: x != val, token_ids))

        try:
            end_position = token_ids.index(end)
            token_ids = token_ids[:end_position+1]
        except ValueError as e:
            end_position = None
        # token_ids = filter_special_token(token_ids, start)
        # token_ids = filter_special_token(token_ids, end)
        token_ids = filter_special_token(token_ids, unk)
        return token_ids, end_position

    def convert_one_token_ids_to_code(self, token_ids, id_to_word_fn):
        if not isinstance(token_ids, list):
            token_ids = list(token_ids)
        # token_ids, _ = self.filter_token_ids(token_ids, start, end, unk)
        tokens = [id_to_word_fn(tok) for tok in token_ids]
        code = ', '.join(tokens)
        return code

    def clear_result(self):
        pass

    def get_result(self):
        pass

    def __str__(self):
        return ''

    def __repr__(self):
        return self.__str__()


if __name__ == "__main__":
    em_eval = SequenceExactMatch(ignore_token=-1, gpu_index=None)

    log_probs = torch.Tensor([
        [0.6, 0.7, 0.4, 0.6, 0.2],
        [0.2, 0.3, 0.5, 0.7, 0.8]
    ]).cuda(0)
    target = torch.LongTensor([
        [1, 1, 0, 1, -1],
        [0, 1, 0, -1, -1]
    ]).cuda(0)
    part = em_eval.add_result(log_probs, target)
    part = em_eval.add_result(log_probs, target)
    # em_eval.clear_result()
    part = em_eval.add_result(log_probs, target)
    print(part)
    log_probs = torch.Tensor([
        [0.6, 0.7, 0.4, 0.6, 0.2],
        [0.2, 0.3, 0.5, 0.7, 0.8]
    ]).cuda(0)
    target = torch.LongTensor([
        [1, 0, 0, 1, -1],
        [0, 0, 1, -1, -1]
    ]).cuda(0)
    part = em_eval.add_result(log_probs, target)
    print(part)

    print(em_eval.match_count, em_eval.batch_count)
    print(em_eval.get_result())


class SequenceBinaryClassExactMatch(Evaluator):

    def __init__(self, rank=1, ignore_token=None, gpu_index=None):
        self.rank = rank
        self.batch_count = 0
        self.match_count = 0
        self.ignore_token = ignore_token
        self.gpu_index = gpu_index

    def clear_result(self):
        self.batch_count = 0
        self.match_count = 0

    def add_result(self, log_probs, target, ignore_token=None, gpu_index=None, batch_data=None):
        """

        :param log_probs: [batch, ...]
        :param target: [batch, ...], LongTensor, padded with target token. target.shape == log_probs.shape[:-1]
        :param ignore_token: optional, you can choose special ignore token and gpu index for one batch.
                            or use global value when ignore token and gpu_index is None
        :param gpu_index:
        :return:
        """
        if ignore_token is None:
            ignore_token = self.ignore_token
        if gpu_index is None:
            gpu_index = self.gpu_index

        if isinstance(target, list):
            target = torch.LongTensor(target)
            if gpu_index is not None:
                target = target.cuda(gpu_index)

        if gpu_index is None:
            log_probs = log_probs.cpu()
            target = target.cpu()

        top1_id = torch.gt(log_probs, 0.5).long()
        # top1_id = torch.squeeze(top1_id, dim=-1)

        not_equal_result = torch.ne(top1_id, target)

        if ignore_token is not None:
            target_mask = torch.ne(target, ignore_token)
            not_equal_result = not_equal_result & target_mask
        batch_error_count = not_equal_result
        for i in range(len(not_equal_result.shape)-1):
            batch_error_count = torch.sum(batch_error_count, dim=-1)

        # [batch]
        batch_result = torch.eq(batch_error_count, 0)
        batch_match_count = torch.sum(batch_result).data.item()

        batch_size = log_probs.shape[0]
        self.batch_count += batch_size
        self.match_count += batch_match_count
        return batch_match_count / batch_size

    def get_result(self):
        if self.batch_count == 0:
            return 0
        return self.match_count / self.batch_count

    def __str__(self):
        return ' SequenceBinaryClassExactMatch top 1: ' + str(self.get_result())

    def __repr__(self):
        return self.__str__()


class TokenAccuracy(Evaluator):
    def __init__(self, ignore_token=None):
        self.total_count = 0.0
        self.match_count = 0.0
        self.ignore_token = ignore_token

    def clear_result(self):
        self.total_count = 0.0
        self.match_count = 0.0

    def add_result(self, output, target, ignore_token=None, batch_data=None):
        ignore_token = ignore_token if ignore_token is not None else self.ignore_token
        output_mask = torch.ne(target, ignore_token)
        count = torch.sum(output_mask).item()
        match = torch.sum(torch.eq(output, target) & output_mask).float().item()
        self.total_count += count
        self.match_count += match
        return match/count

    def get_result(self):
        if self.total_count == 0:
            return 0
        return self.match_count / self.total_count

    def __str__(self):
        return ' TokenAccuracy top 1: ' + str(self.get_result())

    def __repr__(self):
        return self.__str__()


class SequenceCorrect(Evaluator):
    def __init__(self, ignore_token=None):
        self.total_batch = 0.0
        self.match_batch = 0.0
        self.ignore_token = ignore_token

    def clear_result(self):
        self.total_batch = 0.0
        self.match_batch = 0.0

    def add_result(self, output, target, ignore_token=None, batch_data=None):
        ignore_token = ignore_token if ignore_token is not None else self.ignore_token
        output_mask = torch.ne(target, ignore_token)
        not_equal_batch = torch.sum(torch.ne(output, target) & output_mask, dim=-1).float()
        match = torch.sum(torch.eq(not_equal_batch, 0)).float()
        batch_size = output.shape[0]
        self.total_batch += batch_size
        self.match_batch += match.item()
        return match/batch_size

    def get_result(self):
        if self.total_batch == 0:
            return 0
        return self.match_batch / self.total_batch

    def __str__(self):
        return ' SequenceCorrect top 1: ' + str(self.get_result())

    def __repr__(self):
        return self.__str__()


class SLKOutputAccuracyAndCorrect(Evaluator):
    def __init__(self, ignore_token=None, do_extract_evaluate=False):
        self.accuracy_evaluator = TokenAccuracy(ignore_token=ignore_token)
        self.point_accuracy = TokenAccuracy(ignore_token=ignore_token)
        self.correct_evaluator = SequenceCorrect(ignore_token=ignore_token)
        self.is_copy_accuracy = TokenAccuracy(ignore_token=ignore_token)
        self.do_extract_evaluate = do_extract_evaluate

    def clear_result(self):
        self.accuracy_evaluator.clear_result()
        self.correct_evaluator.clear_result()
        self.point_accuracy.clear_result()
        self.is_copy_accuracy.clear_result()

    def add_result(self, output, model_output, model_target, model_input, ignore_token=None, batch_data=None):
        target = model_target[3]
        accuracy = self.accuracy_evaluator.add_result(output, target, ignore_token=ignore_token, batch_data=batch_data)
        correct = self.correct_evaluator.add_result(output, target, ignore_token=ignore_token, batch_data=batch_data)

        if self.do_extract_evaluate:
            point_output = torch.topk(F.softmax(model_output[2], dim=-1), k=1, dim=-1)[1].squeeze(dim=-1)
            point_target = model_target[1]
            point_acc = self.point_accuracy.add_result(point_output, point_target, ignore_token=ignore_token, batch_data=batch_data)

            output_mask = model_target[4]
            is_copy_output = (model_output[0] > 0.5).float() * output_mask.float()
            is_copy_acc = self.is_copy_accuracy.add_result(is_copy_output, model_target[0],
                                             ignore_token=ignore_token, batch_data=batch_data)
        else:
            point_acc = 0
            is_copy_acc = 0

        return 'accuracy: {}, correct: {}, point_accuracy: {}, is_copy_acc: {}'\
            .format(accuracy, correct, point_acc, is_copy_acc)

    def get_result(self):
        accuracy = self.accuracy_evaluator.get_result()
        correct = self.correct_evaluator.get_result()
        point_accuracy = self.point_accuracy.get_result()
        is_copy_acc = self.is_copy_accuracy.get_result()
        return accuracy, correct, point_accuracy, is_copy_acc

    def __str__(self):
        accuracy, correct, point_accuracy, is_copy_acc = self.get_result()
        return ' SLKOutputAccuracy top 1: ' + str(accuracy) + '  SLKOutputCorrect top 1: ' + str(correct) + \
               ' point_accuracy: ' + str(point_accuracy) + ' is_copy_acc: ' + str(is_copy_acc)

    def __repr__(self):
        return self.__str__()


class EncoderCopyAccuracyAndCorrect(Evaluator):
    def __init__(self, ignore_token=None):
        self.is_copy_evaluator = SLKOutputAccuracyAndCorrect(ignore_token)
        self.sample_evaluator = SLKOutputAccuracyAndCorrect(ignore_token)
        self.all_evaluator = SLKOutputAccuracyAndCorrect(ignore_token)
        self.ignore_token = ignore_token

    def clear_result(self):
        self.is_copy_evaluator.clear_result()
        self.sample_evaluator.clear_result()
        self.all_evaluator.clear_result()

    def get_result(self):
        return self.is_copy_evaluator.get_result(), self.sample_evaluator.get_result(), self.all_evaluator.get_result()

    def __str__(self):
        is_copy, sample, all_res = self.get_result()
        return "EncoderCopy is_copy evaluate:{}, sample evaluate:{}, all evaluate:{}".format(is_copy, sample, all_res)

    def __repr__(self):
        return self.__str__()

    def add_result(self, output, model_output, model_target, model_input, ignore_token=None, batch_data=None):
        if ignore_token is None:
            ignore_token = self.ignore_token

        target_length = batch_data['target_length']
        is_copy, sample_output = output
        is_copy_target, sample_target = model_target
        is_copy_res = self.is_copy_evaluator.add_result(is_copy, None,
                                          [None, None, None, is_copy_target], None,
                                          ignore_token=ignore_token, batch_data=batch_data)

        def parse_sample(sample, sample_length):
            begin = 0
            max_length = max(sample_length)
            res = []
            for l in sample_length:
                now_fragment = sample[begin:begin+l]
                begin += l
                res.append(
                    torch.cat(
                        (now_fragment,
                         torch.ones(max_length-now_fragment.shape[0]).long().to(sample.device)*ignore_token
                         )
                    ).view(1, -1)
                )
            return torch.cat(res, dim=0)
        sample_output = parse_sample(sample_output, target_length)
        sample_target = parse_sample(sample_target, target_length)
        sample_res = self.sample_evaluator.add_result(sample_output, None,
                                         [None, None, None, sample_target], None,
                                         ignore_token=ignore_token, batch_data=batch_data)
        all_res = self.all_evaluator.add_result(torch.cat((is_copy, sample_output.float()), dim=-1), None,
                                         [None, None, None, torch.cat((is_copy_target, sample_target.float()), dim=-1)],
                                         None,
                                         ignore_token=ignore_token, batch_data=batch_data)
        return "is_copy evaluate:{}, sample evaluate:{}, all evaluate:{}".format(is_copy_res,
                                                                                 sample_res,
                                                                                 all_res)


class ErrorPositionAndValueAccuracy(Evaluator):
    def __init__(self, ignore_token=None):
        self.ignore_token = ignore_token
        self.is_copy_accuracy = TokenAccuracy(ignore_token=ignore_token)
        self.position_correct = SequenceCorrect(ignore_token=ignore_token)
        self.output_accuracy = TokenAccuracy(ignore_token=ignore_token)
        self.all_correct = SequenceCorrect(ignore_token=ignore_token)

    def add_result(self, output, model_output, model_target, model_input, ignore_token=None, batch_data=None):
        model_output = [t.data for t in model_output]
        if ignore_token is None:
            ignore_token = self.ignore_token
        is_copy = (torch.sigmoid(model_output[2]) > 0.5).float()
        is_copy_target = model_target[2]
        is_copy_accuracy = self.is_copy_accuracy.add_result(is_copy, is_copy_target)
        p0 = torch.topk(F.softmax(model_output[0], dim=-1), dim=-1, k=1)[1]
        p1 = torch.topk(F.softmax(model_output[1], dim=-1), dim=-1, k=1)[1]
        position = torch.cat([p0, p1], dim=1)
        position_target = torch.stack([model_target[0], model_target[1]], dim=1)
        position_correct = self.position_correct.add_result(position, position_target)

        all_output, sample_output_ids = output
        target_output = to_cuda(torch.LongTensor(PaddedList(batch_data['target'], fill_value=ignore_token)))
        sample_output_ids, target_output = expand_tensor_sequence_to_same(sample_output_ids, target_output[:, 1:])
        output_accuracy = self.output_accuracy.add_result(sample_output_ids, target_output)

        full_output_target = to_cuda(
            torch.LongTensor(PaddedList(batch_data['full_output_target'], fill_value=ignore_token)))
        all_output, full_output_target = expand_tensor_sequence_to_same(all_output, full_output_target, fill_value=ignore_token)
        all_correct = self.all_correct.add_result(all_output, full_output_target)
        return "is_copy_accuracy evaluate:{}, position_correct evaluate:{}, output_accuracy evaluate:{}, " \
               "all_correct evaluate: {}".format(is_copy_accuracy, position_correct, output_accuracy, all_correct)

    def clear_result(self):
        self.is_copy_accuracy.clear_result()
        self.position_correct.clear_result()
        self.output_accuracy.clear_result()
        self.all_correct.clear_result()

    def get_result(self):
        return self.is_copy_accuracy.get_result(), self.position_correct.get_result(), \
               self.output_accuracy.get_result(), self.all_correct.get_result()

    def __str__(self):
        is_copy, position, out, correct = self.get_result()
        return "ErrorPositionAndValueAccuracy is_copy evaluate:{}, position evaluate:{}, " \
               "output evaluate:{}, all evaluate: {}".format(is_copy, position, out, correct)

    def __repr__(self):
        return self.__str__()


class CompileResultEvaluate(Evaluator):
    """
    statistics compile result. It is a special evaluator. Do not use it in evaluator_list directly.
    """
    def __init__(self):
        self.total_batch = 0.0
        self.match_batch = 0.0

    def clear_result(self):
        self.total_batch = 0.0
        self.match_batch = 0.0

    def add_result(self, result_list):
        match = sum(result_list)
        batch_size = len(result_list)
        self.match_batch += match
        self.total_batch += batch_size
        return 'step compile result: {}'.format(match/batch_size)

    def get_result(self):
        if self.total_batch == 0:
            return 0
        return self.match_batch / self.total_batch

    def __str__(self):
        return ' CompileResultEvaluate: ' + str(self.get_result())

    def __repr__(self):
        return self.__str__()


class SensibilityRNNEvaluator(Evaluator):
    def __init__(self, ignore_token=None):
        self.ignore_token = ignore_token
        self.forward_accuracy = TokenAccuracy(ignore_token=ignore_token)
        self.backward_accuracy = TokenAccuracy(ignore_token=ignore_token)

    def add_result(self, output_ids, model_output, model_target, model_input, batch_data):
        forward_ids = torch.squeeze(torch.topk(F.log_softmax(model_output[0], dim=-1), k=1, dim=-1)[1], dim=-1)
        backward_ids = torch.squeeze(torch.topk(F.log_softmax(model_output[1], dim=-1), k=1, dim=-1)[1], dim=-1)
        for_acc = self.forward_accuracy.add_result(forward_ids, model_target[0])
        back_acc = self.backward_accuracy.add_result(backward_ids, model_target[1])
        return 'SensibilityRNNEvaluator forward_accuracy: {}, backward_accuracy: {}'.format(for_acc, back_acc)

    def clear_result(self):
        self.forward_accuracy.clear_result()
        self.backward_accuracy.clear_result()

    def get_result(self):
        return self.forward_accuracy.get_result(), self.backward_accuracy.get_result()

    def __str__(self):
        for_acc, back_acc = self.get_result()
        return 'SensibilityRNNEvaluator forward_accuracy: {}, backward_accuracy: {}'.format(for_acc, back_acc)

    def __repr__(self):
        return self.__str__()



