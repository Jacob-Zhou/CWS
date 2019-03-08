import pickle
import torch.nn as nn
from optimizer import Optimizer
from parser_model import ParserModel
from dataset import Dataset
import shutil
import os
import torch
from vocab import VocabDict
import time

class Parser(object):
    def __init__(self, conf):
        self._conf = conf
        self._torch_device = torch.device(self._conf.device)
        # self._cpu_device = torch.device('cpu')
        self._use_cuda, self._cuda_device = ('cuda' == self._torch_device.type, self._torch_device.index)
        if self._use_cuda:
            # please note that the index is the relative index in CUDA_VISIBLE_DEVICES=6,7 (0, 1)  
            assert 0 <= self._cuda_device < 8
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self._cuda_device)
            # an alternative way: CUDA_VISIBLE_DEVICE=6 python ../src/main.py ...
            self._cuda_device = 0
        self._optimizer = None
        self._use_bucket = (self._conf.max_bucket_num > 1)
        self._train_datasets = []
        self._dev_datasets = []
        self._test_datasets = []
        self._char_dict = VocabDict('chars')
        self._bichar_dict = VocabDict('bichars')
        # there may be more than one label dictionaries
        self._label_dict = VocabDict('labels')

        self._eval_metrics = EvalMetrics()
        self._parser_model = ParserModel('ws', conf, self._use_cuda)

    def run(self):
        if self._conf.is_train:
            self.open_and_load_datasets(self._conf.train_files, self._train_datasets,
                                        inst_num_max=self._conf.inst_num_max)
            if self._conf.is_dictionary_exist is False:
                print("create dict...")
                for dataset in self._train_datasets:
                    self.create_dictionaries(dataset, self._label_dict)
                self.save_dictionaries(self._conf.dict_dir)
                self.load_dictionaries(self._conf.dict_dir)
                self._parser_model.init_models(self._char_dict.size(), self._bichar_dict.size(), self._label_dict.size())
                self._parser_model.reset_parameters()
                self._parser_model.save_model(self._conf.model_dir, 0)
                return

        self.load_dictionaries(self._conf.dict_dir)
        self._parser_model.init_models(self._char_dict.size(), self._bichar_dict.size(), self._label_dict.size())

        if self._conf.is_train:
            self.open_and_load_datasets(self._conf.dev_files, self._dev_datasets,
                                        inst_num_max=self._conf.inst_num_max)

        self.open_and_load_datasets(self._conf.test_files, self._test_datasets,
                                    inst_num_max=self._conf.inst_num_max)

        print('numeralizing [and pad if use-bucket] all instances in all datasets', flush=True)
        for dataset in self._train_datasets + self._dev_datasets + self._test_datasets:
            self.numeralize_all_instances(dataset, self._label_dict)
            if self._use_bucket:
                self.pad_all_inst(dataset)

        if self._conf.is_train:
            self._parser_model.load_model(self._conf.model_dir, 0)
        else:
            self._parser_model.load_model(self._conf.model_dir, self._conf.model_eval_num)

        if self._use_cuda:
            # self._parser_model.cuda()
            self._parser_model.to(self._cuda_device)

        if self._conf.is_train:
            assert self._optimizer is None
            self._optimizer = Optimizer([param for param in self._parser_model.parameters() if param.requires_grad], self._conf)
            self.train()
            return

        assert self._conf.is_test
        for dataset in self._test_datasets:
            self.evaluate(dataset, output_file_name=dataset.file_name_short + '.out')
            self._eval_metrics.compute_and_output(self._test_datasets[0], self._conf.model_eval_num)
            self._eval_metrics.clear()


def train(self):
    update_step_cnt, eval_cnt, best_eval_cnt, best_accuracy = 0, 0, 0, 0.
    self._eval_metrics.clear()
    self.set_training_mode(is_training=True)
    while True:
        inst_num = self.train_or_eval_one_batch(self._train_datasets[0], is_training=True)
        assert inst_num > 0
        update_step_cnt += 1
        print('.', end='', flush=True)

        if 0 == update_step_cnt % self._conf.eval_every_update_step_num:
            eval_cnt += 1
            self._eval_metrics.compute_and_output(self._train_datasets[0], eval_cnt)
            self._eval_metrics.clear()

            self.evaluate(self._dev_datasets[0])
            self._eval_metrics.compute_and_output(self._dev_datasets[0], eval_cnt)
            current_las = self._eval_metrics.las
            self._eval_metrics.clear()

            if best_accuracy < current_las - 1e-3:
                if eval_cnt > self._conf.save_model_after_eval_num:
                    if best_eval_cnt > self._conf.save_model_after_eval_num:
                        self.del_model(self._conf.model_dir, best_eval_cnt)
                    self._parser_model.save_model(self._conf.model_dir, eval_cnt)
                    self.evaluate(self._test_datasets[0], output_file_name=None)
                    self._eval_metrics.compute_and_output(self._test_datasets[0], eval_cnt)
                    self._eval_metrics.clear()

                best_eval_cnt = eval_cnt
                best_accuracy = current_las

            self.set_training_mode(is_training=True)

        if (best_eval_cnt + self._conf.train_stop_after_eval_num_no_improve < eval_cnt) or \
                (eval_cnt > self._conf.train_max_eval_num):
            break


def train_or_eval_one_batch(self, dataset, is_training):
    one_batch, total_word_num, max_len = dataset.get_one_batch(rewind=is_training)
    # NOTICE: total_word_num does not include w_0
    inst_num = len(one_batch)
    if 0 == inst_num:
        return 0

    chars, bichars, gold_labels = self.compose_batch_data_variable(one_batch, max_len)

    time1 = time.time()
    mlp_out = self._parser_model(chars, bichars, gold_labels)
    time2 = time.time()

    label_loss = self._parser_model.get_loss(mlp_out, gold_labels) / total_word_num
    self._eval_metrics.loss_accumulated += label_loss.item()
    time3 = time.time()

    if is_training:
        label_loss.backward()
        nn.utils.clip_grad_norm_(self._all_params_requires_grad, max_norm=self._conf.clip)
        self._optimizer.step()
        self.zero_grad()
    time4 = time.time()

    # self.decode(arc_scores_for_decode, label_scores_for_decode, one_batch, self._label_dict)
    time5 = time.time()

    self._eval_metrics.forward_time += time2 - time1
    self._eval_metrics.loss_time += time3 - time2
    self._eval_metrics.backward_time += time4 - time3
    self._eval_metrics.decode_time += time5 - time4
    return inst_num


def evaluate(self, dataset, output_file_name=None):
    with torch.no_grad():
        self.set_training_mode(is_training=False)
        while True:
            inst_num = self.train_or_eval_one_batch(dataset, is_training=False)
            print('.', end='', flush=True)
            if 0 == inst_num:
                break

        if output_file_name is not None:
            with open(output_file_name, 'w', encoding='utf-8') as out_file:
                all_inst = dataset.all_inst
                for inst in all_inst:
                    inst.write(out_file)


@staticmethod
def decode_one_inst(args):
    inst, mlp_out_scores, label_scores, max_label_prob_as_arc_prob, viterbi_decode = args
    length = inst.size()
    # for labeled-crf-loss, the default is sum of label prob, already stored in arc_scores
    if max_label_prob_as_arc_prob:
        '''
        N = arc_scores.shape[0]
        assert N == label_scores.shape[0] == label_scores.shape[1]
        L = label_scores.shape[2]
        label_scores2d = label_scores.reshape(N2, L)
        label_i_max1d = np.argmax(label_scores2d, axis=1)
        # label_score_max1d = np.array([label_scores2d[i, label_i_max1d[i]] for i in range(N2)])
        label_score_max1d = np.array([label_scores2d[i, label_i_max1d[i]] for i in range(N2)])
        arc_scores = label_score_max1d.reshape(N, N)
        '''
        arc_scores = np.max(label_scores, axis=2)

    if viterbi_decode:
        head_pred = crf_loss.viterbi(length, arc_scores, False, inst.candidate_heads)
    else:
        head_pred = np.argmax(arc_scores, axis=1)  # mod-head order issue. BE CAREFUL

    label_score_of_concern = label_scores[np.arange(inst.size()), head_pred[:inst.size()]]
    label_pred = np.argmax(label_score_of_concern, axis=1)
    # Parser.set_predict_result(inst, head_pred, label_pred, label_dict)
    # return inst.eval()
    return head_pred, label_pred


''' 2018.11.3 by Zhenghua
I found that using multi-thread for non-viterbi (local) decoding is actually much slower than single-thread (ptb labeled-crf-loss train 1-iter: 150s vs. 5s)
NOTICE: 
    multi-process: CAN NOT Parser.set_predict_result(inst, head_pred, label_pred, label_dict), this will not change inst of the invoker 
'''


def decode(self, arc_scores, label_scores, one_batch, label_dict):
    inst_num = arc_scores.shape[0]
    assert inst_num == len(one_batch)
    if self._conf.multi_thread_decode:
        data_for_pool = [
            (inst, arc_score, label_score, self._conf.max_label_prob_as_arc_prob_when_decode, self._conf.viterbi_decode)
            for (inst, arc_score, label_score) in zip(one_batch, arc_scores, label_scores)]
        # data_for_pool = [(one_batch[i], arc_scores[i], label_scores[i], self._conf.max_label_prob_as_arc_prob_when_decode, self._conf.viterbi_decode) for i in range(inst_num)]
        with Pool(self._conf.cpu_thread_num) as thread_pool:
            ret = thread_pool.map(Parser.decode_one_inst, data_for_pool)
            thread_pool.close()
            thread_pool.join()
    else:
        ret = [Parser.decode_one_inst((inst, arc_score, label_score, self._conf.max_label_prob_as_arc_prob_when_decode,
                                       self._conf.viterbi_decode))
               for (inst, arc_score, label_score) in zip(one_batch, arc_scores, label_scores)]
        '''
        for (arc_score, label_score, inst) in zip(arc_scores, label_scores, one_batch):
            Parser.decode_one_inst((inst, arc_score, label_score, label_dict, self._conf.max_label_prob_as_arc_prob_when_decode, self._conf.viterbi_decode)) 
            arc_pred = np.argmax(arc_score, axis=1)   # mod-head order issue. BE CAREFUL
            label_score_of_concern = label_score[np.arange(inst.size()), arc_pred[:inst.size()]]
            label_pred = np.argmax(label_score_of_concern, axis=1)
            Parser.set_predict_result(inst, arc_pred, label_pred, label_dict)
        '''
    self._eval_metrics.sent_num += len(one_batch)
    for (inst, preds) in zip(one_batch, ret):
        Parser.set_predict_result(inst, preds[0], preds[1], label_dict)
        Parser.compute_accuracy_one_inst(inst, self._eval_metrics)


def create_dictionaries(self, dataset, label_dict):
    all_inst = dataset.all_inst
    for inst in all_inst:
        for i in range(1, inst.size()):
            self._word_dict.add_key_into_counter(inst.words_s[i])
            self._tag_dict.add_key_into_counter(inst.tags_s[i])
            if inst.heads_i[i] != ignore_label_id:
                label_dict.add_key_into_counter(inst.labels_s[i])


@staticmethod
def get_candidate_heads(length, gold_arcs):
    candidate_heads = np.array([0] * length * length, dtype=data_type_int32).reshape(length, length)
    for m in range(1, length):
        h = gold_arcs[m]
        if h < 0:
            for i in range(length):
                candidate_heads[m][i] = 1
        else:
            candidate_heads[m][h] = 1
    return candidate_heads


def numeralize_all_instances(self, dataset, label_dict):
    all_inst = dataset.all_inst
    for inst in all_inst:
        for i in range(0, inst.size()):
            inst.words_i[i] = self._word_dict.get_id(inst.words_s[i])
            inst.ext_words_i[i] = self._ext_word_dict.get_id(inst.words_s[i])
            inst.tags_i[i] = self._tag_dict.get_id(inst.tags_s[i])
            if inst.heads_i[i] != ignore_id_head_or_label:
                inst.labels_i[i] = label_dict.get_id(inst.labels_s[i])
        if (self._conf.use_unlabeled_crf_loss or self._conf.use_labeled_crf_loss) and (
                self._conf.use_sib_score or inst.is_partially_annotated()):
            assert inst.candidate_heads is None
            inst.candidate_heads = Parser.get_candidate_heads(inst.size(), inst.heads_i)



def load_dictionaries(self, path):
    path = os.path.join(path, 'dict/')
    assert os.path.exists(path)
    self._char_dict.load(path + self._word_dict.name, cutoff_freq=self._conf.word_freq_cutoff,
                         default_keys_ids=((padding_str, padding_id), (unknown_str, unknown_id)))
    self._bichar_dict.load(path + self._tag_dict.name,
                        default_keys_ids=((padding_str, padding_id), (unknown_str, unknown_id)))
    self._label_dict.load(path + self._label_dict.name, default_keys_ids=())
    print("load  dict done", flush=True)


def save_dictionaries(self, path):
    path = os.path.join(path, 'dict/')
    assert os.path.exists(path) is False
    os.mkdir(path)
    self._chars_dict.save(path + self._chars_dict.name)
    self._bichars_dict.save(path + self._bichars_dict.name)
    self._label_dict.save(path + self._label_dict.name)
    print("save dict done", flush=True)


@staticmethod
def del_model(path, eval_num):
    path = os.path.join(path, 'models-%d/' % eval_num)
    if os.path.exists(path):
        # os.rmdir(path)
        shutil.rmtree(path)
        print('Delete model %s done.' % path)
    else:
        print('Delete model %s error, not exist.' % path)


def open_and_load_datasets(self, file_names, datasets, inst_num_max):
    assert len(datasets) == 0
    names = file_names.strip().split(':')
    assert len(names) > 0
    for name in names:
        datasets.append(Dataset(name, max_bucket_num=self._conf.max_bucket_num,
                                word_num_one_batch=self._conf.word_num_one_batch,
                                sent_num_one_batch=self._conf.sent_num_one_batch,
                                inst_num_max=inst_num_max,
                                max_len=self._conf.sent_max_len))


@staticmethod
def set_predict_result(inst, arc_pred, label_pred, label_dict):
    # assert arc_pred.size(0) == inst.size()
    for i in np.arange(1, inst.size()):
        inst.heads_i_predict[i] = arc_pred[i]
        inst.labels_i_predict[i] = label_pred[i]
        inst.labels_s_predict[i] = label_dict.get_str(inst.labels_i_predict[i])


'''
@staticmethod
def update_accuracy(stats, eval_metrics):
    eval_metrics.sent_num += len(stats)
    for (word_num, a, b, c) in stats:
        eval_metrics.word_num += word_num
        eval_metrics.word_num_to_eval += a
        eval_metrics.word_num_correct_arc += b
        eval_metrics.word_num_correct_label += c

@staticmethod
def compute_accuracy(one_batch, eval_metrics):
    eval_metrics.sent_num += len(one_batch)
    for inst in one_batch:
        Parser.compute_accuracy_one_inst(inst, eval_metrics)
'''


@staticmethod
def compute_accuracy_one_inst(inst, eval_metrics):
    word_num, a, b, c = inst.eval()
    eval_metrics.word_num += word_num
    eval_metrics.word_num_to_eval += a
    eval_metrics.word_num_correct_arc += b
    eval_metrics.word_num_correct_label += c


def set_training_mode(self, is_training=True):
    self._parser_model.train(mode=is_training)


def zero_grad(self):
    self._parser_model.zero_grad()


def pad_all_inst(self, dataset):
    for (max_len, inst_num_one_batch, this_bucket) in dataset.all_buckets:
        for inst in this_bucket:
            assert inst.lstm_mask is None
            inst.words_i, inst.ext_words_i, inst.tags_i, inst.heads_i, inst.labels_i, inst.lstm_mask = \
                self.pad_one_inst(inst, max_len)


def pad_one_inst(self, inst, max_sz):
    sz = inst.size()
    assert len(inst.words_i) == sz
    assert max_sz >= sz
    pad_sz = (0, max_sz - sz)
    '''
    return torch.from_numpy(np.pad(inst.words_i, pad_sz, 'constant', constant_values=padding_id)), \
           torch.from_numpy(np.pad(inst.ext_words_i, pad_sz, 'constant', constant_values=padding_id)), \
           torch.from_numpy(np.pad(inst.tags_i, pad_sz, 'constant', constant_values=padding_id)), \
           torch.from_numpy(np.pad(inst.heads_i, pad_sz, 'constant', constant_values=ignore_id_head_or_label)), \
           torch.from_numpy(np.pad(inst.labels_i, pad_sz, 'constant', constant_values=ignore_id_head_or_label)), \
           torch.from_numpy(np.pad(np.ones(sz, dtype=data_type), pad_sz, 'constant', constant_values=padding_id))
    '''
    return np.pad(inst.words_i, pad_sz, 'constant', constant_values=0), \
           np.pad(inst.ext_words_i, pad_sz, 'constant', constant_values=0), \
           np.pad(inst.tags_i, pad_sz, 'constant', constant_values=0), \
           np.pad(inst.heads_i, pad_sz, 'constant', constant_values=ignore_id_head_or_label), \
           np.pad(inst.labels_i, pad_sz, 'constant', constant_values=ignore_id_head_or_label), \
           np.pad(np.ones(sz, dtype=data_type), pad_sz, 'constant', constant_values=0)


def compose_batch_data_variable(self, one_batch, max_len):
    words, ext_words, tags, heads, labels, lstm_masks = [], [], [], [], [], []
    for inst in one_batch:
        if self._use_bucket:
            words.append(inst.words_i)
            ext_words.append(inst.ext_words_i)
            tags.append(inst.tags_i)
            heads.append(inst.heads_i)
            labels.append(inst.labels_i)
            lstm_masks.append(inst.lstm_mask)
        else:
            ret = self.pad_one_inst(inst, max_len)
            words.append(ret[0])
            ext_words.append(ret[1])
            tags.append(ret[2])
            heads.append(ret[3])
            labels.append(ret[4])
            lstm_masks.append(ret[5])
    # dim: batch max-len
    words, ext_words, tags, heads, labels, lstm_masks = \
        torch.from_numpy(np.stack(words, axis=0)), torch.from_numpy(np.stack(ext_words, axis=0)), \
        torch.from_numpy(np.stack(tags, axis=0)), torch.from_numpy(np.stack(heads, axis=0)), \
        torch.from_numpy(np.stack(labels, axis=0)), torch.from_numpy(np.stack(lstm_masks, axis=0))
    '''
    torch.stack(words, dim=0), torch.stack(ext_words, dim=0), \
    torch.stack(tags, dim=0), torch.stack(heads, dim=0), \
    torch.stack(labels, dim=0), torch.stack(lstm_masks, dim=0)
    '''

    # MUST assign for Tensor.cuda() unlike nn.Module
    if self._use_cuda:
        words, ext_words, tags, heads, labels, lstm_masks = \
            words.cuda(self._cuda_device), ext_words.cuda(self._cuda_device), \
            tags.cuda(self._cuda_device), heads.cuda(self._cuda_device), \
            labels.cuda(self._cuda_device), lstm_masks.cuda(self._cuda_device)
    return words, ext_words, tags, heads, labels, lstm_masks


class EvalMetrics(object):
    def __init__(self):
        self.clear()

    def clear(self):
        self.sent_num = 0
        self.word_num = 0
        self.word_num_to_eval = 0
        self.word_num_correct_arc = 0
        self.word_num_correct_label = 0
        self.uas = 0.
        self.las = 0.
        self.loss_accumulated = 0.
        self.start_time = time.time()
        self.time_gap = 0.
        self.forward_time = 0.
        self.loss_time = 0.
        self.backward_time = 0.
        self.decode_time = 0.

    def compute_and_output(self, dataset, eval_cnt):
        assert self.word_num > 0
        self.uas = 100. * self.word_num_correct_arc / self.word_num_to_eval
        self.las = 100. * self.word_num_correct_label / self.word_num_to_eval
        self.time_gap = float(time.time() - self.start_time)
        print(
            "\n%30s(%5d): loss=%.3f las=%.3f, uas=%.3f, %d (%d) words, %d sentences, time=%.3f (%.1f %.1f %.1f %.1f) [%s]" %
            (dataset.file_name_short, eval_cnt, self.loss_accumulated, self.las, self.uas,
             self.word_num_to_eval, self.word_num, self.sent_num, self.time_gap, self.forward_time, self.loss_time,
             self.backward_time, self.decode_time, get_time_str()), flush=True)

