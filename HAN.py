#coding=utf8

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers
import time
import os

def length(sequences):
#返回一个序列中每个元素的长度
    used = tf.sign(tf.reduce_max(tf.abs(sequences), reduction_indices=2))
    seq_len = tf.reduce_sum(used, reduction_indices=1)
    return tf.cast(seq_len, tf.int32)

class HAN():

    def __init__(self, hparams):
        self.hparams = hparams
        self.vocab_size = self.hparams.vocab_size
        self.num_classes = self.hparams.num_classes
        self.embedding_size = self.hparams.embedding_size
        self.hidden_size = self.hparams.hidden_size
        self.setup_input_placeholders()
        self.setup_HAN()
        self.setup_loss()
        if self.is_training():
            self.setup_training()
            self.setup_summary()
        self.saver = tf.train.Saver(tf.global_variables())
        
    def init_model(self, sess, initializer = None):
        if initializer:
            sess.run(initializer)
        else:
            sess.run(tf.global_variables_initializer())

    def save_model(self, sess):
        return self.saver.save(sess, os.path.join(self.hparams.checkpoint_dir,
                            "model.ckpt"), global_step=self.global_step)

    def setup_input_placeholders(self):
        with tf.name_scope('placeholder'):
            self.max_sentence_num = tf.placeholder(tf.int32, name='max_sentence_num')
            self.max_sentence_length = tf.placeholder(tf.int32, name='max_sentence_length')
            self.batch_size = tf.placeholder(tf.int32, name='batch_size')
            #x的shape为[batch_size, 句子数， 句子长度(单词个数)]，但是每个样本的数据都不一样，，所以这里指定为空
            #y的shape为[batch_size, num_classes]
            self.input_x = tf.placeholder(tf.int32, [None, None, None], name='input_x')
            self.input_y = tf.placeholder(tf.int32, [None], name='input_y')
            # for training and evaluation
            if self.hparams.mode in ['train', 'eval']:
                self.dropout_keep_prob = tf.placeholder(
                    dtype=tf.float32, name='keep_prob')
            global_step = tf.Variable(
                initial_value=0,
                name="global_step",
                trainable=False,
                collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

            self.global_step = global_step

    def is_training(self):
        return self.hparams.mode == 'train'         

    def setup_summary(self):
        self.summary_writer = tf.summary.FileWriter(
            self.hparams.checkpoint_dir, tf.get_default_graph())
        tf.summary.scalar("train_loss", self.losses)
        tf.summary.scalar("accuracy_summary",self.accuracy)
        tf.summary.scalar("learning_rate", self.learning_rate)
        tf.summary.scalar('gN', self.gradient_norm)
        tf.summary.scalar('pN', self.param_norm)
        self.summary_op = tf.summary.merge_all()

    def setup_HAN(self):
        #构建模型
        word_embedded = self.word2vec()
        sent_vec = self.sent2vec(word_embedded)
        doc_vec = self.doc2vec(sent_vec)
        out = self.classifer(doc_vec)

        self.out = out

    def restore_model(self, sess, epoch = None):
        if epoch is None:
            self.saver.restore(sess, tf.train.latest_checkpoint(
                self.hparams.checkpoint_dir))
        else:
            self.saver.restore(
                sess, os.path.join(self.hparams.checkpoint_dir, "model.ckpt" + ("-%d" % epoch)))
        print("restored model")

    def word2vec(self):
        #嵌入层
        with tf.name_scope("embedding"):
            embedding_mat = tf.Variable(tf.truncated_normal((self.vocab_size, self.embedding_size)))
            #shape为[batch_size, sent_in_doc, word_in_sent, embedding_size]
            word_embedded = tf.nn.embedding_lookup(embedding_mat, self.input_x)
        return word_embedded

    def sent2vec(self, word_embedded):
        with tf.name_scope("sent2vec"):
            #GRU的输入tensor是[batch_size, max_time, ...].在构造句子向量时max_time应该是每个句子的长度，所以这里将
            #batch_size * sent_in_doc当做是batch_size.这样一来，每个GRU的cell处理的都是一个单词的词向量
            #并最终将一句话中的所有单词的词向量融合（Attention）在一起形成句子向量

            #shape为[batch_size*sent_in_doc, word_in_sent, embedding_size]
            word_embedded = tf.reshape(word_embedded, [-1, self.max_sentence_length, self.embedding_size])
            #shape为[batch_size*sent_in_doce, word_in_sent, hidden_size*2]
            word_encoded = self.BidirectionalGRUEncoder(word_embedded, name='word_encoder')
            #shape为[batch_size*sent_in_doc, hidden_size*2]
            sent_vec = self.AttentionLayer(word_encoded, name='word_attention')
            return sent_vec

    def doc2vec(self, sent_vec):
        #原理与sent2vec一样，根据文档中所有句子的向量构成一个文档向量
        with tf.name_scope("doc2vec"):
            sent_vec = tf.reshape(sent_vec, [-1, self.max_sentence_num, self.hidden_size*2])
            #shape为[batch_size, sent_in_doc, hidden_size*2]
            doc_encoded = self.BidirectionalGRUEncoder(sent_vec, name='sent_encoder')
            #shape为[batch_szie, hidden_szie*2]
            doc_vec = self.AttentionLayer(doc_encoded, name='sent_attention')
            return doc_vec

    def classifer(self, doc_vec):
        #最终的输出层，是一个全连接层
        with tf.name_scope('doc_classification'):
            out = layers.fully_connected(inputs=doc_vec, num_outputs=self.num_classes, activation_fn=None)
            return out

    def BidirectionalGRUEncoder(self, inputs, name):
        #双向GRU的编码层，将一句话中的所有单词或者一个文档中的所有句子向量进行编码得到一个 2×hidden_size的输出向量，然后在经过Attention层，将所有的单词或句子的输出向量加权得到一个最终的句子/文档向量。
        #输入inputs的shape是[batch_size, max_time, voc_size]
        with tf.variable_scope(name):
            GRU_cell_fw = rnn.GRUCell(self.hidden_size)
            GRU_cell_bw = rnn.GRUCell(self.hidden_size)
            #fw_outputs和bw_outputs的size都是[batch_size, max_time, hidden_size]
            ((fw_outputs, bw_outputs), (_, _)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=GRU_cell_fw,
                                                                                 cell_bw=GRU_cell_bw,
                                                                                 inputs=inputs,
                                                                                 sequence_length=length(inputs),
                                                                                 dtype=tf.float32)
            #outputs的size是[batch_size, max_time, hidden_size*2]
            outputs = tf.concat((fw_outputs, bw_outputs), 2)
            return outputs

    def AttentionLayer(self, inputs, name):
        #inputs是GRU的输出，size是[batch_size, max_time, encoder_size(hidden_size * 2)]
        with tf.variable_scope(name):
            # u_context是上下文的重要性向量，用于区分不同单词/句子对于句子/文档的重要程度,
            # 因为使用双向GRU，所以其长度为2×hidden_szie
            u_context = tf.Variable(tf.truncated_normal([self.hidden_size * 2]), name='u_context')
            #使用一个全连接层编码GRU的输出的到期隐层表示,输出u的size是[batch_size, max_time, hidden_size * 2]
            h = layers.fully_connected(inputs, self.hidden_size * 2, activation_fn=tf.nn.tanh)
            #shape为[batch_size, max_time, 1]
            alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(h, u_context), axis=2, keep_dims=True), dim=1)
            #reduce_sum之前shape为[batch_szie, max_time, hidden_szie*2]，之后shape为[batch_size, hidden_size*2]
            atten_output = tf.reduce_sum(tf.multiply(inputs, alpha), axis=1)
            return atten_output

    def setup_loss(self):
        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.out+1e-10, labels=self.input_y)
        self.losses = tf.reduce_mean(self.loss)
        self.prediction = tf.argmax(self.out,1,output_type=tf.int32)
        correct_prediction = tf.equal(self.prediction,self.input_y)
        # self.correct_num=tf.reduce_sum(tf.cast(correct_prediction,tf.float32))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32),name="accuracy")

    def setup_training(self):
        # learning rate decay
        if self.hparams.decay_schema == 'exp':
            self.learning_rate = tf.train.exponential_decay(self.hparams.learning_rate, self.global_step,
                                                            self.hparams.decay_steps, 0.96, staircase=True)
        else:
            self.learning_rate = tf.constant(
                self.hparams.learning_rate, dtype=tf.float32)

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        params = tf.trainable_variables()
        # we need to enable the colocate_gradients_with_ops option in tf.gradients to parallelize the gradients computation.
        gradients = tf.gradients(self.losses, params, colocate_gradients_with_ops=True if self.hparams.num_gpus>1 else False)
        # RNN中常用的梯度截断，防止出现梯度过大难以求导的现象
        clipped_gradients, _ = tf.clip_by_global_norm(
            gradients, self.hparams.max_gradient_norm)
        self.gradient_norm = tf.global_norm(gradients)
        self.param_norm = tf.global_norm(params)
        self.train_op = optimizer.apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step)

    def train_one_batch(self, sess, source, targets):
        feed_dict = {
            self.input_x: source,
            self.input_y: targets,
            self.max_sentence_num: self.hparams.max_sent_in_doc,
            self.max_sentence_length: self.hparams.max_word_in_sent,
            self.batch_size: self.hparams.batch_size
        }
        _, batch_loss,accuracy, summary, global_step= sess.run(
                [self.train_op, self.losses,self.accuracy, self.summary_op,
                    self.global_step],
                feed_dict=feed_dict)
        time_str = str(int(time.time()))
        print("{}:  global step {}, batch loss {:g}, acc {:g}".format(time_str, self.global_step, batch_loss, accuracy))
        return batch_loss,accuracy, summary, global_step

    def eval_one_batch(self, sess, source, targets):
        feed_dict = {
            self.input_x: source,
            self.input_y: targets,
            self.max_sentence_num: self.hparams.max_sent_in_doc,
            self.max_sentence_length: self.hparams.max_word_in_sent,
            self.batch_size: self.hparams.batch_size
        }
        batch_loss, accuracy = sess.run([self.losses, self.accuracy], feed_dict)
        time_str = str(int(time.time()))
        print("++++++++++++++++++dev++++++++++++++{}: loss {:g}, acc {:g}".format(time_str, batch_loss, accuracy))
        return batch_loss, accuracy