import numpy as np

#HYPERPARAMETERS
EMBEDDING_DIM = 768
MAX_SEQUENCE_LENGTH = 10
NO_ATTENTION_HEADS = 8

rng = np.random.default_rng(seed=100)

def get_random_weights(matrix_dim):
    return rng.standard_normal(size=(matrix_dim, matrix_dim))

def get_random_embeddings(length):
    return rng.standard_normal(size=(1, length))

embedding_dict = {
    'i': get_random_embeddings(EMBEDDING_DIM),
    'am': get_random_embeddings(EMBEDDING_DIM),
    'indian': get_random_embeddings(EMBEDDING_DIM),
    'katy': get_random_embeddings(EMBEDDING_DIM),
    'this': get_random_embeddings(EMBEDDING_DIM),
    'perry': get_random_embeddings(EMBEDDING_DIM),
    'loves': get_random_embeddings(EMBEDDING_DIM),
    'songs': get_random_embeddings(EMBEDDING_DIM),
    'election': get_random_embeddings(EMBEDDING_DIM),
    'has': get_random_embeddings(EMBEDDING_DIM),
    'become': get_random_embeddings(EMBEDDING_DIM),
    'hard': get_random_embeddings(EMBEDDING_DIM),
    'to': get_random_embeddings(EMBEDDING_DIM),
    'predict': get_random_embeddings(EMBEDDING_DIM),
    'unk': get_random_embeddings(EMBEDDING_DIM)
}

sentences = [
    ['i', 'am', 'indian'],
    ['katy', 'perry', 'loves', 'songs'],
    ['this', 'election', 'has', 'become', 'hard', 'to', 'predict']
]

def get_emb_matrix(sentences):
    sentence_matrix = []
    no_sentences = len(sentences)

    for sentence in sentences:
        sentence = sentence + (MAX_SEQUENCE_LENGTH - len(sentence)) * ['unk']
        single_sentence_matrix = np.vstack([embedding_dict[word] for word in sentence])
        sentence_matrix.append(single_sentence_matrix)

    sentence_matrix = np.vstack(sentence_matrix)
    sentence_matrix_batch = sentence_matrix.reshape(no_sentences, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM  )
    
    return sentence_matrix_batch

def softmax(matrix):
    exp = np.exp(matrix - np.max(matrix, axis=-1, keepdims=True))
    sum_exp = np.sum(exp, axis=-1, keepdims=True)
    softmax_matrix = exp / sum_exp
    return softmax_matrix
    
def self_attention(q, k , v):
    dim_k = k.shape[1]
    scores = np.matmul(q, np.transpose(k, (0,2,1)))
    scores = scores / np.sqrt(dim_k)
    normalized_scores = softmax(scores)
    values_with_attention = np.matmul(normalized_scores, v)
    return values_with_attention



def get_weight_matrices(size=EMBEDDING_DIM):
    W_q, W_k, W_v = get_random_weights(size), \
                    get_random_weights(size), get_random_weights(size)
    
    return W_q, W_k, W_v

def get_qkv(sentence_batch_matrix, W_q, W_k, W_v):
    q = np.matmul(sentence_batch_matrix, W_q)
    k = np.matmul(sentence_batch_matrix, W_k)
    v = np.matmul(sentence_batch_matrix, W_v)

    return q,k,v

class Attention:

    _weights_generated = False

    def __init__(self, sentence_matrix=None):
        self.z_start = sentence_matrix

    def update_start_z(self, z_new):
        self.z_start = z_new

    def generate_layer_weight_matrices(self):
        if not self._weights_generated:
            W_q, W_k, W_v = get_weight_matrices()
            self.W_q = W_q
            self.W_k = W_k
            self.W_v = W_v
            self._weights_generated = True

    def generate_qkv(self):
        if not self._weights_generated:
            self.generate_layer_weight_matrices()

        q,k,v = get_qkv(self.z_start, self.W_q, self.W_k, self.W_v)

        return q,k,v
    
    def perform_self_attention(self):
        q,k,v = self.generate_qkv()

        self.z_end = self_attention(q,k,v)

        return self.z_end
    
class MultiHeadAttention:

    _attention_block_initialized = False

    def __init__(self, z_start):
        self.z_start = z_start

    def multihead_attention_block(self):
        if not self._attention_block_initialized:
            attention_heads = []
            for i in range(NO_ATTENTION_HEADS):
                attention_heads.append( Attention(sentence_matrix=self.z_start) )

            self.attention_heads = attention_heads
            self._attention_block_initialized = True

        else:
            for attention_block in self.attention_heads:
                attention_block.update_start_z(self.z_start)

    def perform_multihead_attention(self):
        if not self._attention_block_initialized:
            self.multihead_attention_block()

        z_outputs = []
        for attention_block in self.attention_heads:
            z_outputs.append(attention_block.perform_self_attention())

        z_end = np.hstack(z_outputs)

        return z_end

        

    

sentence_batch_matrix = get_emb_matrix(sentences=sentences)

multi_head_attention = MultiHeadAttention(z_start=sentence_batch_matrix)
z_out = multi_head_attention.perform_multihead_attention()

print(sentence_batch_matrix.shape)
print(z_out.shape)







