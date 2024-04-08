// This file is compiled as C++ and linked with stdlib
// into the simulation executable.
#define DEBUG 1

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <random>

struct config {
  int num_features;
  int hypervector_dim;
  int num_classes;
  // int width_classes;
  // int retrain_label;
};

class sim_hdnn_reram {
private:
  float *reram_array = nullptr;

  // Manual seed
  std::mt19937 rnd_gen;
  std::lognormal_distribution<float> dist_hrs;
  std::lognormal_distribution<float> dist_lrs;

  float *enc_mat_1; // Encoding weight matrix with shape (f1, d1)
  float *enc_mat_2; // Encoding weight matrix with shape (f2, d2)
  uint32_t f1, d1, f2, d2;

  void dim_prime_fact();
  void init_buffer();
  void init_reram_array();
  inline int adc_sensing(float inp);

public:
  uint32_t dim_feature;
  uint32_t dim_hv;
  uint32_t num_class;

  uint32_t cnt_enc;
  uint32_t cnt_hamming_search;
  uint32_t bits_data_read;
  uint32_t bits_data_program;
  uint32_t bits_data_write_sram;

  int16_t *feature_mem = nullptr;
  int16_t *class_mem = nullptr;
  int16_t *score_mem = nullptr;

  sim_hdnn_reram(uint32_t dim_f, uint32_t dim_d, uint32_t num_c);
  ~sim_hdnn_reram();

  void allocate_base_mem(int16_t *BasePtr, size_t NumBytes);
  void allocate_feature_mem(int16_t *FeatureMem, size_t NumBytes);
  void allocate_class_mem(int16_t *ClassMem, size_t NumBytes);

  void read_class_mem(int16_t *ClassMem, size_t NumBytes);
  void read_score_mem(int16_t *ScoreMem, size_t NumBytes);

  void init_enc_mat();
  void encode_hypervector();
  void enc_kronecker(int16_t *dst_ptr, int16_t *inp_feature);
  void hamming_distance(int16_t *result, int16_t *encoded_query,
                        bool reram_comp);
  inline int readout_reram_bit(uint32_t i, uint32_t j);
  inline void program_reram_bit(bool inp_bit, uint32_t i, uint32_t j);
  inline void update_class_hvs(int16_t *ptr_dst, int offset, uint64_t dim,
                               bool if_add);

  void calc_energy_runtime(float *energy_nJ, float *runtime_ns);
};

sim_hdnn_reram::sim_hdnn_reram(uint32_t dim_f, uint32_t dim_d, uint32_t num_c) {
  // Initialize HD params
  dim_feature = dim_f;
  dim_hv = dim_d;
  num_class = num_c;

  assert(dim_hv != 0 && "Non zero-dimensional encoding required!");

  // Initialize Kronecker encoder
  srand(0);
  dim_prime_fact();
  init_enc_mat();

  // Initialize reram array and buffers
  init_buffer();
  init_reram_array();

  printf("Initialized sim_hdnn_reram with %d features, %d HD dims. and %d "
         "classes...\n",
         dim_feature, dim_hv, num_class);

  // Initialize performance counters
  cnt_enc = 0;
  cnt_hamming_search = 0;
  bits_data_read = 0;
  bits_data_program = 0;
  bits_data_write_sram = 0;

#ifdef DEBUG
  printf("f1=%d\tf2=%d\td1=%d\td2=%d\n", f1, f2, d1, d2);
#endif
}

sim_hdnn_reram::~sim_hdnn_reram() {

  if (enc_mat_1)
    delete[] enc_mat_1;

  if (enc_mat_2)
    delete[] enc_mat_2;

  if (reram_array)
    delete[] reram_array;

  if (feature_mem)
    delete[] feature_mem;

  if (class_mem)
    delete[] class_mem;

  if (score_mem)
    delete[] score_mem;
}

void sim_hdnn_reram::allocate_base_mem(int16_t *BasePtr, size_t NumBytes) {
  // Do nothing: use internal base matrix for encoding
}

void sim_hdnn_reram::allocate_feature_mem(int16_t *FeatureMem,
                                          size_t NumBytes) {
  std::memcpy(feature_mem, FeatureMem, NumBytes);
}

void sim_hdnn_reram::allocate_class_mem(int16_t *ClassMem, size_t NumBytes) {
  std::memcpy(class_mem, ClassMem, NumBytes);
}

void sim_hdnn_reram::read_class_mem(int16_t *ClassMem, size_t NumBytes) {
  std::memcpy(ClassMem, class_mem, NumBytes);
  bits_data_read += (NumBytes * 8);
}

void sim_hdnn_reram::read_score_mem(int16_t *ScoreMem, size_t NumBytes) {
  std::memcpy(ScoreMem, this->score_mem, NumBytes);
  bits_data_read += (NumBytes * 8);
}

void sim_hdnn_reram::dim_prime_fact() {
  uint32_t n = sqrt(dim_feature);

  // Decompose feature dim.
  while (1) {
    if (dim_feature % n == 0) {
      f1 = n;
      assert(f1 != 0 && "Division by zero!");
      f2 = dim_feature / f1;
      break;
    }
    n--;
  }

  // Decompose HV dim.
  n = sqrt(dim_hv);
  while (1) {
    if (dim_hv % n == 0) {
      d1 = n;
      assert(d1 != 0 && "Division by zero!");
      d2 = dim_hv / d1;
      break;
    }
    n--;
  }
}

void sim_hdnn_reram::init_enc_mat() {
  enc_mat_1 = new float[f1 * d1];
  for (int i = 0; i < f1; i++) {
    for (int j = 0; j < d1; j++) {
      int rnd = rand() % 2;
      enc_mat_1[i * d1 + j] = rnd * 2.0 - 1;
    }
  }

  enc_mat_2 = new float[f2 * d2];
  for (int i = 0; i < f2; i++) {
    for (int j = 0; j < d2; j++) {
      int rnd = rand() % 2;
      enc_mat_2[i * d2 + j] = rnd * 2.0 - 1;
    }
  }
}

void sim_hdnn_reram::init_buffer() {
  feature_mem = new int16_t[dim_feature];
  class_mem = new int16_t[static_cast<uint64_t>(dim_hv) * num_class];
  score_mem = new int16_t[num_class];
}

void sim_hdnn_reram::init_reram_array() {
  this->rnd_gen = std::mt19937(0);
  this->dist_lrs = std::lognormal_distribution<float>(1.6, 0.25);
  this->dist_hrs = std::lognormal_distribution<float>(1.6, 0.25);

  this->reram_array = new float[this->dim_hv * this->num_class];

  memset(this->reram_array, 0.0,
         this->dim_hv * this->num_class * sizeof(float));
}

void sim_hdnn_reram::enc_kronecker(int16_t *dst_ptr, int16_t *inp_feature) {
  float *tmp = new float[f2 * d1];

  // Kronecker 1 encoding (f2,f1) x (f1,d1) -> (f2,d1)
  for (int i = 0; i < f2; i++) {
    for (int j = 0; j < d1; j++) {
      tmp[i * d1 + j] = 0.0;
      for (int k = 0; k < f1; k++) {
        tmp[i * d1 + j] += inp_feature[i * f1 + k] * enc_mat_1[k * d1 + j];
        // printf("tmp @ %d, %d, %d = %f x %f = %f\n", i, j, k, inp_feature[i *
        // f1 + k], enc_mat_1[k * d1 + j], tmp[i * d1 + j]);
      }
      // printf("tmp @ %d, %d = %f\n", i, j, tmp[i * d1 + j]);
    }
  }

  // Kronecker 2 encoding: (f2,d1) x (f2,d2) -> (d1,d2)
  for (int i = 0; i < d1; i++) {
    for (int j = 0; j < d2; j++) {
      float t = 0.0;
      for (int k = 0; k < f2; k++) {
        t += tmp[k * d1 + i] * enc_mat_2[k * d2 + j];
      }
      dst_ptr[i * d2 + j] = t >= 0 ? 1 : -1; // Binarized encoded results
    }
  }

  if (tmp)
    delete[] tmp;

  cnt_enc++;
}

inline int sim_hdnn_reram::adc_sensing(float inp) {
  // return inp < 2000 ? 1 : 0;
  return inp >= 0 ? 1 : -1;
}

inline int sim_hdnn_reram::readout_reram_bit(uint32_t i, uint32_t j) {
  return this->adc_sensing(this->reram_array[i * dim_hv + j]);
}

inline void sim_hdnn_reram::program_reram_bit(bool inp_bit, uint32_t i,
                                              uint32_t j) {
  // this->reram_array[i * this->dim_hv + j] =
  // inp_bit ? this->dist_lrs(this->rnd_gen) : this->dist_hrs(this->rnd_gen);
  this->reram_array[i * this->dim_hv + j] = inp_bit ? 1 : -1;

  bits_data_program++;
}

inline void sim_hdnn_reram::update_class_hvs(int16_t *ptr_dst, int offset,
                                             uint64_t dim, bool if_add) {

  for (int j = 0; j < dim; j++) {
    if (if_add)
      class_mem[offset * dim + j] += ptr_dst[j];
    else
      class_mem[offset * dim + j] -= ptr_dst[j];
  }

  bits_data_write_sram += dim * 2;
}

void sim_hdnn_reram::hamming_distance(int16_t *result, int16_t *encoded_query,
                                      bool reram_comp = false) {
  if (reram_comp) {
    *result = 0;
  } else {
    for (uint32_t c = 0; c < this->num_class; c++) {
      result[c] = 0;
      for (uint32_t j = 0; j < this->dim_hv; j++) {
        result[c] += this->readout_reram_bit(c, j) == encoded_query[j] ? 0 : 1;
      }
    }
  }

  cnt_hamming_search++;
}

void sim_hdnn_reram::calc_energy_runtime(float *energy_nJ, float *runtime_ns) {
  // Calculate energy
  // float reram_out_nJ_per_op = 1 / (5.34 * 1e12 / 1e9); // nJ/op
  // float reram_out_nJ_per_op = 1e3 / 4.23; // nJ/op
  // float reram_out_nJ_per_op = 1.0/1000.0; // nJ/op
  // float reram_program_nJ_per_bit = 5.0375;
  // float asic_nJ_per_enc = (f1+d2)*f2*d1*0.02230; // nJ/op
  // float sram_write_nJ_per_bit = 0.3125/1000.0;

  // (*energy_nJ) =
  //     (cnt_enc * asic_nJ_per_enc) +
  //     (cnt_hamming_search * (num_class * dim_hv) * reram_out_nJ_per_op) +
  //     (bits_data_write_sram * sram_write_nJ_per_bit) +
  //     (bits_data_program * reram_program_nJ_per_bit) +
  //     (bits_data_read * reram_out_nJ_per_op);

  (*energy_nJ) = 0;

  // Calculate runtime
  float asic_ns_per_enc = (f1+d2)*f2*d1*1.0/192*4; // ns
  float reram_ns_per_hamming = 1.0/128;
  float reram_program_ns_per_bit = 800.0/512;
  float reram_read_ns_per_bit = 1.0/6.0;
  float sram_write_ns_per_bit = 4.0/64.0;

  (*runtime_ns) = (cnt_enc * asic_ns_per_enc) +
                  (cnt_hamming_search * (num_class * dim_hv) * reram_ns_per_hamming) +
                  (bits_data_program * reram_program_ns_per_bit) +
                  (bits_data_read * reram_read_ns_per_bit) +
                  (bits_data_write_sram * sram_write_ns_per_bit);
}