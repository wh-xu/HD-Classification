#include "api_hdnn_reram.hpp"
#include <stdlib.h>

//#define TRACE

sim_hdnn_reram *ins_hdnn_reram = nullptr;

//////////////////////////////////////////////////
//////////////// LLVM interface //////////////////
//////////////////////////////////////////////////

extern "C" {
void initialize_encoder(void *cfg_void) {
  config *cfg = (config *)cfg_void;
#ifdef TRACE
  fprintf(stderr, "initialize_encoder(%d, %d, %d)\n", cfg->num_features,
          cfg->hypervector_dim, cfg->num_classes);
#endif
  if (!ins_hdnn_reram)
    ins_hdnn_reram = new sim_hdnn_reram(cfg->num_features, cfg->hypervector_dim,
                                        cfg->num_classes);
}

void initialize_device(void *cfg_void) {
  config *cfg = (config *)cfg_void;
#ifdef TRACE
  fprintf(stderr, "initialize_device(%d, %d, %d)\n", cfg->num_features,
          cfg->hypervector_dim, cfg->num_classes);
#endif
  if (!ins_hdnn_reram)
    ins_hdnn_reram = new sim_hdnn_reram(cfg->num_features, cfg->hypervector_dim,
                                        cfg->num_classes);
}

void encode_hypervector(int16_t *dst_pointer, int16_t *input_features,
                        int input_dimension, int encoded_dimension) {
#ifdef TRACE
  fprintf(stderr, "encode_hypervector(%p, %p, %d, %d)\n", dst_pointer,
          input_features, input_dimension, encoded_dimension);
#endif
  assert(input_dimension == ins_hdnn_reram->dim_feature);
  assert(encoded_dimension == ins_hdnn_reram->dim_hv);

  ins_hdnn_reram->enc_kronecker(dst_pointer, input_features);
}

void execute_encode(void *dst_pointer_void, void *input_features_void,
                    int input_dimension, int encoded_dimension) {
#ifdef TRACE
  fprintf(stderr, "execute_encode(%p, %p, %d, %d)\n", dst_pointer_void,
          input_features_void, input_dimension, encoded_dimension);
#endif
  int16_t *dst_pointer = (int16_t *)dst_pointer_void;
  int16_t *input_features = (int16_t *)input_features_void;

  assert(input_dimension == ins_hdnn_reram->dim_feature);
  assert(encoded_dimension == ins_hdnn_reram->dim_hv);

  ins_hdnn_reram->enc_kronecker(dst_pointer, input_features);
}

void hamming_distance(void *result, void *encoded_query) {
#ifdef TRACE
  fprintf(stderr, "hamming_distance(%p, %p)\n", result, encoded_query);
#endif
  bool reram_comp = false;
  ins_hdnn_reram->hamming_distance((int16_t *)result, (int16_t *)encoded_query,
                                   reram_comp);
}

void allocate_base_mem(void *BasePtr, size_t NumBytes) {
#ifdef TRACE
  fprintf(stderr, "allocate_base_mem(%p, %lu)\n", BasePtr, NumBytes);
#endif
  // Do nothing: use internal base matrix for encoding
}

void allocate_feature_mem(void *FeatureMem, size_t NumBytes) {
#ifdef TRACE
  fprintf(stderr, "allocate_feature_mem(%p, %lu)\n", FeatureMem, NumBytes);
#endif
  ins_hdnn_reram->allocate_feature_mem((int16_t *)FeatureMem, NumBytes);
}

void allocate_class_mem(void *ClassMem, size_t NumBytes) {
#ifdef TRACE
  fprintf(stderr, "allocate_class_mem(%p, %lu)\n", ClassMem, NumBytes);
#endif
  ins_hdnn_reram->allocate_class_mem((int16_t *)ClassMem, NumBytes);
}

void read_class_mem(void *ClassMem, size_t NumBytes) {
#ifdef TRACE
  fprintf(stderr, "read_class_mem(%p, %lu)\n", ClassMem, NumBytes);
#endif
  ins_hdnn_reram->read_class_mem((int16_t *)ClassMem, NumBytes);
}

void read_score_mem(void *ScoreMem, size_t NumBytes) {
#ifdef TRACE
  fprintf(stderr, "read_score_mem(%p, %lu)\n", ScoreMem, NumBytes);
#endif
  ins_hdnn_reram->read_score_mem((int16_t *)ScoreMem, NumBytes);
}

void execute_train(int label) {
#ifdef TRACE
  fprintf(stderr, "execute_train(%d)\n", label);
#endif
  uint32_t dim_hv = ins_hdnn_reram->dim_hv;
  uint32_t dim_feature = ins_hdnn_reram->dim_feature;
  uint32_t num_class = ins_hdnn_reram->num_class;

  // Compute aggregate class HVs
  int16_t *ptr_dst = new int16_t[dim_hv];
  encode_hypervector(ptr_dst, ins_hdnn_reram->feature_mem, dim_feature, dim_hv);

  for (int j = 0; j < dim_hv; j++) {
    ins_hdnn_reram->class_mem[label * static_cast<uint64_t>(dim_hv) + j] +=
        ptr_dst[j];
  }

  // Program class HVs into reram array
  for (int i = 0; i < dim_hv; i++) {
    ins_hdnn_reram->program_reram_bit(
        ins_hdnn_reram->class_mem[label * static_cast<uint64_t>(dim_hv) + i] >=
            0,
        label, i);
  }

  delete[] ptr_dst;
}

int execute_inference() {
#ifdef TRACE
  fprintf(stderr, "execute_inference\n");
#endif
  uint32_t dim_hv = ins_hdnn_reram->dim_hv;
  uint32_t dim_feature = ins_hdnn_reram->dim_feature;
  uint32_t num_class = ins_hdnn_reram->num_class;

  uint32_t correct = 0;
  // Compute encoded query
  int16_t *ptr_dst = new int16_t[dim_hv];
  encode_hypervector(ptr_dst, ins_hdnn_reram->feature_mem, dim_feature, dim_hv);

  // Hamming distance
  hamming_distance(ins_hdnn_reram->score_mem, ptr_dst);
  delete[] ptr_dst;

  int16_t *ptr_hamming_score = ins_hdnn_reram->score_mem;
  int16_t *ptr_min_hamming =
      std::min_element(ptr_hamming_score, ptr_hamming_score + num_class);
  int pred = std::distance(ptr_hamming_score, ptr_min_hamming);
  return pred;
}

void execute_retrain(int label) {
#ifdef TRACE
  fprintf(stderr, "execute_retrain(%d)\n", label);
#endif
  uint32_t dim_hv = ins_hdnn_reram->dim_hv;
  uint32_t dim_feature = ins_hdnn_reram->dim_feature;
  uint32_t num_class = ins_hdnn_reram->num_class;

  // Encode features
  int16_t *ptr_dst = new int16_t[dim_hv];
  encode_hypervector(ptr_dst, ins_hdnn_reram->feature_mem, dim_feature, dim_hv);

  // Hamming distance
  hamming_distance(ins_hdnn_reram->score_mem, ptr_dst);

  int pred =
      std::distance(ins_hdnn_reram->score_mem,
                    std::min_element(ins_hdnn_reram->score_mem,
                                     ins_hdnn_reram->score_mem + num_class));

  if (pred != label) {
    ins_hdnn_reram->update_class_hvs(ptr_dst, label,
                                     static_cast<uint64_t>(dim_hv), true);
    ins_hdnn_reram->update_class_hvs(ptr_dst, pred,
                                     static_cast<uint64_t>(dim_hv), false);
  }

  // Program class HVs into reram array
  for (int i = 0; i < dim_hv; i++) {
    ins_hdnn_reram->program_reram_bit(
        ins_hdnn_reram->class_mem[label * static_cast<uint64_t>(dim_hv) + i] >=
            0,
        label, i);
  }

  delete[] ptr_dst;
}
}
