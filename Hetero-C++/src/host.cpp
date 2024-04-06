#define HPVM 1

#ifdef HPVM
#include <heterocc.h>
#include <hpvm_hdc.h>
#include "DFG.hpp"
#endif
#include "host.h"
#include <vector>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <random>
#include <cstring>

#define OFFLOAD_RP_GEN 

#define DUMP(vec, suffix) {\
  FILE *f = fopen("dump/" #vec suffix, "w");\
  if (f) fwrite(vec.data(), sizeof(vec[0]), vec.size(), f);\
  if (f) fclose(f);\
}

template <int N, typename elemTy>
void print_hv(__hypervector__<N, elemTy> hv) {
    int num_neg_one = 0;
    int num_one = 0;
    std::cout << "[";
    for (int i = 0; i < N-1; i++) {
        std::cout << hv[0][i] << ", ";
        if(hv[0][i] == 1){
            num_one ++;
        } else if(hv[0][i] == -1){
            num_neg_one ++;
        }
    }
    std::cout << hv[0][N-1] << "]\n";

    std::cout <<"# Negative 1: "<< num_neg_one <<", # Positive 1: "<< num_one << "\n";
    return;
}

void datasetBinaryRead(std::vector<int> &data, std::string path){
	std::ifstream file_(path, std::ios::in | std::ios::binary);
	assert(file_.is_open() && "Couldn't open file!");
	int32_t size;
	file_.read((char*)&size, sizeof(size));
	int32_t temp;
	for(int i = 0; i < size; i++){
		file_.read((char*)&temp, sizeof(temp));
		data.push_back(temp);
	}
	file_.close();
}


void ptr_print_hv(hvtype* ptr, int num_elems){
    std::cout << "[ ";
    for(int i = 0; i < num_elems; i++){
        std::cout <<ptr[i] <<" ";
    }
    std::cout<<"\n";
}

void datasetFloatRead(std::vector<float> &data, std::string path){
	std::ifstream file_(path, std::ios::in | std::ios::binary);
	assert(file_.is_open() && "Couldn't open file!");
	float size;
	file_.read((char*)&size, sizeof(size));
	float temp;
	for(int i = 0; i < size; i++){
		file_.read((char*)&temp, sizeof(temp));
		data.push_back(temp);
	}
	file_.close();
}

template <typename T>
T initialize_hv(T* datapoint_vector, size_t loop_index_var) {
	return datapoint_vector[loop_index_var];
}


template <typename T>
T initialize_rp(T* datapoint_vector, size_t loop_index_var) {
	return datapoint_vector[loop_index_var];
}

template <typename T>
T initialize_rp_seed(size_t loop_index_var) {
	int i = loop_index_var / 32;
	int j = loop_index_var % 32;

	//std::cout << i << " " << j << "\n";
    
	long double temp = log2(i+2.5) * pow(2, 31);
	long long int temp2 = (long long int)(temp);
	temp2 = temp2 % 2147483648;

	int ele = temp2 & (0x01 << j); //temp2 && (0x01 << j);
	if (ele) {
		return (T) 1;
	}
	else {
		return (T) -1;
	}
}

extern "C" float run_hd_classification(
	int EPOCH,
	hvtype* rp_matrix_buffer,
	hvtype* training_input_vectors,
	hvtype* inference_input_vectors,
	int* training_labels,
	int* y_test
);

int main(int argc, char** argv)
{
#ifndef NODFG
	__hpvm__init();
#endif

	auto t_start = std::chrono::high_resolution_clock::now();
	std::cout << "Main Starting" << std::endl;

	srand(time(NULL));

	int EPOCH = std::atoi(argv[1]);
   
#ifdef QUANT

    std::cout << "Quantized Dataset!\n" << "\n";

	std::vector<int> X_train; // X data. 
	std::vector<int> y_train; // LABELS
	
	datasetBinaryRead(X_train, X_train_path);
	datasetBinaryRead(y_train, y_train_path);

	std::vector<int> X_test;
	std::vector<int> y_test;
	
	datasetBinaryRead(X_test, X_test_path);
	datasetBinaryRead(y_test, y_test_path);
#else

    std::cout << "Non Quantized Dataset!\n" << "\n";
	std::vector<float> X_train; // X data. 
	std::vector<int> y_train; // LABELS
	
	datasetFloatRead(X_train, X_train_path);
	datasetBinaryRead(y_train, y_train_path);

	std::vector<float> X_test;
	std::vector<int> y_test;
	
	datasetFloatRead(X_test, X_test_path);
	datasetBinaryRead(y_test, y_test_path);

    int count = 5;
    std::cout << " First "<< count<<" values for test" <<"\n";
    for(int k =0 ; k < 5; k++){
        std:: cout <<X_test[k] <<"\n";

    }

#endif
    size_t X_train_samples = X_train.size() / N_FEAT_PAD;

    assert(X_train_samples == y_train.size() && "Incorrect number of training labels");
	std::cout << "\n" << "Read Data Starting" << std::endl;

	srand(0);
	
	assert(N_SAMPLE == y_train.size());


	assert(N_TEST == y_test.size());


	std::cout << "Training Samples: "<<N_SAMPLE<<"\n";
	std::cout << "Test Samples: "<<N_TEST<<"\n";
	
	std::vector<hvtype> temp_vec(X_train.begin(), X_train.end());
	hvtype* training_input_vectors = temp_vec.data();
        size_t training_input_size = temp_vec.size() * sizeof(hvtype);

	// N_FEAT is number of entries per vector
	size_t input_vector_size = N_FEAT * sizeof(hvtype); // Size of a single vector

	int* training_labels = y_train.data(); // Get your training labels.
	// N_SAMPLE is number of input vectors
	size_t training_labels_size = N_SAMPLE * sizeof(int);

	// INFERENCE DATA / TEST DATA
	int inference_labels[N_TEST];
	memset(inference_labels, 0xFF, sizeof(inference_labels));
	size_t inference_labels_size = N_TEST * sizeof(int);

#if 0 //hvtype == int
	int* inference_input_vectors = X_test.data();
#else
	// TRAINING DATA INITIALZIATION
	std::vector<hvtype> temp_vec2(X_test.begin(), X_test.end());
	hvtype* inference_input_vectors = temp_vec2.data();
    assert((temp_vec2.size() / N_FEAT_PAD) == N_TEST && "Incorrect number of tests");
#endif


    

	// N_FEAT is number of entries per vector



	auto t_elapsed = std::chrono::high_resolution_clock::now() - t_start;
	long mSec = std::chrono::duration_cast<std::chrono::milliseconds>(t_elapsed).count();
	long mSec1 = mSec;
	std::cout << "Reading data took " << mSec << " mSec" << std::endl;

	t_start = std::chrono::high_resolution_clock::now();

	// Encoding matrix: First we write into rp_matrix_transpose, then transpose it to get rp_matrix,
	// which is the correct dimensions for encoding input features.

	size_t rp_matrix_size = N_FEAT * Dhv * sizeof(hvtype);

	__hypervector__<Dhv, hvtype> rp_seed = __hetero_hdc_create_hypervector<Dhv, hvtype>(0, (void*) initialize_rp_seed<hvtype>);	


	std::cout << "After seed generation\n";

	// Dhv needs to be greater than N_FEAT for the orthognality to hold.
	
#ifdef OFFLOAD_RP_GEN
	hvtype* rp_matrix_buffer = new hvtype[N_FEAT * Dhv];
	hvtype* shifted_buffer = new hvtype[N_FEAT * Dhv];
	hvtype* row_buffer = new hvtype[Dhv];

#ifndef NODFG
    void* GenRPMatDAG = __hetero_launch(
        (void*) gen_rp_matrix<Dhv,  N_FEAT>,
        4,
        /* Input Buffers: 3*/ 
        &rp_seed, sizeof(hvtype) * Dhv,
        row_buffer, sizeof(hvtype) * Dhv,
        shifted_buffer, sizeof(hvtype) * (N_FEAT * Dhv),
        rp_matrix_buffer, sizeof(hvtype) * (N_FEAT * Dhv),
        /* Output Buffers: 1*/ 
        1,
        rp_matrix_buffer, sizeof(hvtype) * (N_FEAT * Dhv)
    );

    __hetero_wait(GenRPMatDAG);
#else
    gen_rp_matrix<Dhv, N_FEAT>(
        &rp_seed, sizeof(hvtype) * Dhv,
        (__hypervector__<Dhv, hvtype> *) row_buffer, sizeof(hvtype) * Dhv,
        (__hypermatrix__<N_FEAT, Dhv, hvtype> *) shifted_buffer, sizeof(hvtype) * (N_FEAT * Dhv),
        (__hypermatrix__<Dhv, N_FEAT, hvtype> *) rp_matrix_buffer, sizeof(hvtype) * (N_FEAT * Dhv)
    );
#endif

    delete[] shifted_buffer;
    delete[] row_buffer;


#else
    std::cout << "Reading RP Matrix from file" <<"\n";
    std::vector<int> rp_mat_read;
	//datasetBinaryRead(rp_mat_read, rp_matrix_path);

    std::ifstream InFile;
    InFile.open(rp_matrix_txt);
    int number;
    while(InFile >> number)
        rp_mat_read.push_back(number);

	std::vector<hvtype> temp_rp(rp_mat_read.begin(), rp_mat_read.end());

    hvtype* rp_input_vectors = temp_rp.data();
  __hypermatrix__<Dhv, N_FEAT, hvtype> rp_matrix =__hetero_hdc_create_hypermatrix<Dhv, N_FEAT, hvtype>(1, (void*) initialize_rp<hvtype>, rp_input_vectors);
    auto rp_matrix_buffer = &rp_matrix;

#endif


	float test_accuracy = run_hd_classification(
		EPOCH,
		rp_matrix_buffer,
		training_input_vectors,
		inference_input_vectors,
		training_labels,
		y_test.data()
	);


    
	t_elapsed = std::chrono::high_resolution_clock::now() - t_start;
	
	mSec = std::chrono::duration_cast<std::chrono::milliseconds>(t_elapsed).count();

	std::cout << "Overall Benchmark took " << mSec << " mSec" << std::endl;

	std::cout << "Test accuracy = " << test_accuracy << std::endl;

#ifndef NODFG
	__hpvm__cleanup();
#endif	
	return 0;
}

extern "C" float run_hd_classification(
	int EPOCH,
	hvtype* rp_matrix_buffer,
	hvtype* training_input_vectors,
	hvtype* inference_input_vectors,
	int* training_labels,
	int* y_test
) {
	/*
	std::ofstream file_rp_matrix_buffer("rp_matrix_buffer.csv");
	std::ofstream file_training_input_vectors("training_input_vectors.csv");
	std::ofstream file_inference_input_vectors("inference_input_vectors.csv");
	std::ofstream file_training_labels("training_labels.csv");
	std::ofstream file_y_test("y_test.csv");
	for (int i = 0; i < N_FEAT *  Dhv; ++i) {
		file_rp_matrix_buffer << rp_matrix_buffer[i];
		if (i + 1 < N_FEAT *  Dhv) {
			file_rp_matrix_buffer << ",";
		}
	}
	for (int i = 0; i < N_SAMPLE * N_FEAT_PAD; ++i) {
		file_training_input_vectors << training_input_vectors[i];
		if (i + 1 < N_SAMPLE * N_FEAT_PAD) {
			file_training_input_vectors << ",";
		}
	}
	for (int i = 0; i < N_TEST * N_FEAT_PAD; ++i) {
		file_inference_input_vectors << inference_input_vectors[i];
		if (i + 1 < N_TEST * N_FEAT_PAD) {
			file_inference_input_vectors << ",";
		}
	}
	for (int i = 0; i < N_SAMPLE; ++i) {
		file_training_labels << training_labels[i];
		if (i + 1 < N_SAMPLE) {
			file_training_labels << ",";
		}
	}
	for (int i = 0; i < N_TEST; ++i) {
		file_y_test << y_test[i];
		if (i + 1 < N_TEST) {
			file_y_test << ",";
		}
	}
	*/

	size_t rp_matrix_size = N_FEAT * Dhv * sizeof(hvtype);
	size_t input_vector_size = N_FEAT * sizeof(hvtype);
	size_t class_size = Dhv * sizeof(hvtype);
	size_t classes_size = N_CLASS * Dhv * sizeof(hvtype);
	size_t training_labels_size = N_SAMPLE * sizeof(int);
	size_t inference_labels_size = N_TEST * sizeof(int);
	size_t encoded_hv_size = Dhv * sizeof(hvtype);
	size_t update_hv_size = Dhv * sizeof(hvtype);
	size_t scores_size = N_CLASS * sizeof(hvtype);
	size_t norms_size = N_CLASS * sizeof(hvtype);

	__hypervector__<Dhv, hvtype> update_hv = __hetero_hdc_hypervector<Dhv, hvtype>();
	__hypermatrix__<N_CLASS, Dhv, hvtype> classes = __hetero_hdc_create_hypermatrix<N_CLASS, Dhv, hvtype>(0, (void*) zero_hv<hvtype>);

	// Static since this is too large for the stack.
	static __hypervector__<Dhv, hvtype> encoded_hv[N_SAMPLE];
	hvtype encoded_hv_buffer[Dhv];
	hvtype scores_buffer[N_CLASS];
	hvtype norms_buffer[N_CLASS];

	int inference_labels[N_TEST];
	memset(inference_labels, 0xFF, sizeof(inference_labels));

	// ============ Training ===============

	// Initialize class hvs.
	__hetero_hdc_encoding_loop(
		0, (void*) InitialEncodingDFG<Dhv, N_FEAT>,
		N_SAMPLE, N_CLASS, N_FEAT, N_FEAT_PAD,
		rp_matrix_buffer, rp_matrix_size,
		training_input_vectors, input_vector_size,
		encoded_hv, class_size
	);

	for (int i = 0; i < N_SAMPLE; i++) {
		int label = training_labels[i];
		update_hv =  __hetero_hdc_get_matrix_row<N_CLASS, Dhv, hvtype>(classes, N_CLASS, Dhv, label);
		update_hv = __hetero_hdc_sum<Dhv, hvtype>(update_hv, encoded_hv[i]); 
		__hetero_hdc_set_matrix_row<N_CLASS, Dhv, hvtype>(classes, update_hv, label); 
	}

	int argmax[1];
	// Training generates classes from labeled data. 
	// ======= Training Rest Epochs ======= 

	{
	auto t_start = std::chrono::high_resolution_clock::now();
	__hetero_hdc_training_loop(
		22, (void*) training_root_node<Dhv, N_CLASS, N_SAMPLE, N_FEAT>,
		EPOCH, N_SAMPLE, N_FEAT, N_FEAT_PAD,
		rp_matrix_buffer, rp_matrix_size,
		training_input_vectors, input_vector_size,
		&classes, classes_size,
		training_labels, training_labels_size,
		encoded_hv_buffer, encoded_hv_size,
		scores_buffer, scores_size,
		norms_buffer, norms_size,
		&update_hv, update_hv_size,
		&argmax[0], sizeof(int)
	);
	auto t_end = std::chrono::high_resolution_clock::now();
	long mSec = std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count();
	std::cout << "Training: " << mSec << " mSec\n";
	}

	// ============ Inference =============== //

	{
	auto t_start = std::chrono::high_resolution_clock::now();
	__hetero_hdc_inference_loop(17, (void*) inference_root_node<Dhv, N_CLASS, N_TEST, N_FEAT>,
		N_TEST, N_FEAT, N_FEAT_PAD,
		rp_matrix_buffer, rp_matrix_size,
		inference_input_vectors, input_vector_size,
		&classes, classes_size,
		inference_labels, inference_labels_size,
		encoded_hv_buffer, encoded_hv_size,
		scores_buffer, scores_size,
		norms_buffer, norms_size
	);
	auto t_end = std::chrono::high_resolution_clock::now();
	long mSec = std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count();
	std::cout << "Inference: " << mSec << " mSec\n";
	}

	//std::ofstream myfile("out.txt");

	int correct = 0;
	for(int i = 0; i < N_TEST; i++) {
		//myfile << y_test[i] << " " << inference_labels[i] << std::endl;
		if(inference_labels[i] == y_test[i])
			correct += 1;
	}

	return float(correct)/N_TEST;
}
