#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <iterator>
#include <time.h>
#include <chrono>
#include <vector>
#include <unistd.h>
#include <iostream>
#include <fstream>

#define N_FEAT	        617 //feature per input (e.g., isolet: 617)
#define N_CLASS		    26	//number of classes. (e.g., isolet: 26, ucihar 12)
#define Dhv				2048  //hypervectors length
#define N_SAMPLE 		6238 // FIXME: Make these parametesr variable.

//#define N_TEST 			6238// Needs to be constant for templated funciton. 
#define N_TEST 			1559  // Needs to be constant for templated funciton. 

#define COL				8 //number of columns of a matrix-vector multiplication window
#define ROW				32 //number of rows of a matrix-vector multiplication window (32, 64, 128, 256, 512)

#define PAD_			(N_FEAT & (COL - 1))

#if PAD_ == 0
	#define PAD 		0
#else
	#define PAD 		(COL - PAD_)
#endif

#define N_FEAT_PAD		(N_FEAT + PAD)	//feature per input (e.g., isolet: 624, ucihar 568)

//#define QUANT

#ifdef QUANT
// TODO: Add these to directory.
std::string X_train_path = "../dataset/isolet_trainX.bin";
std::string  y_train_path = "../dataset/isolet_trainY.bin";
std::string  X_test_path = "../dataset/isolet_testX.bin";
std::string  y_test_path = "../dataset/isolet_testY.bin";

#else

std::string X_train_path = "../dataset/isolet_nq_trainX.bin";
std::string  y_train_path = "../dataset/isolet_nq_trainY.bin";
std::string  X_test_path = "../dataset/isolet_nq_testX.bin";
std::string  y_test_path = "../dataset/isolet_nq_testY.bin";

#endif
