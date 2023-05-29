#include "Tensor.h"


template<typename T, int DIMENSION_COUNT, typename T2, int DIMENSION_COUNT2, typename T3>
void _matmul(Tensor<DIMENSION_COUNT, T> x1, Tensor<DIMENSION_COUNT2, T2> x2, Tensor<DIMENSION_COUNT2, T3> x3){
	for(std::size_t i=0; i<x2.dimensions[0]; i++){
		//dimension reduction (1:N broadcast)
		_matmul(x1, x2.slice(i), x3.slice(i));
	}
}

template<typename T, int DIMENSION_COUNT, typename T2, int DIMENSION_COUNT2, typename T3>
void _matmul(Tensor<DIMENSION_COUNT, T> x1, Tensor<DIMENSION_COUNT2, T2> x2, Tensor<DIMENSION_COUNT, T3> x3){
	for(std::size_t i=0; i<x1.dimensions[0]; i++){
		//dimension reduction (1:N broadcast)
		_matmul(x1.slice(i), x2, x3.slice(i));
	}
}
template<typename T, int DIMENSION_COUNT, typename T2, typename T3>
void _matmul(Tensor<DIMENSION_COUNT, T> x1, Tensor<DIMENSION_COUNT, T2> x2, Tensor<DIMENSION_COUNT, T3> x3){
	for(std::size_t i=0; i<x1.dimensions[0]; i++){
		_matmul(x1.slice(i), x2.slice(i), x3.slice(i));
	}
}
// 2x2 dimension matmul
template<typename T, typename T2, typename T3>
void _matmul(Tensor<2, T> x1, Tensor<2, T2> x2, Tensor<2, T3> x3){
	//actual mat mul
	for(std::size_t i = 0;i < x1.dimensions[0]; i++){
		Tensor<1, T> x1_1 = x1.slice(i);
		Tensor<1, T3> x3_1 = x3.slice(i);
		for(std::size_t j = 0;j < x2.dimensions[1]; j++){
			x3_1[{j}]=0;
			Tensor<1, T2> x2_2 = x2.slice(j, 1);
			for(std::size_t k = 0; k < x1.dimensions[1]; k++){
				x3_1[{j}] += x1_1[{k}]*x2_2[{k}];
			}
		}
	}
}
template<typename T, int DIMENSION_COUNT, typename T2, int DIMENSION_COUNT2, typename T3, int DIMENSION_COUNT3>
void matmul(Tensor<DIMENSION_COUNT, T> x1, Tensor<DIMENSION_COUNT2, T2> x2, Tensor<DIMENSION_COUNT3, T3> x3){
	
	//checking non-broadcasting dimension matches
	const std::size_t minIDim = (DIMENSION_COUNT<DIMENSION_COUNT2 ? DIMENSION_COUNT : DIMENSION_COUNT2);
	const std::size_t maxIDim = (DIMENSION_COUNT<DIMENSION_COUNT2 ? DIMENSION_COUNT2 : DIMENSION_COUNT);
	if(maxIDim != DIMENSION_COUNT3)throw std::invalid_argument("Output array does not match dimensions");
	for(std::size_t i = 2; i < minIDim; i++){
		if(x1.dimensions[i]!=x2.dimensions[i])throw std::invalid_argument("Dimensions do not match for non-broadcasting indices");
	}
	if(DIMENSION_COUNT>DIMENSION_COUNT2){
		for(std::size_t i = 2; i < DIMENSION_COUNT; i++){
			if(x1.dimensions[i] != x3.dimensions[i]) throw std::invalid_argument("Output array dimension mismatch");
		}
	}else{
		for(std::size_t i = 2; i < DIMENSION_COUNT2; i++){
			if(x2.dimensions[i] != x3.dimensions[i]) throw std::invalid_argument("Output array dimension mismatch");
		}
	}
	
	//expand all matrices in case 1-dims exist
	_matmul(x1, x2, x3);
	
}

template<typename T, typename T2, int DIMENSION_COUNT2, typename T3, int DIMENSION_COUNT3>
void matmul(Tensor<1, T> x1, Tensor<DIMENSION_COUNT2, T2> x2, Tensor<DIMENSION_COUNT3, T3> x3){
	matmul(x1.expand(), x2, x3);
}

template<typename T, int DIMENSION_COUNT, typename T2, typename T3, int DIMENSION_COUNT3>
void matmul(Tensor<DIMENSION_COUNT, T> x1, Tensor<1, T2> x2, Tensor<DIMENSION_COUNT3, T3> x3){
	matmul(x1, x2.expand(), x3);
}

template<typename T, int DIMENSION_COUNT, typename T2, int DIMENSION_COUNT2, typename T3>
void matmul(Tensor<DIMENSION_COUNT, T> x1, Tensor<DIMENSION_COUNT2, T2> x2, Tensor<1, T3> x3){
	matmul(x1, x2, x3.expand());
}

template<typename T, int DIMENSION_COUNT, typename T2, typename T3>
void matmul(Tensor<DIMENSION_COUNT, T> x1, Tensor<1, T2> x2, Tensor<1, T3> x3){
	matmul(x1, x2.expand(), x3.expand());
}

template<typename T, typename T2, int DIMENSION_COUNT2, typename T3>
void matmul(Tensor<1, T> x1, Tensor<DIMENSION_COUNT2, T2> x2, Tensor<1, T3> x3){
	matmul(x1.expand(), x2, x3.expand());
}

template<typename T, typename T2, typename T3, int DIMENSION_COUNT3>
void matmul(Tensor<1, T> x1, Tensor<1, T2> x2, Tensor<DIMENSION_COUNT3, T3> x3){
	matmul(x1.expand(), x2.expand(), x3);
}

template<typename T, typename T2, typename T3>
void matmul(Tensor<1, T> x1, Tensor<1, T2> x2, Tensor<1, T3> x3){
	matmul(x1.expand(), x2.expand(), x3.expand());
}
